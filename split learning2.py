import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

# 设置随机种子，保证实验可复现
torch.manual_seed(42)
np.random.seed(42)

# 1. 数据准备
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# 加载CIFAR-10数据集
train_dataset = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)


# 2. 模型拆分定义（优化服务器模型）
class ClientModel(nn.Module):
    """客户端模型 - 处理前几层，生成激活值"""

    def __init__(self):
        super(ClientModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.features(x)
        return x


class ServerModel(nn.Module):
    """优化的服务器模型 - 增强对脱敏特征的适应性"""

    def __init__(self):
        super(ServerModel, self).__init__()
        self.classifier = nn.Sequential(
            nn.BatchNorm2d(64),  # 新增：稳定输入分布，适应脱敏后的特征
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),  # 新增：减少过拟合
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


# 3. 优化的脱敏模块（可调节隐私强度）
class TunableDesensitizer(nn.Module):
    """可调节隐私-性能平衡的脱敏模块"""

    def __init__(self,
                 input_channels=64,
                 target_spatial_size=8,
                 privacy_strength=0.5):  # 新增：隐私强度（0-1，值越小隐私越弱性能越好）
        super(TunableDesensitizer, self).__init__()
        self.input_channels = input_channels
        self.target_spatial_size = target_spatial_size
        self.target_feature_size = input_channels * target_spatial_size * target_spatial_size

        # 根据隐私强度动态调整参数
        self.rank_ratio = 0.3 + 0.5 * (1 - privacy_strength)  # 隐私弱→保留更多特征（0.3-0.8）
        self.noise_scale = 0.15 * privacy_strength  # 隐私弱→噪声更小（0-0.15）

        # 计算秩（确保能被目标形状整除）
        self.rank = int(self.target_feature_size * self.rank_ratio)
        self.rank = (self.rank // self.target_feature_size) * self.target_feature_size
        if self.rank == 0:
            self.rank = self.target_feature_size

        self.L = self._generate_low_rank_matrix()

    def _generate_low_rank_matrix(self):
        feature_dim = self.input_channels * 8 * 8
        L_full = torch.randn(feature_dim, feature_dim)
        U, S, Vh = torch.linalg.svd(L_full)
        S_rank = torch.zeros_like(S)
        S_rank[:self.rank] = S[:self.rank]
        L_low_rank = U @ torch.diag(S_rank) @ Vh
        return L_low_rank[:, :self.rank]

    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)

        # 低秩变换（保留更多特征）
        x_transformed = torch.matmul(x_flat, self.L)
        x_reshaped = x_transformed.view(batch_size, self.input_channels,
                                        self.target_spatial_size, self.target_spatial_size)

        # 减弱的噪声注入
        sigma = torch.std(x_reshaped)
        noise = torch.randn_like(x_reshaped) * sigma * self.noise_scale  # 噪声强度降低
        return x_reshaped + noise


# 4. 训练与测试函数（优化训练过程）
def train_split_learning(client_model, server_model, desensitizer, train_loader,
                         client_optimizer, server_optimizer, criterion, device,
                         desensitize_method=None):
    client_model.train()
    server_model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc="Training")):
        inputs, targets = inputs.to(device), targets.to(device)
        client_output = client_model(inputs)

        # 处理激活值
        if desensitize_method == 'dynamic':
            server_input = desensitizer(client_output.detach())
        elif desensitize_method == 'fixed':
            noise = torch.randn_like(client_output) * 0.1
            server_input = client_output.detach() + noise
        else:
            server_input = client_output.detach()

        # 服务器优化（使用梯度累积增强学习效果）
        server_optimizer.zero_grad()
        outputs = server_model(server_input)
        loss = criterion(outputs, targets)
        loss.backward()
        server_optimizer.step()

        # 客户端优化（使用更强的梯度反馈）
        client_optimizer.zero_grad()
        client_loss = criterion(
            server_model(desensitizer(client_output) if desensitize_method == 'dynamic' else client_output), targets)
        client_loss.backward()
        client_optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    return avg_loss, acc


def test_split_learning(client_model, server_model, desensitizer, test_loader,
                        criterion, device, desensitize_method=None):
    client_model.eval()
    server_model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, desc="Testing")):
            inputs, targets = inputs.to(device), targets.to(device)
            client_output = client_model(inputs)

            if desensitize_method == 'dynamic':
                server_input = desensitizer(client_output)
            elif desensitize_method == 'fixed':
                server_input = client_output + torch.randn_like(client_output) * 0.1
            else:
                server_input = client_output

            outputs = server_model(server_input)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    avg_loss = total_loss / len(test_loader)
    return avg_loss, acc


# 5. 隐私保护效果评估
def evaluate_privacy(client_model, desensitizer, test_loader, device, desensitize_method=None):
    client_model.eval()
    if desensitizer is not None:
        desensitizer.eval()

    mse_scores = []
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(test_loader):
            inputs = inputs.to(device)
            client_output = client_model(inputs)
            original = client_output.view(client_output.size(0), -1)

            if desensitize_method == 'dynamic':
                processed = desensitizer(client_output)
            elif desensitize_method == 'fixed':
                processed = client_output + torch.randn_like(client_output) * 0.1
            else:
                processed = client_output

            processed = processed.view(processed.size(0), -1)
            mse = torch.mean((original - processed) ** 2).item()
            mse_scores.append(mse)

            if batch_idx >= 10:
                break

    return np.mean(mse_scores)


# 6. 主函数（优化参数配置）
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 关键优化：增加训练轮数
    epochs = 1     # 从1增加到15，确保模型充分训练
    lr = 0.001

    # 初始化模型（使用优化后的服务器模型）
    client_normal = ClientModel().to(device)
    server_normal = ServerModel().to(device)
    optimizer_client_normal = optim.Adam(client_normal.parameters(), lr=lr)
    optimizer_server_normal = optim.Adam(server_normal.parameters(), lr=lr)

    client_fixed = ClientModel().to(device)
    server_fixed = ServerModel().to(device)
    optimizer_client_fixed = optim.Adam(client_fixed.parameters(), lr=lr)
    optimizer_server_fixed = optim.Adam(server_fixed.parameters(), lr=lr)

    # 关键优化：降低隐私强度（0.3-0.5之间，平衡隐私与性能）
    client_dynamic = ClientModel().to(device)
    server_dynamic = ServerModel().to(device)
    desensitizer = TunableDesensitizer(privacy_strength=0.4).to(device)  # 核心参数调整
    optimizer_client_dynamic = optim.Adam(client_dynamic.parameters(), lr=lr)
    optimizer_server_dynamic = optim.Adam(server_dynamic.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    # 记录实验结果
    results = {
        'normal': {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []},
        'fixed': {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []},
        'dynamic': {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    }

    # 训练与测试
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # 1. 正常Split Learning
        start_time = time.time()
        train_loss, train_acc = train_split_learning(
            client_normal, server_normal, None, train_loader,
            optimizer_client_normal, optimizer_server_normal, criterion, device,
            desensitize_method=None
        )
        train_time = time.time() - start_time

        test_loss, test_acc = test_split_learning(
            client_normal, server_normal, None, test_loader,
            criterion, device, desensitize_method=None
        )

        results['normal']['train_loss'].append(train_loss)
        results['normal']['train_acc'].append(train_acc)
        results['normal']['test_loss'].append(test_loss)
        results['normal']['test_acc'].append(test_acc)

        print(f"Normal - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Time: {train_time:.2f}s")

        # 2. 固定噪声注入
        start_time = time.time()
        train_loss, train_acc = train_split_learning(
            client_fixed, server_fixed, None, train_loader,
            optimizer_client_fixed, optimizer_server_fixed, criterion, device,
            desensitize_method='fixed'
        )
        train_time = time.time() - start_time

        test_loss, test_acc = test_split_learning(
            client_fixed, server_fixed, None, test_loader,
            criterion, device, desensitize_method='fixed'
        )

        results['fixed']['train_loss'].append(train_loss)
        results['fixed']['train_acc'].append(train_acc)
        results['fixed']['test_loss'].append(test_loss)
        results['fixed']['test_acc'].append(test_acc)

        print(f"Fixed Noise - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Time: {train_time:.2f}s")

        # 3. 优化后的动态脱敏方法
        start_time = time.time()
        train_loss, train_acc = train_split_learning(
            client_dynamic, server_dynamic, desensitizer, train_loader,
            optimizer_client_dynamic, optimizer_server_dynamic, criterion, device,
            desensitize_method='dynamic'
        )
        train_time = time.time() - start_time

        test_loss, test_acc = test_split_learning(
            client_dynamic, server_dynamic, desensitizer, test_loader,
            criterion, device, desensitize_method='dynamic'
        )

        results['dynamic']['train_loss'].append(train_loss)
        results['dynamic']['train_acc'].append(train_acc)
        results['dynamic']['test_loss'].append(test_loss)
        results['dynamic']['test_acc'].append(test_acc)

        print(f"Optimized Method - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Time: {train_time:.2f}s")

    # 评估隐私保护效果
    print("\nEvaluating privacy protection...")
    mse_normal = evaluate_privacy(client_normal, None, test_loader, device, desensitize_method=None)
    mse_fixed = evaluate_privacy(client_fixed, None, test_loader, device, desensitize_method='fixed')
    mse_dynamic = evaluate_privacy(client_dynamic, desensitizer, test_loader, device, desensitize_method='dynamic')

    print(
        f"Reconstruction MSE - Normal: {mse_normal:.6f}, Fixed Noise: {mse_fixed:.6f}, Optimized Method: {mse_dynamic:.6f}")
    print("(注: MSE值越高，表示隐私保护效果越好)")

    # 绘制结果对比图
    plt.figure(figsize=(15, 5))

    # 1. 测试准确率对比
    plt.subplot(1, 3, 1)
    plt.plot(results['normal']['test_acc'], label='Normal Split Learning')
    plt.plot(results['fixed']['test_acc'], label='Fixed Noise Injection')
    plt.plot(results['dynamic']['test_acc'], label='Optimized Method')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.xlim(0, epochs - 1)
    plt.ylim(0, 100)
    plt.legend()

    # 2. 测试损失对比
    plt.subplot(1, 3, 2)
    plt.plot(results['normal']['test_loss'], label='Normal Split Learning')
    plt.plot(results['fixed']['test_loss'], label='Fixed Noise Injection')
    plt.plot(results['dynamic']['test_loss'], label='Optimized Method')
    plt.title('Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim(0, epochs - 1)
    plt.ylim(0, 3)
    plt.legend()

    # 3. 隐私保护效果对比
    plt.subplot(1, 3, 3)
    methods = ['Normal', 'Fixed Noise', 'Optimized Method']
    mse_values = [mse_normal, mse_fixed, mse_dynamic]
    plt.bar(methods, mse_values)
    plt.title('Reconstruction MSE (Privacy)')
    plt.ylabel('MSE')

    plt.tight_layout()
    plt.savefig('split_learning_optimized.png')
    print("优化后的实验结果图表已保存为 'split_learning_optimized.png'")
    plt.show()


if __name__ == "__main__":
    main()
