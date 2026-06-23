import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 设置随机种子确保可复现性
torch.manual_seed(42)
np.random.seed(42)


# 1. 数据加载与预处理
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    return train_loader, test_loader


# 2. 拆分学习模型定义
class ClientModel(nn.Module):
    def __init__(self):
        super(ClientModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        return self.features(x)  # Output shape: [batch_size, 64, 8, 8]


class ServerModel(nn.Module):
    def __init__(self):
        super(ServerModel, self).__init__()
        self.classifier = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.classifier(x)


# 3. 动态脱敏模块（彻底修复维度匹配问题）
class TunableDesensitizer(nn.Module):
    def __init__(self, privacy_strength=0.5):
        super(TunableDesensitizer, self).__init__()
        self.privacy_strength = privacy_strength
        self.target_shape = (64, 8, 8)  # Target feature shape: [channels, height, width]
        self.target_total = self.target_shape[0] * self.target_shape[1] * self.target_shape[2]  # 64*8*8=4096

        # Dynamically adjust parameters based on privacy strength
        self.rank_ratio = 0.3 + 0.5 * (1 - privacy_strength)  # Feature retention ratio
        self.noise_scale = 0.15 * privacy_strength  # Noise intensity

        # Initialize parameters related to low-rank matrix
        self.feature_dim = None  # Total dimension of input features (flattened output from client)
        self.rank = None  # Rank of the low-rank matrix
        self.L = None  # Low-rank projection matrix

    def _generate_low_rank_matrix(self, feature_dim):
        """Generate a low-rank matrix with controllable rank and ensuring dimension matching"""
        self.feature_dim = feature_dim

        # Calculate rank and ensure it is divisible by the total target dimension (core fix)
        raw_rank = int(feature_dim * self.rank_ratio)
        self.rank = (raw_rank // self.target_total) * self.target_total  # Ensure rank is an integer multiple of target total
        self.rank = max(self.target_total, self.rank)  # Minimum rank is target total to avoid insufficient dimensions

        # Generate low-rank matrix (SVD decomposition)
        L_full = torch.randn(feature_dim, feature_dim, device=self.device)
        U, S, Vh = torch.linalg.svd(L_full)
        S_rank = torch.zeros_like(S)
        S_rank[:self.rank] = S[:self.rank]  # Retain the first 'rank' singular values
        L_low_rank = U @ torch.diag(S_rank) @ Vh
        return L_low_rank[:, :self.rank]  # Output shape: [feature_dim, rank]

    def forward(self, x):
        # Record device information (ensure matrices are on the same device)
        self.device = x.device

        # 1. Calculate total dimension of input features
        batch_size = x.size(0)
        feature_dim = x.size(1) * x.size(2) * x.size(3)  # Flattened dimension of client output: 64*8*8=4096

        # 2. Dynamically generate low-rank matrix (first call or when dimension changes)
        if self.L is None or self.feature_dim != feature_dim:
            self.L = self._generate_low_rank_matrix(feature_dim)

        # 3. Low-rank transformation
        x_flat = x.view(batch_size, -1)  # Flatten to: [batch_size, feature_dim]
        x_transformed = torch.matmul(x_flat, self.L)  # After low-rank transformation: [batch_size, rank]

        # 4. Reshape to target shape (core fix: ensure total number of elements match)
        x_reshaped = x_transformed.view(batch_size, *self.target_shape)  # [batch, 64, 8, 8]

        # 5. Dynamic noise injection
        sigma = torch.std(x_reshaped)
        noise = torch.randn_like(x_reshaped) * sigma * self.noise_scale
        return x_reshaped + noise


# 4. Training and testing functions
def train_model(client, server, desensitizer, train_loader, client_opt, server_opt, criterion, device,
                privacy_strength):
    client.train()
    server.train()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, targets in tqdm(train_loader, desc=f"Training (strength={privacy_strength})"):
        inputs, targets = inputs.to(device), targets.to(device)

        # Client processing
        client_out = client(inputs)  # Output: [batch, 64, 8, 8]

        # Desensitization processing
        if desensitizer:
            server_in = desensitizer(client_out.detach())
        else:
            server_in = client_out.detach()

        # Server training
        server_opt.zero_grad()
        server_out = server(server_in)
        loss = criterion(server_out, targets)
        loss.backward()
        server_opt.step()

        # Client training
        client_opt.zero_grad()
        if desensitizer:
            client_loss = criterion(server(desensitizer(client_out)), targets)
        else:
            client_loss = criterion(server(client_out), targets)
        client_loss.backward()
        client_opt.step()

        # Record metrics
        total_loss += loss.item()
        _, pred = server_out.max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()

    return total_loss / len(train_loader), 100. * correct / total


def test_model(client, server, desensitizer, test_loader, criterion, device):
    client.eval()
    server.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            client_out = client(inputs)

            if desensitizer:
                server_in = desensitizer(client_out)
            else:
                server_in = client_out

            server_out = server(server_in)
            loss = criterion(server_out, targets)

            total_loss += loss.item()
            _, pred = server_out.max(1)
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()

    return total_loss / len(test_loader), 100. * correct / total


# 5. Privacy evaluation function
def evaluate_privacy_leakage(client, desensitizer, test_loader, device):
    client.eval()
    if desensitizer:
        desensitizer.eval()

    mse_scores = []
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(test_loader):
            if batch_idx >= 10:  # Only use first 10 batches for accelerated evaluation
                break

            inputs = inputs.to(device)
            client_out = client(inputs)
            original = client_out.view(client_out.size(0), -1)

            if desensitizer:
                processed = desensitizer(client_out)
            else:
                processed = client_out

            processed = processed.view(processed.size(0), -1)
            mse = torch.mean((original - processed) ** 2).item()
            mse_scores.append(mse)

    return np.mean(mse_scores)


# 6. Sensitivity analysis main function
def sensitivity_analysis():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    train_loader, test_loader = load_data()

    # Experimental parameters
    epochs = 10
    lr = 0.001
    privacy_strengths = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    criterion = nn.CrossEntropyLoss()

    # Store results
    results = {
        'privacy_strength': [],
        'test_accuracy': [],
        'test_loss': [],
        'privacy_mse': [],
        'rank_ratio': [],
        'noise_scale': []
    }

    # Baseline model (no privacy protection)
    print("\n===== Training baseline model (no privacy protection) =====")
    client_base = ClientModel().to(device)
    server_base = ServerModel().to(device)
    opt_client_base = optim.Adam(client_base.parameters(), lr=lr)
    opt_server_base = optim.Adam(server_base.parameters(), lr=lr)

    for epoch in range(epochs):
        train_loss, train_acc = train_model(
            client_base, server_base, None, train_loader,
            opt_client_base, opt_server_base, criterion, device, 0.0
        )
        test_loss, test_acc = test_model(
            client_base, server_base, None, test_loader, criterion, device
        )
        print(f"Epoch {epoch + 1}/{epochs} - Test Accuracy: {test_acc:.2f}%")

    base_mse = evaluate_privacy_leakage(client_base, None, test_loader, device)
    print(f"Baseline model - Final Test Accuracy: {test_acc:.2f}%, Privacy MSE: {base_mse:.6f}")

    # Test different privacy strengths
    for strength in privacy_strengths:
        print(f"\n===== Testing privacy strength: {strength} =====")

        # Initialize models
        client = ClientModel().to(device)
        server = ServerModel().to(device)
        desensitizer = TunableDesensitizer(privacy_strength=strength).to(device)

        # Optimizers
        opt_client = optim.Adam(client.parameters(), lr=lr)
        opt_server = optim.Adam(server.parameters(), lr=lr)

        # Train models
        for epoch in range(epochs):
            train_loss, train_acc = train_model(
                client, server, desensitizer, train_loader,
                opt_client, opt_server, criterion, device, strength
            )
            test_loss, test_acc = test_model(
                client, server, desensitizer, test_loader, criterion, device
            )
            print(f"Epoch {epoch + 1}/{epochs} - Test Accuracy: {test_acc:.2f}%")

        # Evaluate privacy leakage
        mse = evaluate_privacy_leakage(client, desensitizer, test_loader, device)

        # Save results
        results['privacy_strength'].append(strength)
        results['test_accuracy'].append(test_acc)
        results['test_loss'].append(test_loss)
        results['privacy_mse'].append(mse)
        results['rank_ratio'].append(desensitizer.rank_ratio)
        results['noise_scale'].append(desensitizer.noise_scale)

        print(f"Privacy strength {strength} - Final Accuracy: {test_acc:.2f}%, Privacy MSE: {mse:.6f}")

    # Visualize results
    plt.figure(figsize=(15, 10))

    # 1. Privacy strength vs Test accuracy
    plt.subplot(2, 2, 1)
    plt.plot(results['privacy_strength'], results['test_accuracy'], 'o-', label='Dynamic Desensitization')
    plt.axhline(y=test_acc, color='r', linestyle='--', label='Baseline Model')
    plt.xlabel('Privacy Strength')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Impact of Privacy Strength on Test Accuracy')
    plt.xlim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)

    # 2. Privacy strength vs Privacy MSE
    plt.subplot(2, 2, 2)
    plt.plot(results['privacy_strength'], results['privacy_mse'], 'o-', color='g', label='Dynamic Desensitization')
    plt.axhline(y=base_mse, color='r', linestyle='--', label='Baseline Model')
    plt.xlabel('Privacy Strength')
    plt.ylabel('Privacy Leakage MSE (Higher is better for privacy)')
    plt.title('Impact of Privacy Strength on Privacy Protection')
    plt.xlim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)

    # 3. Privacy strength vs Internal parameters
    plt.subplot(2, 2, 3)
    plt.plot(results['privacy_strength'], results['rank_ratio'], 'o-', label='Feature Retention Ratio')
    plt.plot(results['privacy_strength'], results['noise_scale'], 's-', color='orange', label='Noise Scale Ratio')
    plt.xlabel('Privacy Strength')
    plt.ylabel('Parameter Value')
    plt.title('Impact of Privacy Strength on Internal Parameters')
    plt.xlim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)

    # 4. Trade-off between Test accuracy and Privacy MSE
    plt.subplot(2, 2, 4)
    plt.scatter(results['privacy_mse'], results['test_accuracy'],
                c=results['privacy_strength'], cmap='viridis', s=80)
    plt.colorbar(label='Privacy Strength')
    plt.scatter(base_mse, test_acc, color='r', marker='*', s=150, label='Baseline Model')
    plt.xlabel('Privacy Leakage MSE')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Privacy-Performance Trade-off')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('privacy_strength_analysis_results.png', dpi=300)
    print("\nAnalysis results saved as 'privacy_strength_analysis_results.png'")
    plt.show()


if __name__ == "__main__":
    sensitivity_analysis()