import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import re


# 设置随机种子确保可复现性
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


set_seed()


# 1. 数据集加载器（保持不变）
class FlexibleDataset(Dataset):
    def __init__(self, image_folder, noise_folder, transform=None, match_strategy="exact"):
        self.image_folder = image_folder
        self.noise_folder = noise_folder
        self.transform = transform
        self.match_strategy = match_strategy

        # 获取所有图像文件
        self.image_files = [
            f for f in os.listdir(image_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]

        if not self.image_files:
            raise ValueError(f"错误：在 {image_folder} 中未找到任何图像文件")

        # 获取所有噪声文件
        self.noise_files = [
            f for f in os.listdir(noise_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]

        if not self.noise_files:
            raise ValueError(f"错误：在 {noise_folder} 中未找到任何噪声文件")

        # 尝试匹配文件对
        self.matched_pairs = self._match_files()

        if not self.matched_pairs:
            # 最后尝试：使用文件顺序匹配
            min_count = min(len(self.image_files), len(self.noise_files))
            self.matched_pairs = [
                (self.image_files[i], self.noise_files[i])
                for i in range(min_count)
            ]
            print(f"警告：使用顺序匹配，共匹配 {len(self.matched_pairs)} 对文件（可能不准确）")
        else:
            print(f"成功匹配 {len(self.matched_pairs)} 对图像-噪声文件")

        # 生成标签
        self.labels = [hash(img_file) % 5 for img_file, _ in self.matched_pairs]
        self.num_classes = len(set(self.labels))

    def _match_files(self):
        matched = []

        if self.match_strategy == "exact":
            for img_file in self.image_files:
                if img_file in self.noise_files:
                    matched.append((img_file, img_file))

        elif self.match_strategy == "prefix":
            for img_file in self.image_files:
                img_name = os.path.splitext(img_file)[0]
                for noise_file in self.noise_files:
                    if noise_file.startswith(img_name):
                        matched.append((img_file, noise_file))
                        break

        elif self.match_strategy == "number":
            def extract_numbers(filename):
                nums = re.findall(r'\d+', filename)
                return tuple(nums) if nums else None

            noise_num_index = {}
            for noise_file in self.noise_files:
                nums = extract_numbers(noise_file)
                if nums:
                    noise_num_index[nums] = noise_file

            for img_file in self.image_files:
                nums = extract_numbers(img_file)
                if nums and nums in noise_num_index:
                    matched.append((img_file, noise_num_index[nums]))

        return matched

    def __len__(self):
        return len(self.matched_pairs)

    def __getitem__(self, idx):
        img_file, noise_file = self.matched_pairs[idx]
        img_path = os.path.join(self.image_folder, img_file)
        noise_path = os.path.join(self.noise_folder, noise_file)

        try:
            image = Image.open(img_path).convert('RGB')
            noise = Image.open(noise_path).convert('RGB')
        except Exception as e:
            print(f"加载文件对失败 {img_file} 和 {noise_file}: {e}")
            return torch.zeros(3, 64, 64), torch.zeros(3, 64, 64), torch.zeros(3, 64, 64), 0

        if self.transform:
            image = self.transform(image)
            noise = self.transform(noise)

        # 处理噪声
        noise = (noise - 0.5) * 0.5
        noisy_image = torch.clamp(image + noise, 0.0, 1.0)

        return image, noisy_image, noise, self.labels[idx]


# 2. 模型保持不变
class ImageModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# 3. 兼容低版本PyTorch的损失函数（不使用flatten参数）
class ImprovedDualLoss(nn.Module):
    def __init__(self, pred_weight=10.0, reg_weight=5.0):
        super().__init__()
        self.pred_weight = pred_weight
        self.reg_weight = reg_weight
        self.base_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets, noise):
        # 预测损失项
        pred_loss = self.pred_weight * self.base_loss(outputs, targets)

        # 兼容低版本的噪声范数计算：手动展平
        # 噪声形状是 (batch_size, 3, 64, 64)，展平为 (batch_size, 3*64*64)
        batch_size = noise.size(0)
        noise_flat = noise.view(batch_size, -1)  # 手动展平
        noise_magnitude = torch.mean(torch.norm(noise_flat, p=2, dim=1))  # 对展平后的第二维计算范数

        # 输出敏感性
        output_sensitivity = torch.mean(torch.var(outputs, dim=1))

        # 正则化损失
        reg_loss = self.reg_weight * noise_magnitude * output_sensitivity

        return pred_loss + reg_loss


# 4. 训练函数（同样修复范数计算）
def run_experiment(
        image_folder,
        noise_folder,
        exp_name,
        use_pred_loss=True,
        use_reg_loss=True,
        epochs=3,
        batch_size=16
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"实验 [{exp_name}] 使用设备: {device}")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 尝试不同的匹配策略
    match_strategies = ["exact", "prefix", "number"]
    dataset = None

    for strategy in match_strategies:
        try:
            dataset = FlexibleDataset(
                image_folder=image_folder,
                noise_folder=noise_folder,
                transform=transform,
                match_strategy=strategy
            )
            if len(dataset) > 0:
                print(f"使用匹配策略: {strategy}")
                break
        except Exception as e:
            continue

    if dataset is None or len(dataset) == 0:
        raise ValueError("无法匹配任何图像-噪声文件对，请检查文件命名")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    model = ImageModel(num_classes=dataset.num_classes).to(device)
    criterion = ImprovedDualLoss(pred_weight=10.0, reg_weight=5.0)
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    metrics = {
        "loss": [],
        "accuracy": [],
        "noise_impact": []
    }

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_impact = []

        for _, noisy_img, noise, targets in tqdm(
                dataloader, desc=f"Epoch {epoch + 1}/{epochs}"
        ):
            noisy_img = noisy_img.to(device)
            noise = noise.to(device)
            targets = targets.to(device)

            outputs = model(noisy_img)

            if use_pred_loss and use_reg_loss:
                loss = criterion(outputs, targets, noise)
            elif use_pred_loss:
                loss = criterion.pred_weight * criterion.base_loss(outputs, targets)
            elif use_reg_loss:
                # 手动展平计算范数（兼容低版本）
                batch_size = noise.size(0)
                noise_flat = noise.view(batch_size, -1)
                noise_magnitude = torch.mean(torch.norm(noise_flat, p=2, dim=1))
                output_sensitivity = torch.mean(torch.var(outputs, dim=1))
                loss = criterion.reg_weight * noise_magnitude * output_sensitivity
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            # 修复噪声干扰度计算
            batch_size = noise.size(0)
            noise_flat = noise.view(batch_size, -1)
            noise_mag = torch.mean(torch.norm(noise_flat, p=2, dim=1))
            output_var = torch.mean(torch.var(outputs, dim=1))
            impact = (noise_mag * output_var).item()
            epoch_impact.append(impact)

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        avg_impact = np.mean(epoch_impact)

        metrics["loss"].append(epoch_loss)
        metrics["accuracy"].append(epoch_acc)
        metrics["noise_impact"].append(avg_impact)

        print(f"Epoch {epoch + 1} | 损失: {epoch_loss:.4f} | 准确率: {epoch_acc:.2f}% | 噪声干扰度: {avg_impact:.4f}")

    save_dir = os.path.join("experiment_results", exp_name)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))
    np.save(os.path.join(save_dir, "metrics.npy"), metrics)

    return metrics


# 5. 结果分析与主函数（保持不变）
def analyze_results(experiments, metrics_dict):
    plt.figure(figsize=(14, 5))

    plt.subplot(121)
    for name in experiments:
        plt.plot(metrics_dict[name]["accuracy"], marker='o', label=name)
    plt.title("带噪图像准确率对比")
    plt.xlabel("Epoch")
    plt.ylabel("准确率 (%)")
    plt.legend()

    plt.subplot(122)
    for name in experiments:
        plt.plot(metrics_dict[name]["noise_impact"], marker='s', label=name)
    plt.title("噪声对模型的干扰度对比")
    plt.xlabel("Epoch")
    plt.ylabel("干扰度 (越低越好)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("experiment_comparison.png")
    plt.close()
    print("结果对比图已保存为 experiment_comparison.png")

    baseline_acc = np.mean(metrics_dict["baseline"]["accuracy"])
    baseline_impact = np.mean(metrics_dict["baseline"]["noise_impact"])

    no_pred_acc = np.mean(metrics_dict["no_prediction_loss"]["accuracy"])
    acc_drop = (baseline_acc - no_pred_acc) / baseline_acc * 100
    print(f"\n移除预测损失项后，准确率下降: {acc_drop:.1f}%")
    if acc_drop > 20:
        print("结论：预测损失项是必要的")
    else:
        print("提示：可增大 pred_weight 增强预测损失作用")

    no_reg_impact = np.mean(metrics_dict["no_regularization"]["noise_impact"])
    impact_rise = (no_reg_impact - baseline_impact) / baseline_impact * 100
    print(f"移除正则化损失项后，噪声干扰度增加: {impact_rise:.1f}%")
    if impact_rise > 30:
        print("结论：正则化损失项是必要的")
    else:
        print("提示：可增大 reg_weight 增强正则化作用")


def main():
    image_folder = input("请输入包含图像的文件夹路径: ").strip()
    noise_folder = input("请输入包含噪声文件的文件夹路径: ").strip()

    if not os.path.isdir(image_folder):
        print(f"错误：图像文件夹 {image_folder} 不存在")
        return
    if not os.path.isdir(noise_folder):
        print(f"错误：噪声文件夹 {noise_folder} 不存在")
        return

    experiments = [
        {"name": "baseline", "use_pred": True, "use_reg": True},
        {"name": "no_prediction_loss", "use_pred": False, "use_reg": True},
        {"name": "no_regularization", "use_pred": True, "use_reg": False}
    ]

    metrics_dict = {}
    for exp in experiments:
        print(f"\n===== 开始实验: {exp['name']} =====")
        try:
            metrics = run_experiment(
                image_folder=image_folder,
                noise_folder=noise_folder,
                exp_name=exp["name"],
                use_pred_loss=exp["use_pred"],
                use_reg_loss=exp["use_reg"]
            )
            metrics_dict[exp["name"]] = metrics
        except Exception as e:
            print(f"实验 {exp['name']} 失败: {e}")
            continue

    if metrics_dict:
        analyze_results([exp["name"] for exp in experiments if exp["name"] in metrics_dict], metrics_dict)


if __name__ == "__main__":
    main()











