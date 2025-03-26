import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
from tqdm import tqdm


# 1. 自定义数据集类
class FloorplanDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, target_size=(256, 256)):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.target_size = target_size

        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png'))])
        self.label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(('.png'))])

        assert len(self.image_files) == len(self.label_files), "Number of images and labels must match"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 使用Pillow加载图像
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = np.array(Image.open(img_path).convert('RGB').resize(self.target_size))
        img = img.astype(np.float32) / 255.0

        # 使用Pillow加载标签
        lbl_path = os.path.join(self.label_dir, self.label_files[idx])
        lbl = np.array(Image.open(lbl_path).convert('RGB').resize(self.target_size))
        lbl = lbl.astype(np.float32) / 255.0

        if self.transform:
            img = self.transform(img)
            lbl = self.transform(lbl)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1)
            lbl = torch.from_numpy(lbl).permute(2, 0, 1)

        return img, lbl


# 2. 定义模型
class FloorplanSegmentationModel(nn.Module):
    def __init__(self):
        super(FloorplanSegmentationModel, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 输出层
        self.output = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.output(x)
        return x


# 3. 自定义评估指标
def calculate_metrics(y_true, y_pred, threshold=0.5):
    """
    计算图像分割的评估指标
    :param y_true: 真实标签 (N, H, W, C)
    :param y_pred: 预测结果 (N, H, W, C)
    :param threshold: 二值化阈值
    :return: 评估指标字典
    """
    # 将预测结果二值化
    y_pred_bin = (y_pred > threshold).astype(np.uint8)
    y_true_bin = (y_true > threshold).astype(np.uint8)

    # 计算每个通道的IoU
    iou_scores = []
    for c in range(3):  # 对RGB三个通道分别计算
        iou = jaccard_score(
            y_true_bin[..., c].flatten(),
            y_pred_bin[..., c].flatten(),
            average='macro'
        )
        iou_scores.append(iou)

    # 计算平均IoU
    mean_iou = np.mean(iou_scores)

    # 计算像素准确率
    pixel_acc = np.mean(y_pred_bin == y_true_bin)

    return {
        'mean_iou': mean_iou,
        'pixel_accuracy': pixel_acc,
        'iou_red': iou_scores[0],
        'iou_green': iou_scores[1],
        'iou_blue': iou_scores[2],
    }


# 4. 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=50):
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        # 训练阶段
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} - Validation"):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved best model")


# 5. 评估和保存预测结果
def evaluate_and_save_predictions(model, test_loader, device, save_dir='same_predictions'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # 保存预测和真实图像
            for j in range(outputs.size(0)):
                pred_img = outputs[j].cpu().numpy().transpose(1, 2, 0) * 255
                true_img = labels[j].cpu().numpy().transpose(1, 2, 0) * 255

                pred_img = pred_img.astype(np.uint8)
                true_img = true_img.astype(np.uint8)

                cv2.imwrite(os.path.join(save_dir, f'pred_{i}_{j}.png'), cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(save_dir, f'true_{i}_{j}.png'), cv2.cvtColor(true_img, cv2.COLOR_RGB2BGR))

                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.title('Prediction')
                plt.imshow(pred_img)
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.title('Ground Truth')
                plt.imshow(true_img)
                plt.axis('off')

                plt.savefig(os.path.join(save_dir, f'comparison_{i}_{j}.png'))
                plt.close()

            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # 计算评估指标
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    all_preds = all_preds.transpose(0, 2, 3, 1)
    all_labels = all_labels.transpose(0, 2, 3, 1)

    metrics = calculate_metrics(all_labels, all_preds)
    print("\nEvaluation Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    with open(os.path.join(save_dir, 'evaluation_results.txt'), 'w') as f:
        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}\n")

    return metrics


# 6. 主函数
def main():
    # 参数设置
    batch_size = 8
    epochs = 50
    target_size = (256, 256)
    lr = 0.001

    # 检查GPU可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),  # 自动将numpy转换为tensor并归一化到[0,1]
    ])

    # 创建数据集和数据加载器
    train_dataset = FloorplanDataset('train_images', 'same_train_labels', transform, target_size)
    test_dataset = FloorplanDataset('test_images', 'same_test_labels', transform, target_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    model = FloorplanSegmentationModel().to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 使用均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练模型
    print("Starting training...")
    train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs)

    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))

    # 评估模型并保存预测结果
    print("\nEvaluating model...")
    evaluate_and_save_predictions(model, test_loader, device)

    print("Training and evaluation completed!")


if __name__ == "__main__":
    main()