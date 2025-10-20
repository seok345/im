import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from tqdm import tqdm  # 학습 진행률을 표시하기 위한 라이브러리 추가


# --- 0. 라이브러리 설치 필요: tqdm ---
# 터미널에서 'pip install tqdm'을 먼저 실행해야 합니다.

# --- 1. U-Net Architecture Definition (모델 구조) ---
# (이 부분은 이전과 동일하며 생략합니다.)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=3):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.MaxPool2d(2);
        self.conv1 = DoubleConv(64, 128)
        self.down2 = nn.MaxPool2d(2);
        self.conv2 = DoubleConv(128, 256)
        self.down3 = nn.MaxPool2d(2);
        self.conv3 = DoubleConv(256, 512)
        self.down4 = nn.MaxPool2d(2);
        self.conv4 = DoubleConv(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1);
        x2 = self.conv1(x2)
        x3 = self.down2(x2);
        x3 = self.conv2(x3)
        x4 = self.down3(x3);
        x4 = self.conv3(x4)
        x5 = self.down4(x4);
        x5 = self.conv4(x5)

        x = self.up1(x5);
        x = torch.cat([x, x4], dim=1);
        x = self.up_conv1(x)
        x = self.up2(x);
        x = torch.cat([x, x3], dim=1);
        x = self.up_conv2(x)
        x = self.up3(x);
        x = torch.cat([x, x2], dim=1);
        x = self.up_conv3(x)
        x = self.up4(x);
        x = torch.cat([x, x1], dim=1);
        x = self.up_conv4(x)

        return self.outc(x)


# --- 2. Custom Dataset Class ---
# (이 부분은 이전과 동일합니다.)
class ColorizationDataset(Dataset):
    def __init__(self, color_dir, grayscale_dir, image_size=256):
        self.color_dir = color_dir
        self.grayscale_dir = grayscale_dir
        self.image_size = image_size

        self.filenames = [os.path.basename(f) for f in glob.glob(os.path.join(color_dir, '*.jpg'))]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size), antialias=True)
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        grayscale_path = os.path.join(self.grayscale_dir, filename)
        input_img = cv2.imread(grayscale_path, cv2.IMREAD_GRAYSCALE)

        color_path = os.path.join(self.color_dir, filename)
        target_img = cv2.imread(color_path, cv2.IMREAD_COLOR)

        if input_img is None or target_img is None:
            raise RuntimeError(f"이미지 로드 실패. 파일이 두 폴더 모두에 있는지 확인: {filename}")

        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

        input_tensor = self.transform(input_img).float()
        target_tensor = self.transform(target_img).float()

        return input_tensor, target_tensor


# --- 3. Training and Prediction Function (tqdm 적용) ---

def train_and_predict_colorization(color_dir, grayscale_dir, model_save_path='./colorization_unet.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 장치: {device}")

    # Hyperparameters
    num_epochs = 20
    learning_rate = 0.0005
    batch_size = 4
    image_size = 256

    # Data Loading
    print("\n📦 데이터셋 로딩 중...")
    dataset = ColorizationDataset(color_dir, grayscale_dir, image_size)

    if len(dataset) == 0:
        print("\n🚨 오류: 학습할 이미지가 없습니다. 두 폴더에 이미지가 있는지 확인하세요.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num_batches = len(dataloader)

    # Model, Loss, Optimizer
    model = UNet(n_channels=1, n_classes=3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- Training Loop ---
    print(f"\n--- 모델 학습 시작 (총 이미지 수: {len(dataset)}, 총 배치 수: {num_batches}) ---")
    model.train()

    for epoch in range(num_epochs):
        # tqdm을 사용하여 현재 Epoch의 진행 상황을 막대 그래프로 표시
        # desc: 진행률 표시줄의 제목 설정
        progress_bar = tqdm(enumerate(dataloader), total=num_batches,
                            desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")

        running_loss = 0.0

        for i, (inputs, targets) in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 진행률 표시줄에 현재 배치 손실(Loss) 값 업데이트
            progress_bar.set_postfix({'Loss': loss.item()})

        avg_epoch_loss = running_loss / num_batches
        print(f"\n[Epoch {epoch + 1} 완료] 🌟 평균 손실(Avg Loss): {avg_epoch_loss:.6f}")

    print("✅ 학습 완료. 모델 저장 중...")
    torch.save(model.state_dict(), model_save_path)
    print(f"모델이 '{model_save_path}'에 저장되었습니다.")

    # --- Prediction Example (테스트 부분) ---
    print("\n--- 학습된 모델로 테스트 (결과 시각화) ---")

    def load_new_grayscale_image(path, size=256):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: return None
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((size, size), antialias=True)
        ])
        return transform(img).float().unsqueeze(0).to(device)

    model.eval()

    if len(dataset) > 0:
        first_filename = dataset.filenames[0]
        test_grayscale_path = os.path.join(grayscale_dir, first_filename)
        test_input = load_new_grayscale_image(test_grayscale_path, image_size)

        if test_input is not None:
            with torch.no_grad():
                output_tensor = model(test_input)

            output_img = output_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
            input_img = test_input.squeeze().cpu().numpy()

            output_img = np.clip(output_img, 0, 1)

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(input_img, cmap='gray')
            plt.title("Grayscale Input")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(output_img)
            plt.title("Colorized Output")
            plt.axis('off')

            plt.show()


if __name__ == '__main__':
    # 학습에 사용할 데이터 폴더 경로
    COLOR_IMAGE_DIR = './color_images'
    GRAYSCALE_IMAGE_DIR = './grayscale_images'

    # 모델 학습 및 테스트 실행
    train_and_predict_colorization(COLOR_IMAGE_DIR, GRAYSCALE_IMAGE_DIR)