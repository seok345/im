import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse

# --- U-Net 모델 정의 ---
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        for feature in features:
            self.downs.append(self.conv_block(in_channels, feature))
            in_channels = feature
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(self.conv_block(feature*2, feature))
        self.bottleneck = self.conv_block(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](x)
        return torch.sigmoid(self.final_conv(x))

# --- 테스트용 흑백 이미지 생성 ---
def create_test_bw_image(path, size=(256, 256)):
    gradient = np.tile(np.linspace(0, 255, size[0], dtype=np.uint8), (size[1],1))
    img = Image.fromarray(gradient)
    img.save(path)
    print(f"테스트용 흑백 이미지 저장: {path}")

# --- 이미지 불러오기 & 전처리 ---
def load_image(path, device, img_size=(256,256)):
    img = Image.open(path).convert("L")
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    img = transform(img).unsqueeze(0).to(device)
    return img

# --- 결과 저장 ---
def save_image(tensor, path):
    tensor = tensor.squeeze(0).cpu()
    img = tensor.permute(1, 2, 0).numpy()
    img = (img * 255).clip(0,255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(path)
    print(f"컬러 이미지 저장 완료: {path}")

# --- 메인 ---
def main(args):
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    if args.create_test_image:
        create_test_bw_image(args.input)

    model = UNet().to(device)
    model.eval()
    # TODO: 학습된 모델 가중치 있으면 여기서 로드
    # model.load_state_dict(torch.load("model_weights.pth", map_location=device))

    img = load_image(args.input, device)
    with torch.no_grad():
        output = model(img)
    save_image(output, args.output)

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        sys.argv.extend([
            '--input', 'test_bw.png',
            '--output', 'result_color.png',
            '--create_test_image'
        ])

    parser = argparse.ArgumentParser(description="흑백/적외선 이미지를 컬러 이미지로 변환 (테스트 이미지 생성 옵션 포함)")
    parser.add_argument('--input', type=str, required=True, help="입력 흑백 이미지 경로 (테스트 이미지 생성 시 저장 위치)")
    parser.add_argument('--output', type=str, required=True, help="출력 컬러 이미지 경로")
    parser.add_argument('--device', type=str, default=None, help="사용 디바이스 (cpu 또는 cuda)")
    parser.add_argument('--create_test_image', action='store_true', help="테스트용 흑백 이미지 생성 후 변환")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if output_dir != '':
        os.makedirs(output_dir, exist_ok=True)

    main(args)
