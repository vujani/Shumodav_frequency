import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.fft
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
import numpy as np


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, max(channel // reduction, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channel // reduction, 1), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DIV2KDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, noise_level=0.15):
        self.root = root
        self.transform = transform
        self.noise_level = noise_level
        self.image_files = [os.path.join(root, f) for f in os.listdir(root)
                            if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        clean = img
        noisy = img + self.noise_level * torch.randn_like(img)
        noisy = torch.clamp(noisy, 0.0, 1.0)
        return noisy, clean


class LowFreqNet(nn.Module):
    def __init__(self):
        super(LowFreqNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.conv4(out)
        return x + out


class HighFreqNet(nn.Module):
    def __init__(self):
        super(HighFreqNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.conv4(out)
        return x + out


class AdaptiveBlock(nn.Module):
    def __init__(self):
        super(AdaptiveBlock, self).__init__()
        self.low_net = LowFreqNet()
        self.high_net = HighFreqNet()
        self.high_weight = nn.Parameter(torch.tensor(0.5))
        self.low_weight = nn.Parameter(torch.tensor(1.0))

        self.mask_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.se_low = SEBlock(3)
        self.se_high = SEBlock(3)

    def forward(self, x):
        fft = torch.fft.fft2(x, norm="ortho")
        fft_shifted = torch.fft.fftshift(fft)
        b, c, H, W = x.shape

        mag = torch.abs(fft_shifted)
        mag_avg = torch.mean(mag, dim=1, keepdim=True)

        low_mask = self.mask_net(mag_avg)
        low_mask = low_mask.expand(-1, c, -1, -1)
        high_mask = 1.0 - low_mask

        low_fft = fft_shifted * low_mask
        high_fft = fft_shifted * high_mask

        low_fft_ishift = torch.fft.ifftshift(low_fft)
        high_fft_ishift = torch.fft.ifftshift(high_fft)
        low_ifft = torch.fft.ifft2(low_fft_ishift, norm="ortho").real
        high_ifft = torch.fft.ifft2(high_fft_ishift, norm="ortho").real

        low_processed = self.low_net(low_ifft)
        high_processed = self.high_net(high_ifft)

        low_processed = self.se_low(low_processed)
        high_processed = self.se_high(high_processed)

        out = self.low_weight * low_processed + self.high_weight * high_processed
        return torch.clamp(out, 0.0, 1.0)


def ssim_loss(img1, img2, window_size=11):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    channel = img1.size(1)
    coords = torch.arange(window_size, dtype=torch.float32, device=img1.device) - window_size // 2
    gauss = torch.exp(-coords ** 2 / (2 * 1.5 ** 2))
    gauss = gauss / gauss.sum()
    kernel_2d = gauss.unsqueeze(1).mm(gauss.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    window = kernel_2d.expand(channel, 1, window_size, window_size).contiguous()

    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return 1.0 - ssim_map.mean()


def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return (10 * torch.log10(1 / mse)).item()


def calculate_ssim(img1, img2, window_size=11):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    channel = img1.size(1)
    coords = torch.arange(window_size, dtype=torch.float32, device=img1.device) - window_size // 2
    gauss = torch.exp(-coords ** 2 / (2 * 1.5 ** 2))
    gauss = gauss / gauss.sum()
    kernel_2d = gauss.unsqueeze(1).mm(gauss.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    window = kernel_2d.expand(channel, 1, window_size, window_size).contiguous()
    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


def get_optimized_data_loader(dataset, batch_size=8, num_workers=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )


def train_adaptive_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8
    num_epochs = 100
    learning_rate = 1e-3
    patience = 10

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = DIV2KDataset(root='DIV2K/train', transform=transform, noise_level=0.15)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = get_optimized_data_loader(train_dataset, batch_size=batch_size)
    val_loader = get_optimized_data_loader(val_dataset, batch_size=batch_size)

    model = AdaptiveBlock().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for noisy, clean in train_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()

            output = model(noisy)
            loss = 0.7 * F.l1_loss(output, clean) + 0.3 * ssim_loss(output, clean)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * noisy.size(0)

        train_loss /= len(train_loader.dataset)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        psnr_total = 0.0
        ssim_total = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy = noisy.to(device)
                clean = clean.to(device)
                output = model(noisy)
                loss = 0.7 * F.l1_loss(output, clean) + 0.3 * ssim_loss(output, clean)
                val_loss += loss.item() * noisy.size(0)
                for i in range(output.size(0)):
                    psnr_total += calculate_psnr(output[i:i + 1], clean[i:i + 1])
                    ssim_total += calculate_ssim(output[i:i + 1], clean[i:i + 1])
        val_loss /= len(val_loader.dataset)
        avg_psnr = psnr_total / len(val_loader.dataset)
        avg_ssim = ssim_total / len(val_loader.dataset)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}, "
            f"Val Loss: {val_loss:.6f}, PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            os.makedirs('saved_models', exist_ok=True)
            torch.save(model.state_dict(), 'saved_models/adaptive_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Ранняя остановка активирована.")
                break


if __name__ == "__main__":
    train_adaptive_model()
