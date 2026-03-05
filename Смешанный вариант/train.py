import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
import numpy as np
import argparse
import torch.fft


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DIV2KDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, noise_level=0.15, noise_level_max=None):
        self.root = root
        self.transform = transform
        self.noise_level = noise_level
        self.noise_level_max = noise_level_max if noise_level_max is not None else noise_level
        self.image_files = []
        if os.path.isdir(root):
            self.image_files = [os.path.join(root, f) for f in os.listdir(root)
                                if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        clean = img
        sigma = self.noise_level if self.noise_level >= self.noise_level_max else (
            self.noise_level + (self.noise_level_max - self.noise_level) * torch.rand(1).item()
        )
        noisy = clean + sigma * torch.randn_like(clean)
        noisy = torch.clamp(noisy, 0.0, 1.0)
        return noisy, clean, torch.tensor(sigma, dtype=torch.float32)


class SyntheticDenoiseDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=64, size=256, noise_level=0.15, noise_level_max=None):
        self.num_samples = num_samples
        self.size = size
        self.noise_level = noise_level
        self.noise_level_max = noise_level_max if noise_level_max is not None else noise_level

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        torch.manual_seed(idx)
        clean = torch.rand(3, self.size, self.size)
        sigma = self.noise_level if self.noise_level >= self.noise_level_max else (
            self.noise_level + (self.noise_level_max - self.noise_level) * torch.rand(1).item()
        )
        noisy = clean + sigma * torch.randn_like(clean)
        noisy = torch.clamp(noisy, 0.0, 1.0)
        return noisy, clean, torch.tensor(sigma, dtype=torch.float32)


class AugmentFlip(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        noisy, clean, sigma = self.dataset[idx]
        if random.random() > 0.5:
            noisy = torch.flip(noisy, [-1])
            clean = torch.flip(clean, [-1])
        if random.random() > 0.5:
            noisy = torch.flip(noisy, [-2])
            clean = torch.flip(clean, [-2])
        return noisy, clean, sigma


class DenoiseNet(nn.Module):
    def __init__(self):
        super(DenoiseNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(8)]
        )
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.res_blocks(out)
        out = self.conv2(out)
        return x + out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return x + out


class NoiseEstimator(nn.Module):
    def __init__(self):
        super(NoiseEstimator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        feat = self.features(x).flatten(1)
        return self.head(feat)


class LowFreqNet(nn.Module):
    def __init__(self):
        super(LowFreqNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.se = SEBlock(64, reduction=16)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, sigma_map):
        inp = torch.cat([x, sigma_map], dim=1)
        out = self.relu(self.conv1(inp))
        out = self.relu(self.conv2(out))
        out = self.se(out)
        out = self.relu(self.conv3(out))
        out = self.conv4(out)
        return x + out


class HighFreqNet(nn.Module):
    def __init__(self):
        super(HighFreqNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.se = SEBlock(128, reduction=16)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, sigma_map):
        inp = torch.cat([x, sigma_map], dim=1)
        out = self.relu(self.conv1(inp))
        out = self.relu(self.conv2(out))
        out = self.se(out)
        out = self.relu(self.conv3(out))
        out = self.conv4(out)
        return x + out


class AdaptiveBlock(nn.Module):
    def __init__(self):
        super(AdaptiveBlock, self).__init__()
        self.noise_estimator = NoiseEstimator()
        self.low_net = LowFreqNet()
        self.high_net = HighFreqNet()

        self.mask_net = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.weight_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 2),
            nn.Softplus()
        )
        nn.init.zeros_(self.weight_net[2].weight)
        nn.init.constant_(self.weight_net[2].bias, 0.5414)

    def forward(self, x, sigma_true=None):
        b, c, H, W = x.shape

        sigma_est = self.noise_estimator(x)
        sigma = sigma_true.view(b, 1) if sigma_true is not None else sigma_est
        sigma_map = sigma.view(b, 1, 1, 1).expand(b, 1, H, W)

        fft = torch.fft.fft2(x, norm="ortho")
        fft_shifted = torch.fft.fftshift(fft)

        mag = torch.abs(fft_shifted)
        mag_avg = torch.mean(mag, dim=1, keepdim=True)

        mask_input = torch.cat([mag_avg, sigma_map], dim=1)
        low_mask = self.mask_net(mask_input)
        low_mask = (low_mask + torch.flip(low_mask, dims=(-2, -1))) / 2.0
        low_mask = low_mask.expand(-1, c, -1, -1)
        high_mask = 1.0 - low_mask

        low_fft = fft_shifted * low_mask
        high_fft = fft_shifted * high_mask

        low_ifft = torch.fft.ifft2(torch.fft.ifftshift(low_fft), norm="ortho").real
        high_ifft = torch.fft.ifft2(torch.fft.ifftshift(high_fft), norm="ortho").real

        low_processed = self.low_net(low_ifft, sigma_map)
        high_processed = self.high_net(high_ifft, sigma_map)

        weights = self.weight_net(sigma)
        low_w = weights[:, 0:1].view(b, 1, 1, 1)
        high_w = weights[:, 1:2].view(b, 1, 1, 1)
        out = low_w * low_processed + high_w * high_processed

        return out, sigma_est


def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse <= 0:
        return 100.0
    psnr = 10.0 * np.log10(1.0 / (mse + 1e-10))
    return min(100.0, float(psnr))


def gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    window = _2D_window.unsqueeze(0).unsqueeze(0)
    return window.expand(channel, 1, window_size, window_size).contiguous()


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel).to(img1.device)
    mu1 = nn.functional.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = nn.functional.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = nn.functional.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = nn.functional.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = nn.functional.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item() if size_average else ssim_map.mean(1).mean(1).mean(1).item()


def get_optimized_data_loader(dataset, batch_size=4, num_workers=4, pin_memory=None):
    if pin_memory is None:
        pin_memory = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )


def train_denoise_model(use_adaptive=True, freeze_adaptive=True, max_epochs=None, noise_level=0.20, noise_level_max=0.30, batch_size=None, num_workers=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    if batch_size is None:
        batch_size = 8
    if num_workers is None:
        num_workers = 1 if device.type == "cuda" else 0
    accumulation_steps = max(1, 16 // batch_size)
    num_epochs = max_epochs if max_epochs is not None else 200
    learning_rate = 8e-4
    warmup_epochs = 3
    patience = 15

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = DIV2KDataset(root='DIV2K/train', transform=transform, noise_level=noise_level, noise_level_max=noise_level_max)
    if len(dataset) == 0:
        print("DIV2K/train пуст или отсутствует — используем синтетический датасет для теста.")
        dataset = SyntheticDenoiseDataset(num_samples=256, size=256, noise_level=noise_level, noise_level_max=noise_level_max)
    val_size = int(0.2 * len(dataset))
    val_size = max(1, val_size)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataset = AugmentFlip(train_dataset)

    train_loader = get_optimized_data_loader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    val_loader = get_optimized_data_loader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    adaptive_block = None
    if use_adaptive:
        adaptive_block = AdaptiveBlock().to(device)
        adaptive_model_path = 'saved_models/adaptive_model.pth'
        if os.path.exists(adaptive_model_path):
            try:
                adaptive_block.load_state_dict(torch.load(adaptive_model_path, map_location=device))
            except (RuntimeError, KeyError):
                print("Архитектура адаптивного блока изменилась — необходимо переобучить (python adaptive.py).")
                adaptive_block = None
            else:
                if freeze_adaptive:
                    for param in adaptive_block.parameters():
                        param.requires_grad = False
                adaptive_block.eval()
        else:
            print("Adaptive model not found, proceeding without it.")
            adaptive_block = None

    denoise_net = DenoiseNet().to(device)
    ema_denoise_net = DenoiseNet().to(device)
    ema_denoise_net.load_state_dict(denoise_net.state_dict())
    ema_decay = 0.999

    optimizer = optim.Adam(denoise_net.parameters(), lr=learning_rate)
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    loss_mse_weight, loss_l1_weight = 0.85, 0.15
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=2, eta_min=1e-6)

    best_val_psnr = 0.0
    best_psnr_at_03 = 0.0
    epochs_no_improve = 0
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for epoch in range(num_epochs):
        if epoch < warmup_epochs:
            warmup_scale = (epoch + 1) / warmup_epochs
            for g in optimizer.param_groups:
                g["lr"] = learning_rate * warmup_scale
        else:
            scheduler.step(epoch - warmup_epochs)

        denoise_net.train()
        train_loss = 0.0
        optimizer.zero_grad()
        for batch_idx, (noisy, clean, sigma_gt) in enumerate(train_loader):
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            sigma_gt = sigma_gt.to(device, non_blocking=True).view(-1, 1)

            if adaptive_block is not None:
                with torch.no_grad() if freeze_adaptive else torch.enable_grad():
                    adaptive_output, _ = adaptive_block(noisy, sigma_true=sigma_gt)
                input_to_denoise = adaptive_output
            else:
                input_to_denoise = noisy

            if use_amp:
                with torch.cuda.amp.autocast():
                    output = denoise_net(input_to_denoise)
                    mse = criterion_mse(output, clean)
                    l1 = criterion_l1(output, clean)
                    loss = (loss_mse_weight * mse + loss_l1_weight * l1) / accumulation_steps
                scaler.scale(loss).backward()
            else:
                output = denoise_net(input_to_denoise)
                mse = criterion_mse(output, clean)
                l1 = criterion_l1(output, clean)
                loss = (loss_mse_weight * mse + loss_l1_weight * l1) / accumulation_steps
                loss.backward()

            train_loss += (loss_mse_weight * mse.item() + loss_l1_weight * l1.item()) * noisy.size(0)

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                if use_amp:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(denoise_net.parameters(), max_norm=2.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(denoise_net.parameters(), max_norm=2.0)
                    optimizer.step()
                optimizer.zero_grad()
                with torch.no_grad():
                    for p_ema, p in zip(ema_denoise_net.parameters(), denoise_net.parameters()):
                        p_ema.data.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)
        train_loss /= len(train_loader.dataset)

        denoise_net.eval()
        val_loss = 0.0
        psnr_total = 0.0
        ssim_total = 0.0
        psnr_at_03_total = 0.0
        n_03 = 0
        with torch.no_grad():
            for noisy, clean, sigma_gt in val_loader:
                noisy = noisy.to(device, non_blocking=True)
                clean = clean.to(device, non_blocking=True)
                sigma_gt = sigma_gt.to(device, non_blocking=True).view(-1, 1)

                if adaptive_block is not None:
                    input_to_denoise, _ = adaptive_block(noisy, sigma_true=sigma_gt)
                else:
                    input_to_denoise = noisy
                output = denoise_net(input_to_denoise)
                val_loss += criterion_mse(output, clean).item() * noisy.size(0)
                for i in range(output.size(0)):
                    psnr_total += calculate_psnr(output[i:i + 1], clean[i:i + 1])
                    ssim_total += ssim(output[i:i + 1], clean[i:i + 1])

                noisy_03 = (clean + 0.3 * torch.randn_like(clean, device=clean.device)).clamp(0.0, 1.0)
                sigma_03 = torch.full((clean.size(0), 1), 0.3, device=device)
                if adaptive_block is not None:
                    input_03, _ = adaptive_block(noisy_03, sigma_true=sigma_03)
                else:
                    input_03 = noisy_03
                output_03 = denoise_net(input_03)
                for i in range(output_03.size(0)):
                    psnr_at_03_total += calculate_psnr(output_03[i:i + 1], clean[i:i + 1])
                    n_03 += 1

        val_loss /= len(val_loader.dataset)
        avg_psnr = psnr_total / len(val_loader.dataset)
        avg_ssim = ssim_total / len(val_loader.dataset)
        psnr_at_03 = psnr_at_03_total / n_03 if n_03 else 0.0

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{num_epochs}, LR: {current_lr:.2e}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Val PSNR: {avg_psnr:.2f} dB, PSNR@sigma=0.3: {psnr_at_03:.2f} dB, SSIM: {avg_ssim:.4f}", flush=True)

        if psnr_at_03 > best_psnr_at_03:
            best_psnr_at_03 = psnr_at_03
            best_val_psnr = avg_psnr
            epochs_no_improve = 0
            os.makedirs('saved_models', exist_ok=True)
            torch.save(ema_denoise_net.state_dict(), 'saved_models/denoise_model.pth')
            print(f"  -> Сохранена модель. Лучший PSNR@sigma=0.3: {best_psnr_at_03:.2f} dB", flush=True)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Ранняя остановка активирована.", flush=True)
                break

    print(f"Лучший результат: Val PSNR {best_val_psnr:.2f} dB, PSNR@sigma=0.3: {best_psnr_at_03:.2f} dB", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_adaptive', action='store_true', help='Использовать адаптивный блок')
    parser.add_argument('--freeze_adaptive', action='store_true', help='Заморозить веса адаптивного блока')
    parser.add_argument('--max_epochs', type=int, default=None, help='Макс. эпох (для теста, напр. 10)')
    parser.add_argument('--noise_level', type=float, default=0.20, help='Минимальный уровень шума')
    parser.add_argument('--noise_level_max', type=float, default=0.30, help='Максимальный уровень шума')
    parser.add_argument('--batch_size', type=int, default=None, help='Размер батча (по умолч. 8)')
    parser.add_argument('--num_workers', type=int, default=None, help='Воркеры DataLoader (по умолч. 1 при CUDA)')
    args = parser.parse_args()
    train_denoise_model(
        use_adaptive=args.use_adaptive,
        freeze_adaptive=args.freeze_adaptive,
        max_epochs=args.max_epochs,
        noise_level=args.noise_level,
        noise_level_max=args.noise_level_max,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
