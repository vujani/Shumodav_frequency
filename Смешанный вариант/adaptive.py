import os
import torch
import torch.nn as nn
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


class NoiseEstimator(nn.Module):
    """Оценка уровня шума σ из зашумлённого изображения.
    При обучении используется истинный σ (teacher forcing),
    при инференсе — предсказание этой сети."""
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
    """Обработка низкочастотной компоненты.
    Принимает 3 канала изображения + 1 канал sigma_map = 4 входных канала.
    SE-блок применяется на 64 каналах, где канальное внимание эффективно."""
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
    """Обработка высокочастотной компоненты.
    Принимает 3 канала изображения + 1 канал sigma_map = 4 входных канала.
    SE-блок применяется на 128 каналах."""
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
    """Частотно-адаптивный блок с кондиционированием по уровню шума.

    Отличия от предыдущей версии:
    1. NoiseEstimator — оценивает σ; при обучении используется GT σ
    2. mask_net принимает mag_avg + sigma_map (2 канала) — маска учитывает уровень шума
    3. weight_net — MLP, выдающий веса комбинации low/high в зависимости от σ
    4. SE-блоки перенесены внутрь LowFreqNet/HighFreqNet на 64/128 каналов
    """
    def __init__(self):
        super(AdaptiveBlock, self).__init__()
        self.noise_estimator = NoiseEstimator()
        self.low_net = LowFreqNet()
        self.high_net = HighFreqNet()

        # Маска: mag_avg (1 канал) + sigma_map (1 канал) = 2 входных канала
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

        # Адаптивные веса комбинации: зависят от σ
        # При высоком шуме сеть учится снижать вес высокочастотной ветки
        self.weight_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 2),
            nn.Softplus()
        )
        # Инициализация: softplus(0.5414) ≈ 1.0, начинаем с равных весов
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


def train_adaptive_model(max_epochs=None, noise_level=0.15, noise_level_max=0.30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    batch_size = 4
    accumulation_steps = 4
    num_epochs = max_epochs if max_epochs is not None else 100
    learning_rate = 5e-4
    warmup_epochs = 5
    patience = 8
    num_workers = 2 if device.type == "cuda" else 0

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = DIV2KDataset(root='DIV2K/train', transform=transform, noise_level=noise_level, noise_level_max=noise_level_max)
    if len(dataset) == 0:
        print("DIV2K/train пуст или отсутствует — используем синтетический датасет.")
        dataset = SyntheticDenoiseDataset(num_samples=256, size=256, noise_level=noise_level, noise_level_max=noise_level_max)
    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = get_optimized_data_loader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    val_loader = get_optimized_data_loader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    model = AdaptiveBlock().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.6, patience=8)

    best_val_psnr = 0.0
    epochs_no_improve = 0
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for epoch in range(num_epochs):
        if epoch < warmup_epochs:
            warmup_scale = (epoch + 1) / warmup_epochs
            for g in optimizer.param_groups:
                g["lr"] = learning_rate * warmup_scale

        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        for batch_idx, (noisy, clean, sigma_gt) in enumerate(train_loader):
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            sigma_gt = sigma_gt.to(device, non_blocking=True).view(-1, 1)

            if use_amp:
                with torch.cuda.amp.autocast():
                    output, sigma_est = model(noisy, sigma_true=sigma_gt)
                    recon_loss = criterion(output, clean)
                    sigma_loss = nn.functional.mse_loss(sigma_est, sigma_gt)
                    loss = (recon_loss + 0.1 * sigma_loss) / accumulation_steps
                scaler.scale(loss).backward()
            else:
                output, sigma_est = model(noisy, sigma_true=sigma_gt)
                recon_loss = criterion(output, clean)
                sigma_loss = nn.functional.mse_loss(sigma_est, sigma_gt)
                loss = (recon_loss + 0.1 * sigma_loss) / accumulation_steps
                loss.backward()

            train_loss += (recon_loss.item() + 0.1 * sigma_loss.item()) * noisy.size(0)

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                if use_amp:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad()
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        psnr_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for noisy, clean, sigma_gt in val_loader:
                noisy = noisy.to(device, non_blocking=True)
                clean = clean.to(device, non_blocking=True)
                sigma_gt = sigma_gt.to(device, non_blocking=True).view(-1, 1)
                output, _ = model(noisy, sigma_true=sigma_gt)
                loss = criterion(output, clean)
                val_loss += loss.item() * noisy.size(0)
                for i in range(output.size(0)):
                    mse = torch.mean((output[i:i+1] - clean[i:i+1]) ** 2).item()
                    if mse <= 0:
                        psnr_sum += 100.0
                    else:
                        psnr_sum += min(100.0, 10.0 * np.log10(1.0 / (mse + 1e-10)))
                    n_val += 1
        val_loss /= len(val_loader.dataset)
        avg_psnr_val = psnr_sum / n_val if n_val else 0.0

        scheduler.step(avg_psnr_val)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1}/{num_epochs}, LR: {current_lr:.2e}, Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val PSNR: {avg_psnr_val:.2f} dB", flush=True)

        if avg_psnr_val > best_val_psnr:
            best_val_psnr = avg_psnr_val
            epochs_no_improve = 0
            os.makedirs('saved_models', exist_ok=True)
            torch.save(model.state_dict(), 'saved_models/adaptive_model.pth')
            print(f"  -> Сохранена модель. Лучший Val PSNR: {best_val_psnr:.2f} dB", flush=True)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Ранняя остановка активирована.", flush=True)
                break

    print(f"Лучший результат адаптивного блока (Val PSNR): {best_val_psnr:.2f} dB", flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--max_epochs", type=int, default=None, help="Макс. эпох (для теста)")
    p.add_argument("--noise_level", type=float, default=0.15, help="Минимальный уровень шума")
    p.add_argument("--noise_level_max", type=float, default=0.30, help="Макс. уровень шума")
    args = p.parse_args()
    train_adaptive_model(max_epochs=args.max_epochs, noise_level=args.noise_level, noise_level_max=args.noise_level_max)
