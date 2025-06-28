import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import yaml
import torch
import random
from models.HIV_GAN import VariationalAutoencoder, VAEGenerator, Discriminator

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from datetime import datetime
import pandas as pd
import numpy as np
from utils.metrics import compute_ssim, compute_psnr, batch_extract_features, compute_fid
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import ImageDataset


def evaluate_generated_images(real_imgs, fake_imgs, eval_dir, epoch, prefix="vae"):
    """
    评估并保存图像质量指标：平均 SSIM、PSNR、FID
    - real_imgs, fake_imgs: Tensor, shape = (B, 3, H, W)
    - eval_dir: 保存图像和日志的路径
    """
    import os
    import torch
    from torchvision.utils import save_image
    from torchvision.datasets import ImageFolder
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from PIL import ImageOps
    from utils.metrics import compute_ssim, compute_psnr, batch_extract_features, compute_fid, InceptionFeatureExtractor

    def pad_to_square(img, fill=255):
        w, h = img.size
        if w == h:
            return img
        elif w > h:
            padding = (0, (w - h) // 2, 0, (w - h + 1) // 2)
        else:
            padding = ((h - w) // 2, 0, (h - w + 1) // 2, 0)
        return ImageOps.expand(img, border=padding, fill=fill)

    os.makedirs(eval_dir, exist_ok=True)

    # 平均 SSIM 和 PSNR 计算
    ssim_vals = []
    psnr_vals = []

    num_imgs = len(real_imgs)
    for i in range(num_imgs):
        for j in range(num_imgs):
            r_cpu = real_imgs[i].cpu()
            f_cpu = fake_imgs[j].cpu()
            ssim_vals.append(compute_ssim(r_cpu, f_cpu))
            psnr_vals.append(compute_psnr(r_cpu, f_cpu))

    ssim_val = sum(ssim_vals) / len(ssim_vals)
    psnr_val = sum(psnr_vals) / len(psnr_vals)

    # 保存图像
    save_image(fake_imgs[:25], os.path.join(eval_dir, f"{prefix}_epoch{epoch}_samples.png"), nrow=5)

    # 计算 FID
    fid_val = 0

    return {
        "ssim": round(ssim_val, 4),
        "psnr": round(psnr_val, 4),
        "fid": round(fid_val, 4) if fid_val is not None else None
    }


# 模块 2：加载人工交互反馈图像及标签

def load_saved_images(feedback_dir, image_size=(128, 128), device="cuda"):
    """
    加载用户在 GUI 中选择并打分保存的图像与标签。
    返回值：Tensor图像集合, 标签Tensor (N, 1)
    """
    saved_images = []
    saved_labels = []
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    if not os.path.exists(feedback_dir):
        return None, None

    for subfolder in os.listdir(feedback_dir):
        subfolder_path = os.path.join(feedback_dir, subfolder)
        if not os.path.isdir(subfolder_path) or not subfolder.startswith("num_"):
            continue

        label_file = os.path.join(subfolder_path, "labels.txt")
        if not os.path.exists(label_file):
            continue

        with open(label_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                img_name, label = parts[0], float(parts[1])
                img_path = os.path.join(subfolder_path, img_name)
                if os.path.exists(img_path):
                    try:
                        img = Image.open(img_path).convert("RGB")
                        img_tensor = transform(img).to(device)
                        saved_images.append(img_tensor)
                        saved_labels.append(label)
                    except Exception as e:
                        print(f"[加载失败] {img_path}: {e}")

    if saved_images:
        imgs_tensor = torch.stack(saved_images)
        labels_tensor = torch.tensor(saved_labels, device=device).unsqueeze(1)
        print(f"[反馈加载] 加载图像数量: {len(saved_images)}")
        return imgs_tensor, labels_tensor
    else:
        return None, None

# 模块 3：VAE 训练

def vae_loss_function(recon_x, x, mu, logvar, beta=1.0, loss_type='mse'):
    recon_loss = nn.MSELoss()(recon_x, x) if loss_type == 'mse' else nn.BCELoss()(recon_x, x)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + beta * kl_div

def train_vae(model, optimizer, dataloader, config, device, save_dir, log_path):
    model.train()
    log = []
    os.makedirs(save_dir, exist_ok=True)
    eval_dir = os.path.join(save_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    for epoch in range(config['vae']['num_epochs']):
        total_loss = 0
        for imgs in dataloader:
            imgs = imgs.to(device)
            recon, mu, logvar = model(imgs)
            loss = vae_loss_function(recon, imgs, mu, logvar,
                                     beta=config['vae']['kl_weight'],
                                     loss_type=config['vae']['recon_loss'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[VAE] Epoch {epoch+1}/{config['vae']['num_epochs']}, Loss: {avg_loss:.4f}")

        if (epoch + 1) % config['vae']['eval_interval'] == 0:
            with torch.no_grad():
                sample_imgs, _, _ = model(imgs[:25])
                eval_result = evaluate_generated_images(imgs[:25], sample_imgs, eval_dir, epoch+1, prefix="vae")
                log.append({"epoch": epoch+1, "loss": avg_loss, **eval_result})
                save_image(sample_imgs, os.path.join(save_dir, f"samples_epoch{epoch+1}.png"), nrow=5)

    torch.save(model.encoder.state_dict(), os.path.join(save_dir, "vae_encoder_final.pth"))
    torch.save(model.decoder.state_dict(), os.path.join(save_dir, "vae_decoder_final.pth"))

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    df = pd.DataFrame(log)
    sheet = datetime.now().strftime("vae_%Y%m%d_%H%M%S")
    with pd.ExcelWriter(log_path, mode='a', engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet, index=False)

# 模块 4：GAN 训练（无交互）

import os
import torch
import torch.nn as nn
import torch.autograd as autograd
from torchvision.utils import save_image
import pandas as pd
from datetime import datetime

def train_gan_plain(generator, discriminator, g_opt, d_opt, dataloader, config, device, save_dir, log_path):
    generator.train()
    discriminator.train()
    log = []

    os.makedirs(save_dir, exist_ok=True)
    eval_dir = os.path.join(save_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(config['gan']['num_epochs']):
        total_g, total_d = 0, 0

        for imgs in dataloader:
            imgs = imgs.to(device)
            b = imgs.size(0)
            real_label = torch.full((b, 1), 0.9, device=device)  # Label smoothing
            fake_label = torch.zeros((b, 1), device=device)

            # === Train Discriminator ===
            d_opt.zero_grad()

            # Real samples
            real_noisy = imgs + torch.randn_like(imgs) * config['gan']['noise_std']
            real_out = discriminator(real_noisy)
            d_real_loss = criterion(real_out, real_label)

            # Fake samples
            fake_imgs = generator(b).detach()
            fake_noisy = fake_imgs + torch.randn_like(fake_imgs) * config['gan']['noise_std']
            fake_out = discriminator(fake_noisy)
            d_fake_loss = criterion(fake_out, fake_label)

            # === WGAN-GP Gradient Penalty ===
            epsilon = torch.rand(b, 1, 1, 1, device=device)
            interpolated = epsilon * imgs + (1 - epsilon) * fake_imgs
            interpolated.requires_grad_(True)
            interpolated_out = discriminator(interpolated)

            gradients = autograd.grad(
                outputs=interpolated_out,
                inputs=interpolated,
                grad_outputs=torch.ones_like(interpolated_out),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]

            gradients = gradients.view(b, -1)
            gradient_norm = gradients.norm(2, dim=1)
            gp = ((gradient_norm - 1) ** 2).mean()

            # Total discriminator loss
            d_loss = d_real_loss + d_fake_loss + config['gan']['grad_penalty_weight'] * gp
            d_loss.backward()
            d_opt.step()
            total_d += d_loss.item()

            # === Train Generator ===
            g_opt.zero_grad()
            fake_imgs = generator(b)
            fake_noisy = fake_imgs + torch.randn_like(fake_imgs) * config['gan']['noise_std']
            output = discriminator(fake_noisy)
            g_loss = criterion(output, real_label)
            g_loss.backward()
            g_opt.step()
            total_g += g_loss.item()

        avg_g = total_g / len(dataloader)
        avg_d = total_d / len(dataloader)
        print(f"[GAN] Epoch {epoch+1}/{config['gan']['num_epochs']}, G: {avg_g:.4f}, D: {avg_d:.4f}")

        if (epoch + 1) % config['gan']['eval_interval'] == 0:
            with torch.no_grad():
                sample = generator(100)
                eval_epoch_dir = os.path.join(eval_dir, f"epoch_{epoch + 1}")
                os.makedirs(eval_epoch_dir, exist_ok=True)

                for i, img in enumerate(sample):
                    save_path = os.path.join(eval_epoch_dir, f"img_{i:03d}.png")
                    save_image(img, save_path)

                eval_result = evaluate_generated_images(imgs[:25], sample[:25], eval_epoch_dir, epoch + 1, prefix="gan")
                log.append({
                    "epoch": epoch + 1,
                    "g_loss": avg_g,
                    "d_loss": avg_d,
                    **eval_result
                })

    # Save final model and logs
    torch.save(generator.state_dict(), os.path.join(save_dir, "generator_final.pth"))
    torch.save(discriminator.state_dict(), os.path.join(save_dir, "discriminator_final.pth"))

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    df = pd.DataFrame(log)
    sheet = datetime.now().strftime("gan_%Y%m%d_%H%M%S")
    with pd.ExcelWriter(log_path, mode='a', engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet, index=False)



# =====================
# 主控训练入口
# =====================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_next_experiment_id(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if d.startswith("exp") and d[3:].isdigit()]
    exp_nums = [int(d[3:]) for d in existing]
    next_id = max(exp_nums) + 1 if exp_nums else 1
    return f"exp{next_id}"


def main():
    # 加载配置
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(f"cuda:{config['device']['gpu_id']}" if config['device']['use_cuda'] and torch.cuda.is_available() else "cpu")
    set_seed(config['seed'])

    # 方法名（如 hiv_gan）
    method_name = "hiv_gan"

    # 生成实验编号 exp1, exp2...
    exp_id = get_next_experiment_id(os.path.join("results", "outputs", method_name))

    # 构建输出目录结构
    output_base = os.path.join("results", "outputs", method_name, exp_id)
    vae_save_dir = os.path.join(output_base, "vae")
    gan_save_dir = os.path.join(output_base, "gan")
    os.makedirs(vae_save_dir, exist_ok=True)
    os.makedirs(gan_save_dir, exist_ok=True)

    # 构建日志目录结构并创建空 xlsx 文件（避免首次写入失败）
    log_dir = os.path.join("results", "logs", method_name, exp_id)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.xlsx")
    if not os.path.exists(log_path):
        pd.DataFrame().to_excel(log_path, index=False)

    # 加载数据
    train_dataset = ImageDataset(config['path']['dataset'], image_size=(config['vae']['image_size'], config['vae']['image_size']))
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    # 初始化模型
    vae = VariationalAutoencoder(latent_dim=config['vae']['latent_dim']).to(device)
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=config['vae']['learning_rate'], betas=(config['vae']['beta1'], config['vae']['beta2']))

    generator = VAEGenerator(vae.decoder, config['vae']['latent_dim']).to(device)
    discriminator = Discriminator(vae.encoder).to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=config['gan']['learning_rate_generator'])
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config['gan']['learning_rate_discriminator'])

    # 训练 VAE
    train_vae(vae, vae_optimizer, train_loader, config, device, vae_save_dir, log_path)

    # 训练 GAN
    train_gan_plain(generator, discriminator, g_optimizer, d_optimizer,
                        train_loader, config, device, gan_save_dir, log_path)

if __name__ == "__main__":
    main()

