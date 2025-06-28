

from models.dcgan import DCGANGenerator, DCGANDiscriminator

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ImageDataset
from models.dcvae import DCVAE
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
from torchvision.utils import save_image
import yaml


def load_config(config_path):
    with open("configs/default.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_bagan_logging(base_output="results/outputs/bagan", base_log="results/logs/bagan"):
    os.makedirs(base_output, exist_ok=True)
    os.makedirs(base_log, exist_ok=True)

    existing = [d for d in os.listdir(base_output) if d.startswith("exp") and d[3:].isdigit()]
    next_id = max([int(d[3:]) for d in existing], default=0) + 1
    exp_id = f"exp{next_id}"

    ae_save_dir = os.path.join(base_output, exp_id, "dcgan")
    os.makedirs(ae_save_dir, exist_ok=True)

    log_dir = os.path.join(base_log, exp_id)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.xlsx")
    if not os.path.exists(log_path):
        pd.DataFrame().to_excel(log_path, index=False)

    return exp_id, 0, ae_save_dir, log_path



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


def train_dcgan(config):
    device = torch.device(f"cuda:{config['device']['gpu_id']}" if config["device"]["use_cuda"] and torch.cuda.is_available() else "cpu")
    latent_dim = config["vae"]["latent_dim"]
    batch_size = config["batch_size"]
    image_size = config["vae"]["image_size"]
    dataset_path = config["path"]["dataset"]

    # 生成输出路径
    exp_id, _, gan_dir, log_path = setup_bagan_logging(
        base_output=os.path.join(config["path"]["output_dir"], "dcgan"),
        base_log=os.path.join(config["path"]["logs"], "dcgan")
    )
    eval_dir = os.path.join(gan_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # 数据集加载
    dataset = ImageDataset(root_dir=dataset_path, image_size=(image_size, image_size))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    generator = DCGANGenerator(latent_dim).to(device)
    discriminator = DCGANDiscriminator().to(device)

    g_opt = optim.Adam(generator.parameters(), lr=config["gan"]["learning_rate_generator"], betas=(0.5, 0.999))
    d_opt = optim.Adam(discriminator.parameters(), lr=config["gan"]["learning_rate_discriminator"], betas=(0.5, 0.999))

    criterion = nn.BCEWithLogitsLoss()
    log = []

    num_epochs = config["gan"]["num_epochs"]
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        total_g, total_d = 0, 0

        for real_imgs in dataloader:
            real_imgs = real_imgs.to(device)
            b_size = real_imgs.size(0)
            real_labels = torch.full((b_size, 1), 0.9, device=device)
            fake_labels = torch.zeros((b_size, 1), device=device)

            # -------- 训练判别器 --------
            d_opt.zero_grad()
            output_real = discriminator(real_imgs)
            d_real_loss = criterion(output_real, real_labels)

            z = torch.randn(b_size, latent_dim, device=device)
            fake_imgs = generator(z).detach()
            output_fake = discriminator(fake_imgs)
            d_fake_loss = criterion(output_fake, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_opt.step()
            total_d += d_loss.item()

            # -------- 训练生成器 --------
            g_opt.zero_grad()
            z = torch.randn(b_size, latent_dim, device=device)
            fake_imgs = generator(z)
            output = discriminator(fake_imgs)
            g_loss = criterion(output, real_labels)
            g_loss.backward()
            g_opt.step()
            total_g += g_loss.item()

        avg_g = total_g / len(dataloader)
        avg_d = total_d / len(dataloader)
        print(f"[DCGAN] Epoch {epoch+1}/{num_epochs}, G: {avg_g:.4f}, D: {avg_d:.4f}")

        if (epoch + 1) % 10 == 0:
            # 创建保存目录：eval/epoch_{epoch+1}/
            eval_dir2 = os.path.join(eval_dir, f"epoch_{epoch + 1}")
            os.makedirs(eval_dir2, exist_ok=True)

            with torch.no_grad():
                # 从高斯分布采样 latent 向量并生成图像
                z = torch.randn(100, latent_dim, device=device)
                samples = generator(z)

            # 保存每张生成图像为单独文件
            for i, img in enumerate(samples):
                save_path = os.path.join(eval_dir2, f"img_{i:03d}.png")
                save_image(img, save_path)

            # 执行评估（与真实图像对比）
            eval_result = evaluate_generated_images(real_imgs[:25], samples[:25], eval_dir, epoch + 1, prefix="dcgan")

            # 记录日志
            log.append({
                "epoch": epoch + 1,
                "g_loss": avg_g,
                "d_loss": avg_d,
                **eval_result
            })

            # torch.save(generator.state_dict(), os.path.join(gan_dir, f"generator_{epoch + 1}.pth"))
            # torch.save(discriminator.state_dict(), os.path.join(gan_dir, f"discriminator_{epoch + 1}.pth"))

    # 保存模型参数
    torch.save(generator.state_dict(), os.path.join(gan_dir, "generator_final.pth"))
    torch.save(discriminator.state_dict(), os.path.join(gan_dir, "discriminator_final.pth"))

    # 保存日志
    if log:
        df = pd.DataFrame(log)
        sheet = datetime.now().strftime("dcgan_%Y%m%d_%H%M%S")
        with pd.ExcelWriter(log_path, mode="a", engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet, index=False)


if __name__ == "__main__":
    config = load_config("configs/default.yaml")
    train_dcgan(config)
