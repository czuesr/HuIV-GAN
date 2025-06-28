import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
from torchvision.utils import save_image
def setup_bagan_logging(base_output="results/outputs/bagan", base_log="results/logs/bagan"):
    os.makedirs(base_output, exist_ok=True)
    os.makedirs(base_log, exist_ok=True)

    existing = [d for d in os.listdir(base_output) if d.startswith("exp") and d[3:].isdigit()]
    next_id = max([int(d[3:]) for d in existing], default=0) + 1
    exp_id = f"exp{next_id}"

    ae_save_dir = os.path.join(base_output, exp_id, "ae")
    gan_save_dir = os.path.join(base_output, exp_id, "gan")
    os.makedirs(ae_save_dir, exist_ok=True)
    os.makedirs(gan_save_dir, exist_ok=True)

    log_dir = os.path.join(base_log, exp_id)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.xlsx")
    if not os.path.exists(log_path):
        pd.DataFrame().to_excel(log_path, index=False)

    return exp_id, ae_save_dir, gan_save_dir, log_path

def compute_latent_distribution(encoder, fc_layer, dataloader, device):
    encoder.eval()
    fc_layer.eval()
    latents = []
    with torch.no_grad():
        for imgs in dataloader:
            imgs = imgs.to(device)
            feat = encoder(imgs).view(imgs.size(0), -1)
            z = fc_layer(feat)
            latents.append(z)
    all_latent = torch.cat(latents, dim=0)
    return all_latent.mean(dim=0), all_latent.std(dim=0)




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

def train_ae(model, dataloader, optimizer, device, save_dir, log_path, epochs=50):
    import torch.nn as nn
    from torchvision.utils import save_image

    criterion = nn.MSELoss()
    log = []
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for imgs in dataloader:
            imgs = imgs.to(device)
            recons = model(imgs)
            loss = criterion(recons, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[AE] Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")

        if (epoch+1) % 10 == 0:
            with torch.no_grad():
                sample = model(imgs[:25])
                eval_dir = os.path.join(save_dir, "eval")
                os.makedirs(eval_dir, exist_ok=True)
                metrics = evaluate_generated_images(imgs[:25], sample, eval_dir, epoch+1, prefix="ae")
                save_image(sample, os.path.join(eval_dir, f"samples_epoch{epoch+1}.png"), nrow=5)
                log.append({"epoch": epoch+1, "ae_loss": avg_loss, **metrics})

    torch.save(model.encoder.state_dict(), os.path.join(save_dir, "ae_encoder.pth"))
    torch.save(model.decoder.state_dict(), os.path.join(save_dir, "ae_decoder.pth"))
    torch.save(model.fc.state_dict(), os.path.join(save_dir, "ae_fc.pth"))

    if log:
        df = pd.DataFrame(log)
        sheet = datetime.now().strftime("ae_%Y%m%d_%H%M%S")
        with pd.ExcelWriter(log_path, mode='a', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet, index=False)


def train_gan(generator, discriminator, dataloader, g_opt, d_opt, device,
              latent_dim, strategy, save_dir, log_path, epochs=50):
    criterion = nn.BCEWithLogitsLoss()
    log = []

    for epoch in range(epochs):
        total_g, total_d = 0, 0

        for imgs in dataloader:
            imgs = imgs.to(device)
            b = imgs.size(0)
            real_label = torch.full((b, 1), 0.9, device=device)
            fake_label = torch.zeros((b, 1), device=device)

            # 判别器训练
            d_opt.zero_grad()
            real_out = discriminator(imgs)
            d_real_loss = criterion(real_out, real_label)

            fake_imgs = generator(b).detach() if strategy == "z_dist" else generator.random_sample(b)
            fake_out = discriminator(fake_imgs)
            d_fake_loss = criterion(fake_out, fake_label)
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_opt.step()
            total_d += d_loss.item()

            # 生成器训练
            g_opt.zero_grad()
            fake_imgs = generator(b) if strategy == "z_dist" else generator.random_sample(b)
            g_out = discriminator(fake_imgs)
            g_loss = criterion(g_out, real_label)
            g_loss.backward()
            g_opt.step()
            total_g += g_loss.item()

        avg_g = total_g / len(dataloader)
        avg_d = total_d / len(dataloader)
        print(f"[GAN-{strategy}] Epoch {epoch+1}/{epochs} G: {avg_g:.4f}, D: {avg_d:.4f}")

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                # 生成100张图像
                sample = generator(100) if strategy == "z_dist" else generator.random_sample(100)

                # 创建保存路径：eval/epoch_{epoch+1}/
                eval_dir = os.path.join(save_dir, "eval", f"epoch_{epoch + 1}")
                os.makedirs(eval_dir, exist_ok=True)

                # 执行评估（你可能只需传前25张用于评估）
                metrics = evaluate_generated_images(imgs[:25], sample[:25], eval_dir, epoch + 1,
                                                    prefix=f"gan_{strategy}")

                # 保存每张图像为单独文件
                for i, img in enumerate(sample):
                    save_path = os.path.join(eval_dir, f"img_{i:03d}.png")
                    save_image(img, save_path)

                # 记录日志
                log.append({
                    "epoch": epoch + 1,
                    "g_loss": avg_g,
                    "d_loss": avg_d,
                    **metrics
                })

                # torch.save(generator.state_dict(), os.path.join(save_dir, f"generator_{strategy}_{epoch + 1}.pth"))
                # torch.save(discriminator.state_dict(), os.path.join(save_dir, f"discriminator_{strategy}_{epoch + 1}.pth"))

    torch.save(generator.state_dict(), os.path.join(save_dir, f"generator_{strategy}_final.pth"))
    torch.save(discriminator.state_dict(), os.path.join(save_dir, f"discriminator_{strategy}_final.pth"))

    if log:
        df = pd.DataFrame(log)
        sheet = datetime.now().strftime(f"gan_{strategy}_%Y%m%d_%H%M%S")
        with pd.ExcelWriter(log_path, mode='a', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet, index=False)


import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.bagan import Autoencoder, AEGenerator, Discriminator

import os
import yaml

def load_config(config_path):
    with open("configs/default.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    # ======= 加载配置 =======
    config = load_config("configs/default.yaml")
    vae_cfg = config["vae"]
    gan_cfg = config["gan"]
    path_cfg = config["path"]
    use_cuda = config["device"]["use_cuda"]
    batch_size = config["batch_size"]
    latent_dim = vae_cfg["latent_dim"]
    image_size = vae_cfg["image_size"]
    dataset_path = path_cfg["dataset"]

    device = torch.device(f"cuda:{config['device']['gpu_id']}" if use_cuda and torch.cuda.is_available() else "cpu")

    # ======= 加载数据集 =======
    from dataset import ImageDataset  # 按你的文件名修改

    dataset = ImageDataset(root_dir=dataset_path, image_size=(image_size, image_size))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ======= 创建输出路径 =======
    exp_id, ae_dir, gan_dir, log_path = setup_bagan_logging(
        base_output=os.path.join(path_cfg["output_dir"], "bagan"),
        base_log=os.path.join(path_cfg["logs"], "bagan")
    )

    # ======= 训练 Autoencoder（VAE） =======
    ae = Autoencoder(latent_dim=latent_dim).to(device)
    ae_optimizer = optim.Adam(
        ae.parameters(),
        lr=vae_cfg["learning_rate"],
        betas=(vae_cfg["beta1"], vae_cfg["beta2"]),
        weight_decay=vae_cfg["weight_decay"]
    )
    train_ae(
        model=ae,
        dataloader=dataloader,
        optimizer=ae_optimizer,
        device=device,
        save_dir=ae_dir,
        log_path=log_path,
        epochs=vae_cfg["num_epochs"]
    )

    # ======= 计算潜在分布 =======
    mean, std = compute_latent_distribution(ae.encoder, ae.fc, dataloader, device)

    # ======= 两种策略分别训练 GAN =======
    # for strategy in ["z_dist", "random"]:
    for strategy in ["z_dist"]:
        print(f"\n[开始训练 GAN - 策略: {strategy}]")

        # 重新加载 AE 模块的全部参数，并提取
        ae_reloaded = Autoencoder(latent_dim=latent_dim).to(device)
        ae_reloaded.decoder.load_state_dict(torch.load(os.path.join(ae_dir, "ae_decoder.pth")))
        ae_reloaded.encoder.load_state_dict(torch.load(os.path.join(ae_dir, "ae_encoder.pth")))
        ae_reloaded.fc.load_state_dict(torch.load(os.path.join(ae_dir, "ae_fc.pth")))

        decoder = ae_reloaded.decoder
        encoder = ae_reloaded.encoder
        fc_layer = ae_reloaded.fc

        # Generator 初始化
        if strategy == "z_dist":
            generator = AEGenerator(decoder, latent_dim, mean, std).to(device)
        else:
            generator = AEGenerator(decoder, latent_dim, torch.zeros_like(mean), torch.ones_like(std)).to(device)

        # Discriminator 使用共享 encoder
        discriminator = Discriminator(encoder).to(device)

        g_optimizer = optim.Adam(generator.parameters(), lr=gan_cfg["learning_rate_generator"], betas=(0.5, 0.999))
        d_optimizer = optim.Adam(discriminator.parameters(), lr=gan_cfg["learning_rate_discriminator"], betas=(0.5, 0.999))

        # GAN 训练
        train_gan(
            generator=generator,
            discriminator=discriminator,
            dataloader=dataloader,
            g_opt=g_optimizer,
            d_opt=d_optimizer,
            device=device,
            latent_dim=latent_dim,
            strategy=strategy,
            save_dir=gan_dir,
            log_path=log_path,
            epochs=gan_cfg["num_epochs"]
        )


if __name__ == "__main__":
    main()
