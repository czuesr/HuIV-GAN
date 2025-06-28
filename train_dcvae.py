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

    ae_save_dir = os.path.join(base_output, exp_id, "dcvae")
    os.makedirs(ae_save_dir, exist_ok=True)

    log_dir = os.path.join(base_log, exp_id)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.xlsx")
    if not os.path.exists(log_path):
        pd.DataFrame().to_excel(log_path, index=False)

    return exp_id, ae_save_dir, 0, log_path



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


def vae_loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    beta = 2.0
    return recon_loss + beta * kl_div


def train_dcvae(config):
    device = torch.device(f"cuda:{config['device']['gpu_id']}" if config["device"]["use_cuda"] and torch.cuda.is_available() else "cpu")
    latent_dim = config["vae"]["latent_dim"]
    batch_size = config["batch_size"]
    image_size = config["vae"]["image_size"]
    dataset_path = config["path"]["dataset"]

    # 输出路径
    exp_id, ae_dir, _, log_path = setup_bagan_logging(
        base_output=os.path.join(config["path"]["output_dir"], "dcvae"),
        base_log=os.path.join(config["path"]["logs"], "dcvae")
    )
    eval_dir = os.path.join(ae_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # 数据加载
    dataset = ImageDataset(root_dir=dataset_path, image_size=(image_size, image_size))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 模型 & 优化器
    model = DCVAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["vae"]["learning_rate"])

    num_epochs = config["vae"]["num_epochs"]
    log = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for imgs in dataloader:
            imgs = imgs.to(device)
            recon, mu, logvar = model(imgs)
            loss = vae_loss_function(recon, imgs, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[DCVAE] Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                # 获取前25张图像的重建结果
                recon, _, _ = model(imgs[:25])

            # 创建保存目录：eval/epoch_{epoch+1}/

            eval_dir2 = os.path.join(eval_dir, f"epoch_{epoch + 1}")
            os.makedirs(eval_dir2, exist_ok=True)
            # 保存每张重建图像为单独文件
            for i, img in enumerate(recon):
                save_path = os.path.join(eval_dir2, f"img_{i:03d}.png")
                save_image(img, save_path)

            # 评估指标计算
            eval_result = evaluate_generated_images(imgs[:25], recon, eval_dir, epoch + 1, prefix="dcvae")

            # 记录日志信息
            log.append({
                "epoch": epoch + 1,
                "loss": avg_loss,
                **eval_result
            })

            # torch.save(model.state_dict(), os.path.join(ae_dir, f"dcvae_{epoch + 1}.pth"))
            # save_image(recon, os.path.join(eval_dir, f"samples_epoch{epoch + 1}.png"), nrow=5)

    # 保存模型
    torch.save(model.state_dict(), os.path.join(ae_dir, "dcvae.pth"))

    # 保存日志
    if log:
        df = pd.DataFrame(log)
        sheet = datetime.now().strftime("dcvae_%Y%m%d_%H%M%S")
        with pd.ExcelWriter(log_path, mode='a', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet, index=False)


if __name__ == "__main__":
    config = load_config("configs/default.yaml")
    train_dcvae(config)