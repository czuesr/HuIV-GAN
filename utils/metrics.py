import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import models, transforms
from skimage.metrics import structural_similarity as ssim
from scipy.linalg import sqrtm
import lpips
import os

# =============================
# 图像预处理函数
# =============================
def preprocess_image(img):
    """将图像转为 [0, 1] 范围的 float32 numpy"""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if img.ndim == 4:
        img = img[0]
    img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    img = ((img + 1) / 2.0).clip(0, 1)  # [-1,1] -> [0,1]
    return img.astype(np.float32)

# =============================
# PSNR
# =============================
def compute_psnr(img1, img2):
    img1 = preprocess_image(img1) * 255
    img2 = preprocess_image(img2) * 255
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

# =============================
# SSIM
# =============================
def compute_ssim(img1, img2):
    img1 = preprocess_image(img1)
    img2 = preprocess_image(img2)
    return ssim(img1, img2, channel_axis=2, data_range=1.0)

# =============================
# FID
# =============================
class InceptionFeatureExtractor:
    def __init__(self):
        inception = models.inception_v3(weights=None)
        state_dict = torch.load("checkpoints/inception_v3.pth")
        inception.load_state_dict(state_dict)
        inception.fc = torch.nn.Identity()
        inception.eval()
        self.model = inception.cuda()

    def extract(self, img_tensor):
        if isinstance(img_tensor, tuple):  # 来自 ImageFolder
            img_tensor = img_tensor[0]
        img_tensor = F.interpolate(img_tensor, size=(299, 299), mode='bilinear', align_corners=False)
        img_tensor = (img_tensor - 0.5) / 0.5  # Normalize manually to [-1, 1]
        with torch.no_grad():
            feat = self.model(img_tensor.cuda())
        return feat

def compute_fid(real_feats, gen_feats):
    mu1 = real_feats.mean(0).cpu().numpy()
    mu2 = gen_feats.mean(0).cpu().numpy()
    sigma1 = np.cov(real_feats.cpu().numpy(), rowvar=False)
    sigma2 = np.cov(gen_feats.cpu().numpy(), rowvar=False)

    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real
    if np.any(np.isnan(covmean)):
        print("[FID] sqrtm 数值不稳定，协方差乘积产生 NaN")
        return -0.0001

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return max(fid, 0.0)

# =============================
# LPIPS 多样性评分（平均距离）
# =============================
def compute_lpips_diversity(images, net='alex'):
    """images: list of torch.Tensor, each shape [3, H, W]"""
    loss_fn = lpips.LPIPS(net=net).cuda()
    n = len(images)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = loss_fn(images[i].unsqueeze(0).cuda(), images[j].unsqueeze(0).cuda())
            dists.append(d.item())
    return np.mean(dists)

# =============================
# 批量特征提取 + FID 计算接口
# =============================
def batch_extract_features(dataloader, model):
    """提取 dataloader 中所有图像的 Inception 特征"""
    feats = []
    for img_batch in dataloader:
        imgs = img_batch[0]  # ImageFolder 返回 (image, label)
        feats.append(model.extract(imgs.cuda()).detach().cpu())
    return torch.cat(feats, dim=0)

# =============================
# 示例统一评估接口
# =============================
def evaluate_pairwise_metrics(real_img, fake_img):
    return {
        "PSNR": compute_psnr(real_img, fake_img),
        "SSIM": compute_ssim(real_img, fake_img),
    }

def evaluate_fid_from_loaders(real_loader, fake_loader):
    model = InceptionFeatureExtractor()
    real_feats = batch_extract_features(real_loader, model)
    fake_feats = batch_extract_features(fake_loader, model)
    return compute_fid(real_feats, fake_feats)

__all__ = [
    'compute_psnr', 'compute_ssim', 'compute_fid', 'compute_lpips_diversity',
    'evaluate_pairwise_metrics', 'evaluate_fid_from_loaders'
]  # 供外部模块导入使用

