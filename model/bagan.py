import torch
import torch.nn as nn
import torch.nn.functional as F
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=512):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim

        # DCGAN 风格的编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 使用全连接层直接获得确定性潜在表示
        self.fc = nn.Linear(512 * 8 * 8, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 512 * 8 * 8)

        # DCGAN 风格的解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        # 编码器部分：提取特征并映射到潜在空间（确定性表示）
        enc_out = self.encoder(x).view(x.size(0), -1)
        latent = self.fc(enc_out)
        # 将潜在向量映射回特征图，再经过解码器生成图像
        z = self.fc_decode(latent).view(x.size(0), 512, 8, 8)
        reconstructed = self.decoder(z)
        return reconstructed


class AEGenerator(nn.Module):
    def __init__(self, decoder, latent_dim, dataset_mean, dataset_std):
        """
        参数说明：
        - decoder: 解码器网络
        - latent_dim: 潜在空间的维度
        - dataset_mean: 在数据集上计算得到的潜在向量均值（torch.Tensor，形状为 (latent_dim,) 或标量）
        - dataset_std: 在数据集上计算得到的潜在向量标准差（torch.Tensor，形状为 (latent_dim,) 或标量）
        """
        super(AEGenerator, self).__init__()
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std
        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)

    def forward(self, batch_size):
        device = next(self.decoder.parameters()).device

        # 将 dataset_mean 和 dataset_std 转移到对应设备，并扩展到 batch_size
        if self.dataset_mean.dim() == 1:
            mean = self.dataset_mean.to(device).unsqueeze(0).expand(batch_size, -1)
        else:
            mean = self.dataset_mean.to(device)
        if self.dataset_std.dim() == 1:
            std = self.dataset_std.to(device).unsqueeze(0).expand(batch_size, -1)
        else:
            std = self.dataset_std.to(device)

        # 根据数据集分布采样 latent 向量：z = mean + N(0,1) * std
        z = mean + torch.randn(batch_size, self.latent_dim, device=device) * std
        # z = torch.randn(batch_size, self.latent_dim, device=device)
        # 可选：增加额外的噪声扰动
        noise = torch.randn_like(z) * 0.1
        z = z + noise

        # 将潜在向量映射回特征图，并生成假图像
        z = self.fc(z).view(batch_size, 512, 8, 8)
        fake_images = self.decoder(z)
        return fake_images

    def random_sample(self, batch_size):
        """
        完全基于标准正态分布采样 latent 向量并生成图像
        """
        device = next(self.decoder.parameters()).device
        z = torch.randn(batch_size, self.latent_dim, device=device)
        z = self.fc(z).view(batch_size, 512, 8, 8)
        fake_images = self.decoder(z)
        return fake_images


# 修改后的 Discriminator（建议同时修改，使其符合 BCEWithLogitsLoss 的要求）
class Discriminator(nn.Module):
    def __init__(self, autoencoder_encoder):
        super(Discriminator, self).__init__()
        self.encoder = autoencoder_encoder
        # 将全连接层输出单个数值（logit），假设图像大小为128x128
        self.fc = nn.Linear(32768, 1)

    def forward(self, x):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        output = self.fc(features)
        return output
