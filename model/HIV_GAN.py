import torch
import torch.nn as nn
import torch.nn.functional as F
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim=512):
        super(VariationalAutoencoder, self).__init__()
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

        # 潜在变量的均值和对数方差
        self.fc_mu = nn.Linear(512 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(512 * 8 * 8, latent_dim)
        self.fc_z = nn.Linear(latent_dim, 512 * 8 * 8)

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

    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # 编码器部分
        enc_out = self.encoder(x).view(x.size(0), -1)
        mu = self.fc_mu(enc_out)
        logvar = self.fc_logvar(enc_out)
        z = self.reparameterize(mu, logvar)
        # 将潜在向量映射回特征图，再经过解码器生成图像
        z = self.fc_z(z).view(x.size(0), 512, 8, 8)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar


class VAEGenerator(nn.Module):
    def __init__(self, decoder, latent_dim):
        super(VAEGenerator, self).__init__()
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.fc_z = nn.Linear(latent_dim, 512 * 8 * 8)

    def forward(self, batch_size):
        z = torch.randn(batch_size, self.latent_dim).to(next(self.decoder.parameters()).device)
        noise = torch.randn_like(z) * 0.1  # 增加随机噪声扰动
        z = z + noise

        # 还原 z 形状并生成假图像
        z = self.fc_z(z).view(batch_size, 512, 8, 8)
        fake_images = self.decoder(z)
        return fake_images


class Discriminator(nn.Module):
    def __init__(self, autoencoder_encoder):
        super(Discriminator, self).__init__()
        self.encoder = autoencoder_encoder
        # 将全连接层输出单个数值（logit）
        self.fc = nn.Linear(32768, 1)  # 假设图像大小为128x128

    def forward(self, x):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        output = self.fc(features)
        # 不使用 softmax/sigmoid，这里直接输出 logit，后面用 BCEWithLogitsLoss
        # output = F.softmax(output, dim=1)
        return output



