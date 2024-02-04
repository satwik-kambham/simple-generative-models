import lightning as L

import torch
import torch.nn as nn
from torchvision.utils import make_grid


class VAE(L.LightningModule):
    def __init__(self, latent_size=10, kld_weight=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(latent_size)
        self.decoder = Decoder(latent_size)

        self.sample_latent = torch.randn(64, latent_size)

        self.criterion = nn.MSELoss()

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = torch.randn_like(torch.exp(0.5 * log_var)) + mu
        out = self.decoder(z)
        return out, mu, log_var

    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        pred, mu, log_var = self(imgs)

        recon_loss = self.criterion(pred, imgs)
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

        loss = recon_loss + kld_loss * self.hparams["kld_weight"]
        self.log("train_recon_loss", recon_loss)
        self.log("train_kld_loss", kld_loss)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, _ = batch
        self.val_imgs = imgs
        pred, mu, log_var = self(imgs)

        recon_loss = self.criterion(pred, imgs)
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

        loss = recon_loss + kld_loss * self.hparams["kld_weight"]
        self.log("val_recon_loss", recon_loss)
        self.log("val_kld_loss", kld_loss)
        self.log("val_loss", loss)

    def on_validation_epoch_end(self):
        val_imgs = make_grid(self.val_imgs, 8)
        pred, _, _ = self(self.val_imgs)
        pred_imgs = make_grid(pred, 8)
        samples = self.decoder(self.sample_latent.to(self.device))
        sample_imgs = make_grid(samples, 8)

        for logger in self.loggers:
            if isinstance(logger, L.pytorch.loggers.wandb.WandbLogger):
                logger.log_image(key="original", images=[val_imgs])
                logger.log_image(key="pred", images=[pred_imgs])
                logger.log_image(key="samples", images=[sample_imgs])

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)

    def sample(self):
        sample_latent = torch.randn(64, latent_size).to(self.device)
        pred = self.decoder(sample_latent)
        imgs = make_grid(pred, 8)
        return grid


class Encoder(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 512, 5),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(1600, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.mu_fc = nn.Linear(256, latent_size)
        self.log_var_fc = nn.Linear(256, latent_size)

    def forward(self, x):
        out = self.conv(x)
        out = self.flatten(out)
        out = self.fc(out)
        mu = self.mu_fc(out)
        log_var = self.log_var_fc(out)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1600),
            nn.ReLU(),
        )
        self.convT = nn.Sequential(
            nn.ConvTranspose2d(64, 256, 5, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 512, 3, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, padding=1),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        out = self.fc(x)
        out = out.view(-1, 64, 5, 5)
        out = self.convT(out)
        return out
