import lightning as L

import torch
import torch.nn as nn
from torchvision.utils import make_grid


class DCGAN(L.LightningModule):
    def __init__(self, n_latent=100, lr=1e-5):
        super().__init__()
        self.save_hyperparameters()

        self.gen = Generator(n_latent)
        self.dis = Discriminator()

        self.fixed_noise = self.sample_noise(64, n_latent)

        self.criterion = nn.BCEWithLogitsLoss()

        self.automatic_optimization = False

    @classmethod
    def sample_noise(cls, batch_size, n_latent):
        return torch.normal(mean=0, std=1, size=(batch_size, n_latent, 1, 1))

    def training_step(self, batch, batch_idx):
        optimizer_g, optimizer_d = self.optimizers()

        imgs, _ = batch
        batch_size = imgs.shape[0]
        real_labels = torch.ones((batch_size, 1), device=self.device)
        fake_labels = torch.zeros((batch_size, 1), device=self.device)
        gen_noise = self.sample_noise(batch_size, self.hparams["n_latent"]).to(
            self.device
        )
        pred_gen = self.gen(gen_noise)

        # Training discriminator on real data
        pred_real = self.dis(imgs)
        loss_real = self.criterion(pred_real, real_labels)

        # Training discriminator on fake data
        pred_fake = self.dis(pred_gen.detach())
        loss_fake = self.criterion(pred_fake, fake_labels)

        loss_dis = loss_real + loss_fake

        optimizer_d.zero_grad()
        self.manual_backward(loss_dis)
        optimizer_d.step()

        # Training generator
        pred = self.dis(pred_gen)
        loss_gen = self.criterion(pred, real_labels)

        optimizer_g.zero_grad()
        self.manual_backward(loss_gen)
        optimizer_g.step()

        self.log_dict({"loss_gen": loss_gen, "loss_dis": loss_dis}, prog_bar=True)

    def on_train_epoch_end(self):
        pred = self.gen(self.fixed_noise.to(self.device))
        imgs = make_grid(pred, 8)

        for logger in self.loggers:
            if isinstance(logger, L.pytorch.loggers.wandb.WandbLogger):
                logger.log_image(key="samples", images=[imgs])

    def configure_optimizers(self):
        optimizer_g = torch.optim.Adam(self.gen.parameters(), 1e-4, betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(self.dis.parameters(), 1e-4, betas=(0.5, 0.999))
        return optimizer_g, optimizer_d


class Generator(nn.Module):
    def __init__(self, n_latent):
        super().__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(n_latent, 512, 5, 2),
            nn.BatchNorm2d(512),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(512, 256, 5),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 4, 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, 3),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 16, 5),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 8, 3),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 4, 3),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.flatten(out)
        out = self.fc(out)
        return out
