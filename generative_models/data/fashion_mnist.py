import lightning as L
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST as MNIST
from torchvision.transforms import v2


class FashionMNISTDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir="./datasets",
        batch_size=64,
        num_workers=2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def prepare_data(self):
        MNIST(self.hparams["data_dir"], download=True)

    def setup(self, stage: str):
        self.train_ds = MNIST(
            self.hparams["data_dir"],
            train=True,
            transform=self.transforms,
        )
        self.val_ds = MNIST(
            self.hparams["data_dir"],
            train=False,
            transform=self.transforms,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
        )
