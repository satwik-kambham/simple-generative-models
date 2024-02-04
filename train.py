import lightning as L
from lightning.pytorch.loggers import WandbLogger
from generative_models.models.vae import VAE
from generative_models.data.fashion_mnist import FashionMNISTDataModule

dm = FashionMNISTDataModule()
model = VAE()
logger = WandbLogger(project="VAEs")
trainer = L.Trainer(max_epochs=20, logger=logger)
trainer.fit(model, dm)
