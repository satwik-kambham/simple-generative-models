import lightning as L
from lightning.pytorch.loggers import WandbLogger
from generative_models.models.dcgan import DCGAN
from generative_models.data.fashion_mnist import FashionMNISTDataModule

dm = FashionMNISTDataModule()
model = DCGAN()
logger = WandbLogger(project="DCGAN")
trainer = L.Trainer(max_epochs=20, logger=logger)
trainer.fit(model, dm)
