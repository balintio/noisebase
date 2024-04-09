import lightning as L
from noisebase.loaders.torch import TrainingSampleLoader_v1 as PytorchDataloader

class TrainingSampleLoader_v1(L.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.loader_args = kwargs

    def setup(self, stage):
        # We don't want pytorch lightning to mess with the sampler
        self.trainer._accelerator_connector.use_distributed_sampler = False

    def train_dataloader(self):
        self.loader_args['stage'] = 'train'
        return PytorchDataloader(get_epoch=lambda: self.trainer.current_epoch, **self.loader_args)
    
    def val_dataloader(self):
        self.loader_args['stage'] = 'val'
        return PytorchDataloader(get_epoch=lambda: self.trainer.current_epoch, **self.loader_args)