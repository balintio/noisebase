from lightning.pytorch.callbacks import ModelCheckpoint

class MidepochCheckpoint(ModelCheckpoint):
    # Extends ModelCheckpoint to schedule mid-epoch checkpoints

    # ONLY saves during training. Validation is not resumable
    # as lightning doesn't save tensorboard logging state

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__pending_checkpoint_path = None

    def _save_checkpoint(self, trainer, filepath):
        # Update pending checkpoint
        # Wait for sampler to allow checkpointing
        self.__pending_checkpoint_path = filepath
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Call parent method first in case it schedules a checkpoint
        super().on_train_batch_start(trainer, pl_module, batch, batch_idx)
        if self.__pending_checkpoint_path != None:
            # Ask sampler if it allows checkpointing before this batch
            can_save = trainer.train_dataloader.batch_sampler.checkpoint(
                trainer.current_epoch, 
                batch_idx
            )
            if can_save:
                # State is cached by sampler
                # Save with dataloader
                super()._save_checkpoint(trainer, self.__pending_checkpoint_path)
                trainer.train_dataloader.batch_sampler.cached_state = {} # delete cache
                self.__pending_checkpoint_path = None
    
    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        if self.__pending_checkpoint_path != None:
            # We can always save between epochs
            # No need to save sampler state
            super()._save_checkpoint(trainer, self.__pending_checkpoint_path)
            self.__pending_checkpoint_path = None