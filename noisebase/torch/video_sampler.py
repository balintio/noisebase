"""
noisebase.torch.video_sampler
-----------------------------

also available under noisebase.torch
"""

import numpy as np
import torch.distributed as dist

# TODO: could use some docstrings

class VideoSampler():
    """Samples frames sequentially from video datasets
    
    Datasets must be indexed as concatenated sequences of frames
     - Each sequence must have frames_per_sequence frames
     - I.e. the dataset contains the jth frame of the ith
       sequence at i * frames_per_sequence + j
    
    Each sampled batch contains frames at the same 
    index from batch_size number of sequences
    
    This sampler supports batching, shuffling, 
    distributed training, and mid-epoch checkpoints
    """

    def __init__(self, batch_size, frames_per_sequence, num_sequences, 
                 shuffle = False, 
                 drop_last = False, 
                 get_epoch = None, # Override epoch counting
                 shuffle_fn = None # Override shuffle
                ):
        # Assign some stuff
        self.frames_per_sequence = frames_per_sequence
        self.num_sequences = num_sequences
        self.shuffle = shuffle
        self.shuffle_fn = shuffle_fn
        self.drop_last = drop_last
        
        # Distributed sampling, treating single device as a special case
        self.rank = 0
        self.num_replicas = 1

        if dist.is_available() and dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
        
        # Mismatched batch size across replicas could cause issues
        self.max_sequence_idx = self.num_sequences // self.num_replicas * self.num_replicas

        # Assign some more stuff
        self.batch_size = batch_size // self.num_replicas
        self.start_epoch_idx = 0
        self.start_sequence_idx = 0
        self.cached_state = {}
        
        # Default epoch counting
        self.epoch_idx = 0 

        if get_epoch == None:
            self.get_epoch = lambda: self.epoch_idx
        else:
            self.get_epoch = get_epoch

    def checkpoint(self, epoch, batch_idx):
        # Check if we can save a checkpoint BEFORE training on this batch
        # Return state dict for restart if yes

        # Only save between sequences
        if batch_idx % self.frames_per_sequence != 0:
            return False
        
        sequence_idx = batch_idx // self.frames_per_sequence * self.batch_size * self.num_replicas
        #if epoch == self.start_epoch_idx:
            # batch_idx is offset if we started mid-epoch
            
            # update: it actually isn't, it's restored from the ckpt
            # this is a problem if the batch size changes mid-epoch
            #sequence_idx += self.start_sequence_idx
        
        self.cached_state = {'start_epoch_idx': epoch, 'start_sequence_idx': sequence_idx}
        return True

    def __iter__(self):
        # Simple stuff
        epoch = self.get_epoch()

        global_sequence_idxs = np.arange(self.num_sequences)
        if self.shuffle:
            if self.shuffle_fn != None:
                # Externally seeded shuffling
                self.shuffle_fn(epoch, global_sequence_idxs)
            else:
                rng = np.random.default_rng(seed = epoch)
                rng.shuffle(global_sequence_idxs)

        if epoch == self.start_epoch_idx:
            global_sequence_idxs = global_sequence_idxs[self.start_sequence_idx:]

        sequence_idxs = global_sequence_idxs[self.rank:self.max_sequence_idx:self.num_replicas]
        
        # Loop through sequences
        #
        # We take a batch of sequences and yield them frame-by-frame.
        # E.g. For a batch of 8 64-frame-long sequences we give
        # 64 batches, each with 8 frames.
        #
        # Each batch contains frames at the same index from
        # their respective sequences. E.g. frame 0 from each
        # of the 8 sequences in the example above.
        #
        # Sequences may be shuffled but frames within a 
        # sequence are always sequential.
        while True:

            # Finish iteration when we run out of sequences
            epoch_done = False
            epoch_done |= len(sequence_idxs) == 0
            epoch_done |= len(sequence_idxs) < self.batch_size and self.drop_last

            if epoch_done:
                # Default epoch counting
                #  - Bump count when we finish an epoch
                #  - Use max in case we have an older iterator
                #  - Should be OK for most use cases
                #  - Provide a custom get_epoch if you can do better
                #    - e.g. our lightning datamodule
                self.epoch_idx = max(self.epoch_idx, epoch + 1)
                return

            # Take batch
            batch_idxs = sequence_idxs[:self.batch_size]
            sequence_idxs = sequence_idxs[self.batch_size:]

            # Yield indices
            for frame_idx in range(self.frames_per_sequence):
                # Dataset is a list; concatenated sequences of frames
                yield [
                    {
                        'epoch': epoch, 
                        'idx': sequence_idx * self.frames_per_sequence + frame_idx 
                    } 
                    for sequence_idx in batch_idxs
                ]
    
    def __len__(self):
        num_samples = self.num_sequences // self.num_replicas // self.batch_size
        
        if num_samples // self.num_replicas % self.batch_size != 0 and not self.drop_last:
            num_samples += 1
        
        return num_samples * self.frames_per_sequence