import torch
import os
import numpy as np
import zarr
from multiprocessing.pool import ThreadPool
from noisebase.torch import VideoSampler, Shuffler
from noisebase.projective import FlipRotate, screen_space_normal, motion_vectors, log_depth
from noisebase.compression import decompress_RGBE
from noisebase.data import resolve_data_path, downloader
from urllib.parse import urljoin

class TrainingSampleDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 src, 
                 sequence_idxs, 
                 rng, 
                 flip_rotate,
                 buffers,
                 samples, 
                 data_path,
                 batch_size,
                 **kwargs
                ):
        super().__init__()

        data_path = resolve_data_path(data_path)
        self.files = list(map(lambda i: os.path.join(data_path, src['files'].format(index=i)), sequence_idxs))
        self.frames_per_sequence = src['frames_per_sequence']
        self.rng = rng
        self.flip_rotate = flip_rotate
        self.crop = src['crop']
        self.rendering_height = src['rendering_height']
        self.rendering_width = src['rendering_width']
        self.buffers = buffers
        self.samples = samples
        self.batch_size = batch_size # Just for worker_init_fn


        self.sqidxs = sequence_idxs
    
    def __getitem__(self, idx):
        # idx: sequence_idx * frames_per_sequence + frame_idx
        frame_idx = idx['idx'] % self.frames_per_sequence
        sequence_idx = idx['idx'] // self.frames_per_sequence

        if self.flip_rotate:
            orientation = self.rng.derive(idx['epoch']).integers(sequence_idx, 0, 8)
        else:
            orientation = 7
        flip_rotate = FlipRotate(orientation, self.rendering_height, self.rendering_width, self.crop)

        ds = zarr.group(store = zarr.ZipStore(self.files[sequence_idx], mode='r'))

        forward, up, left, pos, offset, pv = flip_rotate.apply_camera(
            ds['camera_target'][frame_idx], 
            ds['camera_up'][frame_idx],
            ds['camera_position'][frame_idx],
            ds['proj_mat'][frame_idx],
            ds['crop_offset'][frame_idx]
        )

        frame = {
            'view_proj_mat': pv,
            'camera_position': pos,
            'camera_forward': forward,
            'camera_up': up,
            'camera_left': left,
            'crop_offset': offset,
        }

        if frame_idx > 0:
            prev_forward, prev_up, prev_left, prev_pos, prev_offset, prev_pv = flip_rotate.apply_camera(
                ds['camera_target'][frame_idx - 1], 
                ds['camera_up'][frame_idx - 1],
                ds['camera_position'][frame_idx - 1],
                ds['proj_mat'][frame_idx - 1],
                ds['crop_offset'][frame_idx - 1]
            )
            frame['prev_camera'] = {
                'view_proj_mat': prev_pv,
                'camera_position': prev_pos,
                'camera_forward': prev_forward,
                'camera_up': prev_up,
                'camera_left': prev_left,
                'crop_offset': prev_offset,
            }
        else:
            frame['prev_camera'] = frame.copy()
        
        frame['frame_index'] = frame_idx,
        frame['file'] = self.files[sequence_idx]

        if 'w_normal' in self.buffers or 'normal' in self.buffers:
            w_normal = flip_rotate.apply_array(ds['normal'][frame_idx, ..., 0:self.samples]).astype(np.float32)
        
        if 'w_normal' in self.buffers:
            frame['w_normal'] = w_normal

        if 'normal' in self.buffers:
            frame['normal'] = screen_space_normal(w_normal, forward, up, left)
        
        if 'depth' in self.buffers or 'motion' in self.buffers or 'w_position' in self.buffers:
            w_position = flip_rotate.apply_array(ds['position'][frame_idx, ..., 0:self.samples])
        
        if 'motion' in self.buffers or 'w_motion' in self.buffers:
            w_motion = flip_rotate.apply_array(ds['motion'][frame_idx, ..., 0:self.samples])
        
        if 'w_position' in self.buffers:
            frame['w_position'] = w_position
        
        if frame_idx > 0:
            if 'w_motion' in self.buffers:
                frame['w_motion'] = w_motion
            
            if 'motion' in self.buffers:
                motion = motion_vectors(
                    w_position, w_motion, 
                    pv, prev_pv,
                    self.rendering_height, self.rendering_width
                )
                frame['motion'] = np.clip(motion, -5e3, 5e3)
        else:
            if 'w_motion' in self.buffers:
                frame['w_motion'] = np.zeros_like(w_motion)
            
            if 'motion' in self.buffers:
                frame['motion'] = np.zeros_like(w_motion[0:2])
        
        if 'depth' in self.buffers:
            frame['depth'] = log_depth(w_position, pos)
        
        if 'color' in self.buffers:
            exposure = ds['exposure'][frame_idx]
            rgbe = flip_rotate.apply_array(ds['color'][frame_idx, ..., 0:self.samples])
            frame['color'] = decompress_RGBE(rgbe, exposure)
        
        if 'diffuse' in self.buffers:
            frame['diffuse'] = flip_rotate.apply_array(ds['diffuse'][frame_idx, ..., 0:self.samples]).astype(np.float32)
        
        if 'reference' in self.buffers:
            frame['reference'] = flip_rotate.apply_array(ds['reference'][frame_idx])
        
        ds.store.close()

        return frame

    def __getitems__(self, idxs):
        if hasattr(self, 'thread_pool'):
            return self.thread_pool.map(self.__getitem__, idxs)
        else:
            return list(map(self.__getitem__, idxs))

def collate_fixes(batch):
    collated = torch.utils.data.default_collate(batch)
    collated['frame_index'] = collated['frame_index'][0][0]
    return collated

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:  # In the main process
        return
    # Loader pool
    worker_info.dataset.thread_pool = ThreadPool(worker_info.dataset.batch_size)
    
class TrainingSampleLoader_v1(torch.utils.data.DataLoader):
    def __init__(self, 
                 src, # Source options for loading
                 stage, # 'train' or 'val'
                 seed,
                 val_split,
                 shuffle,
                 batch_size,
                 drop_last,
                 num_workers,
                 flip_rotate, # Augmentation
                 get_epoch = None,
                 **kwargs
                ):

        # RNG for shuffling, test-train splitting, augmentations
        rng = Shuffler(seed)

        sequence_idxs = np.arange(src['sequences'])
        val_idxs, train_idxs = rng.split(-1, sequence_idxs, val_split, shuffle, batch_size)
        sequence_idxs = train_idxs if stage == 'train' else val_idxs

        self.__src = src
        self.__kwargs = kwargs

        sampler = VideoSampler(
            batch_size, # Sampler computes correct distributed batch size
            src['frames_per_sequence'], 
            len(sequence_idxs),
            shuffle=shuffle and stage == 'train',
            shuffle_fn=lambda epoch, sequence: rng.shuffle(epoch, sequence),
            drop_last=drop_last,
            get_epoch=get_epoch
        )

        ds = TrainingSampleDataset(
            src = src, 
            sequence_idxs = sequence_idxs, 
            rng = rng,
            flip_rotate = flip_rotate and stage == 'train',
            batch_size=sampler.batch_size,
            **kwargs
        )

        super().__init__(
            ds,
            batch_sampler=sampler, 
            collate_fn=collate_fixes,
            num_workers=num_workers,
            #num_workers=0,
            worker_init_fn=worker_init_fn,
            prefetch_factor=4 if num_workers > 0 else None,
            #multiprocessing_context='spawn', # Fork may break with CUDA, but spawn starts very slowly
            pin_memory=True # Speeed
        )
    
    ### Resumable data loader

    def state_dict(self):
        return self.batch_sampler.cached_state

    def load_state_dict(self, state_dict):
        if state_dict != {}:
            self.batch_sampler.start_epoch_idx = state_dict['start_epoch_idx']
            self.batch_sampler.epoch_idx = state_dict['start_epoch_idx']
            self.batch_sampler.start_sequence_idx = state_dict['start_sequence_idx']
    
    def download(self):
        data_path = resolve_data_path(self.__kwargs['data_path'])
        remote_path = self.__kwargs['data_remote']

        if remote_path == None:
            print(f'Download aborted: the remote path for \"{self.__kwargs["name"]}\" was not provided.')
            print('Consult the dataset\'s documentation for instructions on acquiring it.')
            return

        files = [
            self.__src['files'].format(index=i)
            for i in range(self.__src['sequences'])
        ]

        local_files = list(map(lambda x: os.path.join(data_path, x), files))
        remote_files = list(map(lambda x: urljoin(remote_path, x), files))

        downloader(local_files, remote_files)