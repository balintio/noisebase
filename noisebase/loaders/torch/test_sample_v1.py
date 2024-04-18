from multiprocessing.pool import ThreadPool
import os
import numpy as np
import zarr
from noisebase.projective import normalize, screen_space_normal, motion_vectors, log_depth
from noisebase.compression import decompress_RGBE
from noisebase.data import resolve_data_path, ACES, downloader
import torch
from tqdm import tqdm
import imageio.v3 as iio
from urllib.parse import urljoin

def compute_camera(target, up, pos):
    # simplified version of FlipRotate.apply_camera

    W = normalize(target - pos) # forward
    U = normalize(np.cross(W, up)) # right
    V = normalize(np.cross(U, W)) # up 

    return W, V, U, pos,

class TestSampleDataset(torch.utils.data.Dataset):
    def __init__(self,
                 sequence,
                 src,
                 buffers,
                 data_path,
                 samples,
                 **kwargs
                ):
        super().__init__()
            
        data_path = resolve_data_path(data_path)
        self.files = list(map(
            lambda i: os.path.join(
                data_path, 
                src['files'].format(index=i, sequence_name=sequence['name'])
            ), 
            range(sequence['frames'])
        ))

        self.rendering_height = src['rendering_height']
        self.rendering_width = src['rendering_width']
        self.buffers = buffers
        self.samples = samples

    def __getitem__(self, idx):
        ds = zarr.group(store = zarr.ZipStore(self.files[idx], mode='r'))

        forward, up, left, pos, = compute_camera(
            np.array(ds['camera_target']), 
            np.array(ds['camera_up']),
            np.array(ds['camera_position']),
        )
        pv = np.array(ds['view_proj_mat'])

        frame = {
            'view_proj_mat': pv,
            'camera_position': pos,
            'camera_forward': forward,
            'camera_up': up,
            'camera_left': left,
            'crop_offset': np.array([28, 0], dtype=np.int32),
        }

        if idx > 0:
            pds = zarr.group(store = zarr.ZipStore(self.files[idx-1], mode='r'))
            prev_forward, prev_up, prev_left, prev_pos = compute_camera(
                np.array(pds['camera_target']), 
                np.array(pds['camera_up']),
                np.array(pds['camera_position']),
            )
            prev_pv = np.array(pds['view_proj_mat'])

            frame['prev_camera'] = {
                'view_proj_mat': prev_pv,
                'camera_position': prev_pos,
                'camera_forward': prev_forward,
                'camera_up': prev_up,
                'camera_left': prev_left,
                'crop_offset': np.array([28, 0], dtype=np.int32),
            }
            pds.store.close()
        else:
            frame['prev_camera'] = frame.copy()
        
        frame['frame_index'] = np.array((idx,), dtype=np.int32),
        #frame['file'] = file TODO: load strings to pytorch

        if 'w_normal' in self.buffers or 'normal' in self.buffers:
            w_normal = ds['normal'][..., 28:-28, :, 0:self.samples].astype(np.float32)
        
        if 'w_normal' in self.buffers:
            frame['w_normal'] = w_normal

        if 'normal' in self.buffers:
            frame['normal'] = screen_space_normal(w_normal, forward, up, left)
        
        if 'depth' in self.buffers or 'motion' in self.buffers or 'w_position' in self.buffers:
            w_position = ds['position'][..., 28:-28, :, 0:self.samples]
        
        if 'motion' in self.buffers or 'w_motion' in self.buffers:
            w_motion = ds['motion'][..., 28:-28, :, 0:self.samples]
        
        if 'w_position' in self.buffers:
            frame['w_position'] = w_position
        
        if idx > 0:
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
            exposure = np.array(ds['exposure'])
            rgbe = ds['color'][..., 28:-28, :, 0:self.samples]
            frame['color'] = decompress_RGBE(rgbe, exposure)
        
        if 'diffuse' in self.buffers:
            frame['diffuse'] = ds['diffuse'][..., 28:-28, :, 0:self.samples].astype(np.float32)
        
        if 'reference' in self.buffers:
            frame['reference'] = np.array(ds['reference'][:, 28:-28, :])
        
        ds.store.close()

        return frame

    def __len__(self):
        return len(self.files)

class Sequence:
    def __init__(self,
                 src,
                 sequence,
                 save_dir,
                 output,
                 **kwargs
                ):
        self.src = src
        self.sequence = sequence
        self.save_dir = save_dir
        self.output = output
        self.kwargs = kwargs
        self.save_pool = ThreadPool(32)
        self.joined_pool = False

        file = os.path.join(
            self.save_dir, 
            self.output.format(sequence_name=self.sequence['name'], index=0)
        )
        os.makedirs(os.path.dirname(file), exist_ok=True)
    
    @property
    def frames(self):
        ds = TestSampleDataset(
            self.sequence,
            self.src,
            **self.kwargs
        )

        loader = torch.utils.data.DataLoader(
            ds,
            batch_size = 1,
            shuffle = False,
            num_workers = 8,
            pin_memory = True,
        )
        return loader

    def to_torch(self, data, device):
        return {
            #key: torch.from_numpy(value).unsqueeze(0).to(device)
            key: value.to(device)
            #if isinstance(value, np.ndarray)
            if isinstance(value, torch.Tensor)
            else self.to_torch(value, device) 
            if isinstance(value, dict) 
            else value
            for key, value in data.items()
        }

    def save(self, i, image):
        file = os.path.join(
            self.save_dir, 
            self.output.format(sequence_name=self.sequence['name'], index=i)
        )

        image = np.transpose(image.cpu().numpy()[0], (1,2,0))
        image = (ACES(image)*255).astype(np.uint8)

        self.save_pool.apply_async(iio.imwrite, [file, image])
    
    def join(self):
        if not self.joined_pool:
            self.joined_pool = True
            self.save_pool.close()
            self.save_pool.join()
        else:
            pass
    
    def __del__(self):
        if not self.joined_pool:
            self.join()

class TestSampleLoader_v1():
    def __init__(self, 
                 src, # Source options for loading
                 **kwargs
                ):
        
        self.src = src
        self.kwargs = kwargs
    
    def save_dir(self, save_dir):
        self.kwargs['save_dir'] = save_dir
    
    def __iter__(self):
        return map(lambda x: Sequence(src=self.src, sequence=x, **self.kwargs), self.src['sequences'])
    
    def save_reference(self):
        data_path = resolve_data_path(self.kwargs['data_path'])

        ref_saver = TestSampleLoader_v1(
            src=self.src,
            data_path=data_path,
            save_dir=data_path,
            output=self.kwargs['reference'],
            buffers=['reference'],
            samples=0
        )
        for sequence in tqdm(ref_saver, total=len(self.src.sequences), unit='seq', desc='Processing sequences'):
            for i, frame in tqdm(enumerate(sequence.frames), leave=False, total=sequence.sequence['frames'], unit='frame', desc=sequence.sequence['name']):
                sequence.save(i, frame['reference'])
            sequence.join()
    
    # Required for every test dataset
    def test_metadata(self):
        data_path = resolve_data_path(self.kwargs['data_path'])

        metadata = {
            'sequences': [
                {
                    'name': seq['name'],
                    'frames': seq['frames'],
                    'height': 1024,
                    'width': 1920,
                    'fps': 30,
                }
                for seq in self.src.sequences
            ],
            'test_format': self.kwargs['output'],
            'reference_format': self.kwargs['reference'],
            'reference_path': data_path,
            'warmup': self.kwargs['warmup'],
            'metric_format': self.kwargs['metrics']
        }
        return metadata

    def download(self):
        data_path = resolve_data_path(self.kwargs['data_path'])
        remote_path = self.kwargs['data_remote']

        if remote_path == None:
            print(f'Download aborted: the remote path for \"{self.kwargs["name"]}\" was not provided.')
            print('Consult the dataset\'s documentation for instructions on acquiring it.')
            return

        files = [
            self.src['files'].format(index=i, sequence_name=sequence['name'])
            for sequence in self.src.sequences
            for i in range(sequence['frames'])
        ]

        local_files = list(map(lambda x: os.path.join(data_path, x), files))
        remote_files = list(map(lambda x: urljoin(remote_path, x), files))

        downloader(local_files, remote_files)