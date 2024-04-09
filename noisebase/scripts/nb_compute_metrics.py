from noisebase import Noisebase
from noisebase.metrics.fvvdp import nb_fvvdp
from noisebase.metrics.flip import flip
from pyfvvdp.fvvdp import reshuffle_dims
import pytorch_msssim
import argparse
import os
import imageio.v3 as iio
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
import numpy as np
import torch
import json


def srgb_to_lum(image):
    image = torch.pow(image, 2.2)
    return 0.2126 * image[:, 0:1] + 0.7152 * image[:, 1:2] + 0.0722 * image[:, 2:3]

def np_to_torch(hwc):
    t_hwc = torch.tensor(hwc)
    frame_t = reshuffle_dims(t_hwc, in_dims='HWC', out_dims="NCHW")
    frame_t = frame_t.to(torch.float32)/255

    return frame_t

def ssim(test, reference):
    return float(pytorch_msssim.ssim(srgb_to_lum(test), srgb_to_lum(reference), data_range=1.0))

def msssim(test, reference):
    return float(pytorch_msssim.ms_ssim(srgb_to_lum(test), srgb_to_lum(reference), data_range=1.0))

def mse(test, reference):
    return float(torch.mean((test - reference) ** 2))

def psnr(test, reference):
    return float(10 * np.log10(1 / mse(test, reference)))

def rmae(test, reference):
    return float(torch.mean(torch.abs(test - reference) / (torch.abs(reference) + 0.01))) / 3

class TemporalMetricAdapter:
    def __init__(self, spatial_metric):
        self.spatial_metric = spatial_metric
        self.prev_test = None
        self.prev_ref = None
        self.N = 0
        self.sum = 0.0
    
    def feed(self, test, reference):
        if self.prev_test is not None:
            self.sum += self.spatial_metric(test - self.prev_test, reference - self.prev_ref)
            self.N += 1
        
        self.prev_test = test
        self.prev_ref = reference
    
    def compute(self):
        return self.sum / self.N


def main():

    # Arguments
    parser = argparse.ArgumentParser(description="Compute metrics on a test dataset.")
    parser.add_argument('dataset', type=str, help='Input test dataset')
    parser.add_argument('outputs', type=str, help='Output directory of the method being evaluated')
    parser.add_argument('--data_path', type=str, default=None, help='Custom data path')
    args = parser.parse_args()

    if not os.path.isdir(args.outputs):
        print(f"The output directory provided does not exist or is not a directory: {args.outputs}")
        exit(1)

    test_path = args.outputs
    test_set = Noisebase(args.dataset, {
        'data_path': args.data_path
    })

    # Test setup
    metadata = test_set.test_metadata()
    loader_pool = ThreadPool(32)

    for sequence in tqdm(metadata['sequences'], unit='seq', desc='Processing sequences'):
        
        # Loaders
        test_stream = loader_pool.imap(
            iio.imread, 
            map(
                lambda i: os.path.join(
                    test_path,
                    metadata['test_format'].format(index=i, sequence_name=sequence['name'])
                ), 
                range(metadata['warmup'], sequence['frames'])
            ),
            chunksize=8
        )

        reference_stream = loader_pool.imap(
            iio.imread, 
            map(
                lambda i: os.path.join(
                    metadata['reference_path'],
                    metadata['reference_format'].format(index=i, sequence_name=sequence['name'])
                ), 
                range(metadata['warmup'], sequence['frames'])
            ),
            chunksize=8
        )

        # Metrics
        fvvdp = nb_fvvdp(display_name='standard_fhd').video_feeder(sequence['height'], sequence['width'], sequence['frames']-metadata['warmup'], sequence['fps'])
        tpsnr = TemporalMetricAdapter(psnr)
        trmae = TemporalMetricAdapter(rmae)
        mse_sum = 0
        psnr_sum = 0
        ssim_sum = 0
        msssim_sum = 0
        flip_sum = 0

        for test, reference in tqdm(zip(test_stream, reference_stream), leave=False, total=sequence['frames']-metadata['warmup'], unit='frame', desc=sequence['name']):
            reference = np_to_torch(reference).to('cuda')
            test = np_to_torch(test).to('cuda')

            fvvdp.feed(test, reference)
            tpsnr.feed(test, reference)
            trmae.feed(test, reference)

            mse_sum += mse(test, reference)
            psnr_sum += psnr(test, reference)
            ssim_sum += ssim(test, reference)
            msssim_sum += msssim(test, reference)
            flip_sum += flip(test, reference)
        
        metrics = {
            'fvvdp': fvvdp.compute(),
            'tpsnr': tpsnr.compute(),
            'trmae': trmae.compute(),
            'mse': mse_sum / (sequence['frames']-metadata['warmup']),
            'psnr': psnr_sum / (sequence['frames']-metadata['warmup']),
            'ssim': ssim_sum / (sequence['frames']-metadata['warmup']),
            'msssim': msssim_sum / (sequence['frames']-metadata['warmup']),
            'flip': flip_sum / (sequence['frames']-metadata['warmup']),
        }
        
        with open(os.path.join(test_path, metadata["metric_format"].format(sequence_name=sequence["name"])), 'w') as f:
            json.dump(metrics, f)

