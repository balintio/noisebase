Sample Training Loader v1
=========================

Supported frameworks: `torch`, `lightning`.

## Arguments

A Sample Training Loader takes the following arguments:

| Key | Description | Default | Type |
|-|-|-|-|
|`samples`|Number of samples to load per pixel|8|`int`|
|`flip_rotate`|Enable flip and rotation augmentations|True|`bool`|
|`batch_size`|Number of sequences to load in a batch|8|`int`|
|`num_workers`|Number of data loading processes|4|`int`|
|`shuffle`|Load sequences in random order|True|`bool`|
|`drop_last`|Ignore last batch if unevenly sized|True|`bool`|
|`val_split`|Ratio of sequences to use for validation|0.05|`float`|
|`seed`|Random seed to use for shuffling and augmentations|42|`int`|
|`framework`|Deep learning framework to use|`torch`|`string`|
|`stage`|Epoch stage: `train` or `validation`. (`torch` only)|`train`|`string`|

There's also a `buffers` argument, which determines the kinds of data the loader loads. Should be a list containing some of: `normal`, `motion`, `depth`, `w_normal`, `w_motion`, `w_position`, `diffuse`, `color`, `reference`.

:::{tip}
By default, the `torch` data loader tries to count the number of epochs passed when iterated multiple times. This ensures the augmentations and shuffling changes in each epoch. If you find this mechanism unreliable in your case, pass an `fn() -> int` function as the argument `get_epoch`. The `lightning` loader handles this automatically.
:::

## Usage

The Lightning loader is a `LightningDataModule`, while the Pytorch loader is a `torch.utils.data.DataLoader`. Each batch yielded by them is a dictionary including the following keys:

### Camera data

| Key | Description | Dimensions | DType |
|-|-|-|-|
|`camera_position`|Position of the camera in world-space|`[N, 3]`|`float32`|
|`camera_forward`|Vector in world-space that points <br> into the screen in screen-space (what's forwards for the camera)|`[N, 3]`|`float32`|
|`camera_up`|Vector in world-space that points straight <br> upwards in screen-space|`[N, 3]`|`float32`|
|`camera_left`|Vector in world-space that points straight <br> left in screen-space|`[N, 3]`|`float32`|
|`view_proj_mat`|Matrix mapping from world-space to screen-space|`[N, 4, 4]`|`float32`|
|`crop_offset`|Offset of the image crop from `(0,0)` in pixel coordinates|`[N, 2]`|`int32`|

You can find more information about `crop_offset` in the [technical description of temporal cropping](/algorithms/temporal_cropping).

The camera configuration in the previous frame is available under `prev_camera`. If this is the first frame in the sequence, this is the same as the current camera.

There's also a `frame_index` key. This has no batch dimension, as the sequences in a given batch are provided in lockstep.

### Buffer data

Excluding those not in the `buffers` argument.

| Key | Description | Dimensions | DType |
|-|-|-|-|
|`color`|Per-sample radiance|`[N, 3, H, W, S]`|`float32`|
|`reference`|Clean radiance, ground truth|`[N, 3, H, W]`|`float32`|
|`w_position`|Sample position in world-space|`[N, 3, H, W, S]`|`float32`|
|`w_motion`|Change of world-space sample position <br> from last frame|`[N, 3, H, W, S]`|`float32`|
|`w_normal`|Sample normal in world-space|`[N, 3, H, W, S]`|`float32`|
|`motion`|Change of screen-space sample position <br> (pixel) from last frame, optical flow|`[N, 3, H, W, S]`|`float32`|
|`normal`|Sample normal in screen-space <br> (relative to camera orientation)|`[N, 3, H, W, S]`|`float32`|
|`depth`|Log-disparity of the sample, <br> compressed distance from camera|`[N, 1, H, W, S]`|`float32`|
|`diffuse`|Diffuse colour of the sample's material|`[N, 3, H, W, S]`|`float32`|

### Temporal processing

Sample Training Loaders load sequences per frame in lockstep. Meaning every batch contains the frame with the same index from N sequences. For example:
```
Batch 00: [Seq 44 / Frame 00, Seq 67 / Frame 00, Seq 37 / Frame 00, Seq 23 / Frame 00]
Batch 01: [Seq 44 / Frame 01, Seq 67 / Frame 01, Seq 37 / Frame 01, Seq 23 / Frame 01]
Batch 02: [Seq 44 / Frame 02, Seq 67 / Frame 02, Seq 37 / Frame 02, Seq 23 / Frame 02]
...
Batch 63: [Seq 44 / Frame 63, Seq 67 / Frame 63, Seq 37 / Frame 63, Seq 23 / Frame 63]
######## Sequences over, new sequences are shuffled for the following batches ########
Batch 64: [Seq 81 / Frame 00, Seq 12 / Frame 00, Seq 06 / Frame 00, Seq 52 / Frame 00]
...
```

Methods often reproject data from the previous frame to the current one. Noisebase's `motion` buffer, camera data, and `backproject_pixel_centers` function are helpful here:
```python
from noisebase.torch import backproject_pixel_centers
import torch.nn.functional as F

for x in loader:
    ...
    grid = backproject_pixel_centers(
        torch.mean(x['motion'], -1),
        x['crop_offset'], 
        x['prev_camera']['crop_offset'],
        as_grid=True
    )

    reprojected = F.grid_sample(
        prev_frame,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    )
    ...
```

## Distributed training

Sample Training Loaders will work with DDP or any other strategy as long as `torch.distributed` is configured correctly—no need to mess around with `DistributedSampler`. The Lightning loader works out of the box.

## Checkpointing

Sample Training Loaders can be checkpointed and resumed mid-epoch. *It is highly recommended that you use the Lightning loader to achieve this.*

### Lightning

The `MidepochCheckpoint` function supplied by `noisebase.lightning` extends the standard `lightning.pytorch.callbacks.ModelCheckpoint`, implementing the mid-epoch checkpointing functionality of the loader. Validation is not resumable as Pytorch Lightning's monitoring of validation metrics is not resumable either.

:::{warning}
If you use this for safety checkpoints, like in the following code snippet, remember that Lightning's method of saving checkpoints is very unsafe. First, it deletes the old checkpoint, and only then does it save the new one, potentially leaving you with a corrupted file if the process dies in the meantime. This has happened to us multiple times.
:::

```python
from noisebase.lightning import MidepochCheckpoint

checkpoint_time = MidepochCheckpoint(
    dirpath=os.path.join(output_folder, 'ckpt_resume'), 
    train_time_interval=datetime.timedelta(minutes=10),
    filename='last',
    enable_version_counter=False,
    save_on_train_epoch_end=True
)
```

:::{warning}
Also, frequently restarted `ModelCheckpoint`s seem unreliable at keeping a fixed number of files. We have seen some seriously weird checkpointing behaviour from Lightning. I recommend keeping all checkpoints (`save_top_k = -1`) and cleaning manually instead.
:::

### Pytorch

First, you need to call `loader.batch_sampler.checkpoint(current_epoch, batch_idx)`, where `batch_idx` is the index of the last batch on which you haven't yet trained your method. If a checkpoint can be saved at this time, this function will return `True` and cache the loader's state. (We don't want to save checkpoints mid-sequence.)

Second, you can call `loader.state_dict()` and store the returned dict in your checkpoint. Finally, you should clear the cached state with `loader.batch_sampler.cached_state = {}`.

To load back the state, you can call `loader.load_state_dict(state_dict)` on a fresh loader before iterating the dataset.