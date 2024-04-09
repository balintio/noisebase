Sample Training Format v1
=========================

:::{note}
This is the documentation of our on-disk storage format for per-sample training datasets. If you are looking to export Noisebase-compatible data from your renderer or write a Noisebase-compatible data loader for your framework, you're in the right place.

**If you want to use a dataset, our corresponding [data loader manual](/loaders/sample_training_v1) should be more helpful. If you are looking for datasets using the format, check out our [datasets page](/datasets/index).**
:::

We store each training sequence in a separate [Zarr](https://zarr.readthedocs.io/en/stable/) `ZipStore`. We found that Zarr provides an excellent balance between compression and read speed. Each file contains several arrays, which we describe below. The first dimension of every array is the frame counter `F`, and the last three are the height `H`, width `W`, and sample count `S` of the sequence wherever applicable.

We use level 9 LZ4HC compression for each array in `FCHWS` format, chunked for every frame and every 4 samples. We found that encoding in `HWC` format would increase file sizes by about 30%.

## Supported loaders

- Pytorch: `noisebase.loaders.torch.TrainingSampleLoader_v1`
- Pytorch Lightning: `noisebase.loaders.lightning.TrainingSampleLoader_v1`

Default format definition is in `conf/noisebase/format/training_sample_v1.yaml`.

A YAML file describing a dataset using the format needs to give a name and the following source parameters:

```yaml
name: "Sample Set v1 - Training Dataset"
src:
  sequences: 1024
  files: sampleset_training_v1/scene{index:04d}.zip
  frames_per_sequence: 64
  crop: 256
  samples: 32
  rendering_height: 1080
  rendering_width: 1920
```

Check `conf/noisebase/sampleset_v1.yaml` for a complete example.

## Radiance

| Key | Description | Dimensions | DType |
|-|-|-|-|
|`color`|RGBE encoded sample radiance|`[F, 4, H, W, S]`|`uint8`|
|`exposure`|Minimum and maximum exposure per frame <br> for RGBE decoding|`[F, 2]`|`float32`|
|`reference`|Clean radiance|`[F, 3, H, W]`|`float32`|

You can find more information in the [technical description of our RGBE compression](/algorithms/rgbe).

## Sample geometry

We store world-space positions, motion, normals, and diffuse colours for every sample. Why store world-space data when most denoisers operate in screen-space? As most world-space motion is zero, we get much better compression this way. We can still calculate all kinds of screen-space data using the camera data described below.

| Key | Description | Dimensions | DType |
|-|-|-|-|
|`position`|Sample position in world-space|`[F, 3, H, W, S]`|`float32`|
|`motion`|Change of world-space sample position <br> from last frame|`[F, 3, H, W, S]`|`float32`|
|`normal`|Sample normal in world-space|`[F, 3, H, W, S]`|`float16`|
|`diffuse`|Diffuse colour of the sample's material|`[F, 3, H, W, S]`|`float16`|

## Camera data

We store camera data to encode the projection between world-space sample positions and screen-space pixel coordinates. Generally you can ignore these and use our helper functions to convert between coordinates, compute motion, depth etc. as we describe in our [getting started guide](/guides/simple_denoiser.md).

| Key | Description | Dimensions | DType |
|-|-|-|-|
|`camera_position`|Position of the camera in world-space|`[F, 3]`|`float32`|
|`camera_target`|A point in world-space in the center <br> of the image (where the camera is looking)|`[F, 3]`|`float32`|
|`camera_up`|Vector in world-space that points straight <br> upwards in screen-space (what's upwards for the camera)|`[F, 3]`|`float32`|
|`view_proj_mat`|Matrix mapping from world-space to screen-space|`[F, 4, 4]`|`float32`|
|`proj_mat`|Matrix mapping from camera-space to screen-space|`[F, 4, 4]`|`float32`|
|`crop_offset`|Offset of the image crop from `(0,0)` in pixel coordinates|`[F, 2]`|`int32`|

You can find more information about `crop_offset` in the [technical description of temporal cropping](/algorithms/temporal_cropping).
