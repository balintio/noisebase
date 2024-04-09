Sample Test Format v1
=====================

:::{note}
This is the documentation of our on-disk storage format for per-sample test datasets. If you are looking to export Noisebase-compatible data from your renderer or write a Noisebase-compatible data loader for your framework, you're in the right place.

**If you want to use a dataset, our corresponding [data loader manual](/loaders/sample_test_v1) should be more helpful. If you are looking for datasets using the format, check out our [datasets page](/datasets/index).**
:::

Follows almost the same format as our [training data](/formats/sample_training_v1), except that we store one frame per zip file for test sequences. In practice, each array is missing the `F` dimension but is otherwise the same.

## Supported loaders

- Pytorch: `noisebase.loaders.torch.TestSampleLoader_v1`

Default format definition is in `conf/noisebase/format/test_sample_v1.yaml`.

A YAML file describing a dataset using the format needs to give the following source parameters and directories:

```yaml
src:
  samples: 8
  rendering_height: 1080
  rendering_width: 1920
  files: sampleset_v1/test8/{sequence_name}/frame{index:04d}.zip
  sequences:
    - name: bistro1
      frames: 160
    - name: bistro2
      frames: 160
    # and so on

name: "Sample Set v1 - Test8 Dataset"

# Where denoised images should be saved
# relative to the provided `save_dir`
output: test8/{sequence_name}/frame{index:04d}.png

# Where reference images should be saved
# relative to the data directory
reference: sampleset_v1/ref/test8/{sequence_name}/frame{index:04d}.png

# Where computed metrics should be saved
# relative to the provided `save_dir`
metrics: test8/{sequence_name}.json

# Optionally skip the first few frames during metric computations
# to let the temporal buffer fill up
warmup: 16
```

Check `conf/noisebase/sampleset_test8_v1.yaml` for an example.

## Radiance

| Key | Description | Dimensions | DType |
|-|-|-|-|
|`color`|RGBE encoded sample radiance|`[4, H, W, S]`|`uint8`|
|`exposure`|Minimum and maximum exposure per frame <br> for RGBE decoding|`[2]`|`float32`|
|`reference`|Clean radiance|`[3, H, W]`|`float32`|

## Sample geometry

| Key | Description | Dimensions | DType |
|-|-|-|-|
|`position`|Sample position in world-space|`[3, H, W, S]`|`float32`|
|`motion`|Change of world-space sample position <br> from last frame|`[3, H, W, S]`|`float32`|
|`normal`|Sample normal in world-space|`[3, H, W, S]`|`float16`|
|`diffuse`|Diffuse colour of the sample's material|`[3, H, W, S]`|`float16`|

## Camera data

| Key | Description | Dimensions | DType |
|-|-|-|-|
|`camera_position`|Position of the camera in world-space|`[3]`|`float32`|
|`camera_target`|A point in world-space in the center <br> of the image (where the camera is looking)|`[3]`|`float32`|
|`camera_up`|Vector in world-space that points straight <br> upwards in screen-space (what's upwards for the camera)|`[3]`|`float32`|
|`view_proj_mat`|Matrix mapping from world-space to screen-space|`[4, 4]`|`float32`|
|`proj_mat`|Matrix mapping from camera-space to screen-space|`[4, 4]`|`float32`|
|`crop_offset`|Offset of the image crop from `(0,0)` in pixel coordinates|`[2]`|`int32`|
