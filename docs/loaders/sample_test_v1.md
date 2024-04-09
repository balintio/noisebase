Sample Test Loader v1
=====================

Supported frameworks: `torch`.

## Arguments

Similar arguments to the [Sample Training Loader](/loaders/sample_training_v1), except that augmentation, shuffling, and batching are not necessary at test time.

| Key | Description | Default | Type |
|-|-|-|-|
|`samples`|Number of samples to load per pixel|8|`int`|
|`num_workers`|Number of data loading processes|4|`int`|
|`framework`|Deep learning framework to use|`torch`|`string`|

The `buffers` argument is the same as for the [Sample Training Loader](/loaders/sample_training_v1).

## Usage

The test loader makes it easy to run inference on multiple test sequences and save the inferred images. You can use these functions as in the following example:

```python
test_set = hydra.utils.instantiate(cfg['test_data'])
# or
test_set = Noisebase('sample_test8_v1')

test_set.save_dir(output_folder) # Where to save the inferred images
for sequence in test_set:
    first = True
    for i, frame in enumerate(sequence.frames):
        frame = sequence.to_torch(frame, model.device) # Convenience function to convert frames

        # Temporal initialization for NPPD in this example
        if first:
            first = False
            model.temporal = model.temporal_init(frame)

        with torch.no_grad(): # Save memory at test time
            output = model.test_step(frame)

        sequence.save(i, output) # Save inferred images asynchronously
    sequence.join() # Wait for images to finish saving in the background
```

The `frame` value here is compatible with batches produced by the [Sample Training Loader](/loaders/sample_training_v1).