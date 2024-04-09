User manual
===========

## Loading datasets

### Simple approach

`noisebase.Noisebase` provides a convenient function for instantiating datasets:

```python
from noisebase import Noisebase

data_loader = Noisebase(
   'sampleset_v1', # Dataset name
   {
      # Data loader configuration
      'framework': 'torch',
      'train': True,
      'buffers': ['diffuse', 'color', 'reference'],
      'samples': 8,
      'batch_size': 16
   }
)
...
```

Keep in mind this still uses Hydra under the hood but without any side effects. If you are already using Hydra globally (e.g. with `hydra.main`), this won't work. Instead, you can use the Noisebase Hydra configs directly, as shown below.

### [Data loader configuration and use](/loaders/index)

### Editable installs

Noisebase datasets are *large*. You might want to keep them organized in a dedicated folder, separate from your denoiser implementations. A separate, editable install will also make adding your own datasets and managing configurations easier. You can simply clone this repository to replicate this:
```bash
git clone https://github.com/balintio/noisebase
```

Say you want to reproduce the denoising results of [NPPD](https://github.com/balintio/nppd); your setup will look somewhat like this:
```
/ # Root directory of your denoising projects
   noisebase/
      ...   # Contents of this repo
      data/ # Terabytes of downloaded training and testing data
   nppd/
      ...      # Contents of balintio/nppd
      env/     # Conda environment for NPPD
      outputs/ # Checkpoints and inference outputs for NPPD

      conf/config.yaml # The Hydra configuration file shown below
```

After setting up the environments for your denoisers you can always add Noisebase as an editable package:
```bash
(nppd/env) ~/denoisers/nppd$ pip install -e ../noisebase
```
This should work in most environments. Just make sure to install the requirements of the denoiser BEFORE the Nosiebase package. (You might end up with the wrong version of Pytorch otherwise.)

### Hydra

Hydra configs for all datasets are available under `noisebase/dataset_name`. Noisebase automatically adds them to your Hydra path, you only need to `import noisebase` in the file where your `hydra.main` is.

```python
import noisebase
import hydra

@hydra.main()
def main(cfg):
    loader = hydra.utils.instantiate(cfg['training_data'])
    ...
```

For example, NPPD configures its datasets like this:

```yaml
defaults:
  # Import the test and training datasets:
  # I like to do this with default lists
  # overriding the locations to something sensible
  - noisebase/sampleset_v1@training_data
  - noisebase/sampleset_test8_v1@test_data

  # I prefer to disable Hydra logging
  # If you like it, remove these
  - override hydra/job_logging: stdout
  - override hydra/hydra_logging: none

  # Set composition order
  - _self_

# Data loader configuration for NPPD
training_data:
  samples: 8
  batch_size: 8
  buffers:
    - normal
    - motion
    - depth
    - diffuse
    - color
    - reference
  framework: lightning

# Test data loaders usually use very similar arguments
# so you can just interpolate most things
test_data:
  buffers: ${training_data.buffers}
  samples: ${training_data.samples}

# For instantiating the Lightning NPPD model
model:
  _target_: model.Model

# Disable Hydra working directory shenanigans
# Like logging, this is personal preference
hydra:
  run:
    dir: ''
  output_subdir: null
```

## Scripts

### nb-download

Downloads a dataset.

```bash
# Example:
nb-download sampleset_v1

# Usage:
nb-download --data_path DATA_PATH dataset
```

The default data path (where the dataset is downloaded) is either a data directory within the current working directory if Noisebase was installed from PyPI or a data folder in the repository's root if Noisebase was cloned from Git. This should work seamlessly in most cases.

If you decide to use a custom data path, you must pass it as the `data_path` parameter to data loaders or the `--data_path` argument to scripts.

:::{note}
Downloading is deliberately limited to a single connection. Our data formats don't require small files, so this does not waste bandwidth. If you get faster downloads by modifying the script, that will come at others' expense. Please be patient and mindful of others.
:::

### nb-save-reference

Exports reference images from test datasets for metrics computation. Our data formats don't encode images as PNGs, so these need to be unpacked.

```bash
# Example:
nb-save-reference sampleset_test8_v1

# Usage:
nb-save-reference --data_path DATA_PATH dataset
```

### nb-compute-metrics

Computes metrics of the output of a method on a test dataset. You need to run inference and `nb-save-reference` beforehand.

```bash
# Example:
nb-compute-metrics sampleset_test8_v1 outputs/large_8_spp  

# Usage:
nb-save-reference --data_path DATA_PATH dataset outputs
```

### nb-result-table

Produces tables from metrics computed by `nb-compute-metrics`.

```bash
# Example:
nb-result-table sampleset_test8_v1 outputs/large_8_spp,outputs/small_4_spp psnr,ssim,msssim,fvvdp,flip --sep=" | "

# Usage:
nb-result-table --digits DIGITS --sep SEP --data_path DATA_PATH dataset outputs metrics
```

Output folders and metrics can be comma-separated lists. Formatting can be controlled by setting the number of floating point digits and column separator.