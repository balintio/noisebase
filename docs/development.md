Development
===========

:::{tip}
If anything is unclear, ask away on the project's [Issues page](https://github.com/balintio/noisebase/issues). I'll expand this page with answers as needed.
:::

## Datasets, Formats, Loaders

These make up the core components of Nosiebase; however, their differences can be quite confusing.

* **Loaders**: Actual Python classes doing the data loading. They are written to load datasets that conform to format specifications.

* **Formats**: Specification of the on-disk data format. Describes what a dataset can (and should) provide and what a loader must be able to load.

* **Datasets**: The actual data. This includes configuration files and data files as required by the format the dataset is using.

The composition of dataset configurations, format definitions and loader parameters is handled through Hydra.

* Each dataset must be derived from a format definition. Format definitions are found in the `noisebase/conf/noisebase/format` directory while dataset configurations are found in the `noisebase/conf/noisebase` directory.

* A format definition must contain all the arguments for the loader, the configuration options the dataset should override, and the references to instantiate the loaders written for loading the format.

## Building documentation

Noisebase's documentation is generated using Sphinx:
```bash
cd docs
conda env create -p ./env -f environment.yaml
conda activate ./env
make html
```