```{image} _static/logo-01.png
:width: 100%
:align: center
```

<div align="center">

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![PyPI - Version](https://img.shields.io/pypi/v/noisebase)](https://pypi.org/project/noisebase/) [![GitHub Repo stars](https://img.shields.io/github/stars/balintio/noisebase)](https://github.com/balintio/noisebase)

</div>

<p align="center">
<a href="news.html">News</a> &emsp;|&emsp; <a href="benchmarks/index.html">Benchmarks</a> &emsp;|&emsp; <a href="datasets/index.html">Datasets</a>
</p>

Overview
========

**Datasets and benchmarks for neural Monte Carlo denoising.**

```{raw} html
<div class="juxtapose">
    <img src="_static/ours.png" />
    <img src="_static/input.png" />
</div>
<script src="_static/juxtapose.min.js"></script>
<link rel="stylesheet" href="_static/juxtapose.css">
```

What is Monte Carlo denoising?
------------------------------

```{figure} _static/Pi_monte_carlo_all.gif
:figwidth: 30%
:align: right
Monte Carlo integration [*Kmhkmh*](https://commons.wikimedia.org/w/index.php?curid=140013480)
```
Monte Carlo methods approximate integrals by sampling random points from the function's domain, evaluating the function, and averaging the resulting samples. We mainly focus on *light transport simulation* as it's a complex and mature application, usually producing visual and intuitive results. In this case, our samples are light paths that a "photon" might take. Above on the right, you see an image rendered with 4 samples per pixel. It's quite noisy.

With a bit of napkin maths, we can estimate that rendering a relatively noise-free 4K image requires tens of billions of samples while rendering a two-hour movie requires quadrillions of samples. Astonishingly, we have data centres fit for this task. Not only do they consume electricity on par with a small town, but such computational requirements put creating 3D art outside the reach of many.

Deep neural networks have an incredible ability to reconstruct noisy data. They learn to combine the sliver of useful information contained in samples from the same object, both spatially from nearby pixels and temporally from subsequent frames. The images denoised with such neural networks (like above on the left) look absurd in comparison.

Getting started
---------------
You can start prototyping your denoiser by calling a single function:

```python
from noisebase import Noisebase

data_loader = Noisebase(
   'sampleset_v1', # Our first per-sample dataset
   {
      'framework': 'torch',
      'train': True,
      'buffers': ['diffuse', 'color', 'reference'],
      'samples': 8,
      'batch_size': 16
   }
)

# Get training, pytorch stuff...
for epoch in range(25):
   for data in data_loader:
      ...
```

And here's the kicker: with just that, our data loaders seamlessly support asynchronous and distributed loading, decompression, and augmentation of large video datasets containing anything from normal maps, diffuse maps, motion vectors, temporally changing camera intrinsics, and noisy HDR samples.

As you scale up, you'll want a little more control. Thankfully, Noisebase is fully integrated with [Hydra](https://hydra.cc/) and [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/).

Noisebase can also:
* Download training and testing data
* Runs benchmark with many metrics
* Neatly summarize everything into tables
* Help you keep track of denoising performance while keeping your implementation simple

Installation
------------
You can quickly install Noisebase from PyPI:
```bash
pip install noisebase
```
For more complicated workflows, we recommend cloning the repo instead:
```bash
git clone https://github.com/balintio/noisebase
cd noisebase
pip install -e . # Editable install
```

Check our [manual](/manual) for more details.

Citation
--------

Please cite our paper introducing Noisebase when used in academic projects:

```bibtex
@inproceedings{balint2023nppd,
    author = {Balint, Martin and Wolski, Krzysztof and Myszkowski, Karol and Seidel, Hans-Peter and Mantiuk, Rafa\l{}},
    title = {Neural Partitioning Pyramids for Denoising Monte Carlo Renderings},
    year = {2023},
    isbn = {9798400701597},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3588432.3591562},
    doi = {10.1145/3588432.3591562},
    booktitle = {ACM SIGGRAPH 2023 Conference Proceedings},
    articleno = {60},
    numpages = {11},
    keywords = {upsampling, radiance decomposition, pyramidal filtering, kernel prediction, denoising, Monte Carlo},
    location = {<conf-loc>, <city>Los Angeles</city>, <state>CA</state>, <country>USA</country>, </conf-loc>},
    series = {SIGGRAPH '23}
}
```

```{toctree}
:hidden:

self
```

```{toctree}
:caption: Explore
:hidden:

news
benchmarks/index
datasets/index
```

```{toctree}
:caption: Learn
:hidden:

guides/index
manual
loaders/index
```

```{toctree}
:caption: Contribute
:hidden:

development
formats/index
algorithms/index
reference
```