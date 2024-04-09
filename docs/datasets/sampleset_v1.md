SampleSet v1
============

*From [Neural Partitioning Pyramids for Denoising Monte Carlo Renderings](https://dl.acm.org/doi/10.1145/3588432.3591562)*

Inspired by [Hypersim](https://github.com/apple/ml-hypersim), we leverage [Evermotion](https://www.turbosquid.com/Search/Artists/evermotion)'s Archinteriors and Archexteriors collections to build our production-quality training dataset that exceeds the quality and diversity of datasets used in previous works.

We optimise 7 exterior and 8 interior scenes for the [Falcor](https://github.com/NVIDIAGameWorks/Falcor) renderer and manually add camera trajectories. We generate 1024 64-frame-long training sequences. We randomly pick a scene, a one-second camera trajectory segment, and an environment map for each sequence. We further perturb the camera trajectory and add randomly moving sphere lights and objects nearby the trajectory. We pick objects from the [Amazon Berkeley Objects Dataset](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) containing 7941 high-quality 3D models with physically based materials and environment maps from the [Poly Haven HDRI Dataset](https://polyhaven.com/) containing 388 4K environment maps. We randomise our generated sphere lights' colour, size, and intensity. We extract 256 by 256 motion compensated patches cropped from a 1080 by 1920 virtual camera frames. The motion is compensated by adjusting the crop offset according to the average optical flow in the cropped region. This allows our training patches to capture more temporal information.

We render the supervision reference images for our training dataset at 6144 samples per pixel. Due to our scenes' complexity, some noise and fireflies are present in these images. Mitigating the noise by increasing our reference sample count by several orders of magnitude would make the generation of our dataset impractical. Therefore, we use [OIDN](https://github.com/OpenImageDenoise/oidn/) to denoise three uncorrelated 2048 spp estimates and take their median as our training reference. We do not apply denoising to our test set.

`sampleset_v1` — training data
------------------------------

SampleSet v1 contains 1024 64 frame long sequences, 256 by 256 in resolution, containing 32 samples per pixel.

```{image} ../_static/sampleset_v1.png
:width: 70%
:align: center
```

Loaded with [Sample Training Loader v1](/loaders/sample_training_v1). <br>
Stored in [Sample Training Format v1](/formats/sample_training_v1).

{#test8}
`sampleset_test8_v1` — long test sequences
------------------------------------------

Up to 8 samples-per-pixel.

Loaded with [Sample Test Loader v1](/loaders/sample_test_v1). <br>
Stored in [Sample Test Format v1](/formats/sample_test_v1).

{#test32}
`sampleset_test32_v1` — short test sequences
--------------------------------------------

Up to 32 samples-per-pixel.

Loaded with [Sample Test Loader v1](/loaders/sample_test_v1). <br>
Stored in [Sample Test Format v1](/formats/sample_test_v1).