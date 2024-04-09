Path Tracing Denoising v1
=========================

Benchmark measuring performance denoising Monte Carlo renderings.

```{raw} html
<style>
.alg-ref-header > h2 {
    font-size: 1.5em; /* like h3 */
}
</style>
```

{.alg-ref-header}
## 2 samples-per-pixel, long sequences

| Method | Dataset | PSNR | SSIM | MS-SSIM | FVVDP | ꟻLIP↓ |
|-|-|-|-|-|-|-|
| [NPPD-small-2spp](https://github.com/balintio/nppd) | [Test8 v1](/datasets/sampleset_v1.md#test8) | 28.3952 | 0.87283 | 0.95905 | 7.35612 | 0.10702 | 

{.alg-ref-header}
## 4 samples-per-pixel, long sequences

| Method | Dataset | PSNR | SSIM | MS-SSIM | FVVDP | ꟻLIP↓ |
|-|-|-|-|-|-|-|
| [NPPD-small-4spp](https://github.com/balintio/nppd) | [Test8 v1](/datasets/sampleset_v1.md#test8) | 29.3461 | 0.88300 | 0.96597 | 7.66470 | 0.09624 |

{.alg-ref-header}
## 8 samples-per-pixel, long sequences

| Method | Dataset | PSNR | SSIM | MS-SSIM | FVVDP | ꟻLIP↓ |
|-|-|-|-|-|-|-|
| [NPPD-large-8spp](https://github.com/balintio/nppd) | [Test8 v1](/datasets/sampleset_v1.md#test8) | 30.3686 | 0.89657 | 0.97347 | 7.96943 | 0.08451 |

{.alg-ref-header}
## 32 samples-per-pixel, short sequences

| Method | Dataset | PSNR | SSIM | MS-SSIM | FVVDP | ꟻLIP↓ |
|-|-|-|-|-|-|-|
| [NPPD-large-32spp](https://github.com/balintio/nppd) | [Test32 v1](/datasets/sampleset_v1.md#test32) | 32.5832 | 0.92474 | 0.98520 | 8.51527 | 0.06371 |