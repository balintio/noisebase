HDR Compression
===============

*Adapted from Gregory Ward Larson's [Radiance (.pic) file format](https://radsite.lbl.gov/radiance/refer/filefmts.pdf).*

*Relevant functions: [](#noisebase.compression.compress_RGBE), [](#noisebase.compression.decompress_RGBE)*

## Compression

First, we transform luminances into log space and compute the minimum `min_exposure` and maximum exposure `max_exposure`. We use 256 exposure quantisation levels evenly distributed in log space between these extremes to quantise the exposure of the brightest channel of each pixel. During quantisation, we choose the upper bound such that the quantised exposure of a given pixel is larger or equal to the exposure of each of its channels.

Second, given the quantised exposure of a given pixel, we compute an 8-bit value between {math}`[0, 1]` for each channel that, when multiplied with the pixel's exposure, gives the correct RGB luminance. In this second quantisation step, we randomly dither between the lower and upper 8-bit values to make our scheme unbiased.

## Bias

Assuming the input is a Monte Carlo rendering, we must discuss any potential bias our compression adds. 

The quantisation levels {math}`q \sim Q` depend on the input luminance {math}`i \sim I` and are therefore correlated. For example, a bright firefly will incur a much larger quantisation error than a darker pixel with less variance. This should only translate to biased behaviour if your method depends on the exact quantised floating point values. *We have yet to find such a real-world use case.* Dithering the decompressed values may help, although this isn't currently implemented.

**The compressed and then decompressed luminance {math}`C` is unbiased**, meaning it shares the same expected value as the input luminance {math}`I`. To show this, we model compression as a random choice between a lower {math}`c_{\text{lower}}` and upper {math}`c_{\text{upper}}` compressed luminance level in the function of a sampled input luminance {math}`i` and quantisation level {math}`q`. To get the expectation {math}`\mathop{\mathbb{E}}[C]`, we first need to integrate over the possible input luminances {math}`I`, then over the possible quantisation levels {math}`Q`, and then sum the lower and upper compressed values, accounting for marginal probabilities as needed:

```{math}
\mathop{\mathbb{E}}[C] = \int_I \mathrm{p}(i) \int_{Q} \mathrm{p}(q|i) \Bigl[ c_{\text{lower}}(q, i)\mathrm{p}(c_{\text{lower}}|q,i) + c_{\text{upper}}(q, i)\mathrm{p}(c_{\text{upper}}|q,i) \Bigr] \mathrm{d}q \mathrm{d}i &= \\
\int_I \mathrm{p}(i) \int_{Q} \mathrm{p}(q|i) i \mathrm{d}q \mathrm{d}i = \int_I \mathrm{p}(i) i \int_{Q} \mathrm{p}(q|i) \mathrm{d}q \mathrm{d}i &= \\
\int_I \mathrm{p}(i) i \cdot 1 \mathrm{d}i &= \mathop{\mathbb{E}}[I]
```

By design, we dither the lower and upper compressed values during compression such that:

```{math}
c_{\text{lower}}(q, i)\mathrm{p}(c_{\text{lower}}|q,i) + c_{\text{upper}}(q, i)\mathrm{p}(c_{\text{upper}}|q,i) = i
```