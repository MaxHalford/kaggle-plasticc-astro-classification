## To do

- Pour graph d'un filtre sur plusieur acquisitions :
  - essayer de determiner la periode caracteristique des "oscillations des maximas"
  - si plusieur acquisition determiner la difference entre la mesure max et min
  - faire ecart entre extrema et moyenne
  - difference moyenne entres ecarts


- essayer de reconstruire un spectre
  - deduire energie
  - puissance
  - temperature
  - couleur de l'objet

- Spectral properties of each light curve
    - Amplitudes
    - Periods
    - Regularity
    - Achromacity (no difference between the different passbands)
    - Long vs. short timescales
    - Ratios of fluxes measured at different wavelengths, expressed logarithmically

https://www.kaggle.com/jeffkk/guessing-the-objects-behind-the-classes

Important: use a Bayesian mean to correct the features with low flux count


## Features

### Per passband

[x] Mean
[x] Min
[x] Max
[x] Standard deviation
[x] Skew
[x] Kurtosis
[x] `[p10, p25, p50, p75, p90]` percentiles
[x] Count
[x] Count above mean
[x] Count below mean
[x] Shapiro-Wilk test statistic
[x] Complexity [1]
[x] Mean of differences
[x] Min of differences
[x] Max of differences
[x] Standard deviation of differences
[x] Count of positive differences
[x] Count of negative differences
[x] Linear regression slope
[x] Linear regression intercept
[x] Ratio of first two Fourier amplitudes
[x] Sample entropy
[x] Time reversal asymmetry [2]

There are also derived statistics that can be obtained through simple manipulations of the above features.

[ ] `max - min` (amplitude)
[ ] `(p75 - p25) / (p90 - p10)`
[ ] `(p90 - p10) / p50`
[ ] Ratio of values above and below the mean

### Interactions

For some features we also compute ratios between each pair of passbands. This produces `6 choose 2 = 15` interaction features per base feature.


## References

1. Batista, Gustavo EAPA, et al (2014).
CID: an efficient complexity-invariant distance for time series.
Data Mining and Knowledge Difscovery 28.3 (2014): 634-669.
2. Fulcher, B.D., Jones, N.S. (2014).
Highly comparative feature-based time-series classification.
Knowledge and Data Engineering, IEEE Transactions on 26, 3026–3037.
