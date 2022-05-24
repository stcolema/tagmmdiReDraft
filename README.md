# tagmmdiReDraft (working title)

R wrapper and helper functions for C++ implementation of Multiple Dataset Integration (Kirk et al., 2012), an extrapolation of Bayesian mixture models to the multi-view/multiple dataset/integrative clustering setting, and Bayesian mixture models. Available densities are the Multivariate Gaussian (MVN), Gaussian with a diagonal covariance matrix (G), Gaussian Process with a squared exponential kernal (GP) and categorical (C). The continuous densities can be augmented with a Multivariate t-distribution (MVT) to model outliers.

The models can be used for semi-supervised classification by passing some labels as observed or fixed.

## C++17 and Windows

I used C++17 for parallelising some of the for loops. To compile an R package built using C++17 on Windows you need RTools 4.0 (or later) and to follow the instructions at https://rviews.rstudio.com/2020/07/08/r-package-integration-with-modern-reusable-c-code-using-rcpp/.
