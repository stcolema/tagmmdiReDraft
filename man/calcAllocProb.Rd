% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/calcAllocProb.R
\name{calcAllocProb}
\alias{calcAllocProb}
\title{Calculate allocation probabilities}
\usage{
calcAllocProb(mcmc_samples, view, burn = 0, method = "mean")
}
\arguments{
\item{mcmc_samples}{Output from ``callMDI``.}

\item{view}{The view for which to calculate the allocation probabilities.}

\item{burn}{The number of samples to discard.}

\item{method}{The point estimate to use. ``method = 'mean'`` or
``method = 'median'``. ``'median'`` is the default.}
}
\value{
An N x K matrix of class probabilities.
}
\description{
Calculate the empirical allocation probability for each class
based on the sampled allocation probabilities. Only makes sense in
semi-supervised views.
}
\examples{

N <- 100
X <- matrix(c(rnorm(N, 0, 1), rnorm(N, 3, 1)), ncol = 2, byrow = TRUE)
Y <- matrix(c(rnorm(N, 0, 1), rnorm(N, 3, 1)), ncol = 2, byrow = TRUE)

truth <- c(rep(1, N / 2), rep(2, N / 2))
data_modelled <- list(X, Y)

V <- length(data_modelled)

# This R is much too low for real applications
R <- 100
thin <- 5
burn <- 10

alpha <- rep(1, V)
K_max <- 10
K <- rep(K_max, V)
labels <- matrix(1, nrow = N, ncol = V)
fixed <- matrix(0, nrow = N, ncol = V)

# A random quarter of labels are known in view 1
fixed[, 1] <- sample(c(0, 1), N, replace = TRUE, prob = c(3, 1))
labels[, 1] <- generateInitialSemiSupervisedLabels(truth, fixed = fixed[, 1])
types <- rep("G", V)

mcmc_out <- callMDI(data_modelled, R, thin, types, K, labels, fixed, alpha)
calcAllocProb(mcmc_out, 1)

}
