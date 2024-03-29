% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/generateSimulationDataset.R
\name{generateSimulationDataset}
\alias{generateSimulationDataset}
\title{Generate simulation dataset}
\usage{
generateSimulationDataset(
  K,
  N,
  P,
  delta_mu = 1,
  cluster_sd = 1,
  pi = rep(1/K, K),
  P_n = 0
)
}
\arguments{
\item{K}{The number of components to sample from.}

\item{N}{The number of samples to draw.}

\item{P}{The number of relevant (i.e. signal-bearing) features.}

\item{delta_mu}{The difference between the means defining each component
within each feature (defaults to 1).}

\item{cluster_sd}{The standerd deviation of the Gaussian distributions.}

\item{pi}{The K-vector of the populations proportions across each component.}

\item{P_n}{The number of irrelevant features (defaults to 0).}
}
\value{
A list of `data` (a data.frame of the generated data) and
`cluster_IDs` (a vector of the cluster membership of each item).

Named list containing ``data``, a matrix of the generated Gaussian
data and ``cluster_IDs``, the true generating structure.
}
\description{
Generates a dataset based upon a mixture of $K$ Gaussian
distributions with $P$ independent, relevant features and $P_n$ irrelevant
features. Irrelevant features contain no signal for underlying structure and 
all measurements for an irrelevant feature are drawn from a common standard 
Gaussian distribution.
}
\examples{
K <- 4
N <- 100
P <- 4
generateSimulationDataset(K, N, P)
}
