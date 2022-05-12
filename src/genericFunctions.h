// mixture.h
// =============================================================================
// include guard
#ifndef GENFUN_H
#define GENFUN_H

// =============================================================================
// included dependencies
# include <RcppArmadillo.h>

using namespace arma ;

// =============================================================================
// a set of functions used in a few different places

//' @title The Inverse Gamma Distribution
//' @description Random generation from the inverse Gamma distribution.
//' @param shape Shape parameter.
//' @param scale Scale parameter.
//' @return Sample from invGamma(shape, scale).
double rInvGamma(double shape, double scale);

//' @title The Half-Cauchy Distribution
//' @description Random generation from the Half-Cauchy distribution.
//' See https://en.wikipedia.org/wiki/Cauchy_distribution#Related_distributions
//' @param mu Location parameter.
//' @param scale Scale parameter.
//' @return Sample from HalfCauchy(mu, scale).
double rHalfCauchy(double mu, double scale) ;

//' @title The Half-Cauchy Distribution
//' @description Calculates the pdf of the Half-Cauchy distribution for value x.
//' See https://en.wikipedia.org/wiki/Cauchy_distribution#Related_distributions
//' @param x Value to calculate the probability density of.
//' @param mu Location parameter.
//' @param scale Scale parameter.
//' @return Sample from HalfCauchy(mu, scale).
double pHalfCauchy(double x, double mu, double scale, bool logValue = true);

//' @title The Beta Distribution
//' @description Random generation from the Beta distribution.
//' See https://en.wikipedia.org/wiki/Beta_distribution#Related_distributions.
//' Samples from a Beta distribution based using two independent gamma
//' distributions.
//' @param a Shape parameter.
//' @param b Shape parameter.
//' @return Sample from Beta(a, b).
double rBeta(double a, double b);

//' @title The Beta Distribution
//' @description Random generation from the Beta distribution.
//' See https://en.wikipedia.org/wiki/Beta_distribution#Related_distributions.
//' Samples from a Beta distribution based using two independent gamma
//' distributions.
//' @param n The number of samples to draw.
//' @param a Shape parameter.
//' @param b Shape parameter.
//' @return Sample from Beta(a, b).
arma::vec rBeta(arma::uword n, double a, double b);

//' @title Calculate sample covariance
//' @description Returns the unnormalised sample covariance. Required as
//' arma::cov() does not work for singletons.
//' @param data Data in matrix format
//' @param sample_mean Sample mean for data
//' @param n The number of samples in data
//' @param n_col The number of columns in data
//' @return One of the parameters required to calculate the posterior of the
//'  Multivariate normal with uknown mean and covariance (the unnormalised
                                                          //'  sample covariance).
arma::mat calcSampleCov(arma::mat data,
                        arma::vec sample_mean,
                        arma::uword N,
                        arma::uword P
);

//' @title Metropolis acceptance step
//' @description Given a probaility, randomly accepts by sampling from a uniform 
//' distribution.
//' @param acceptance_prob Double between 0 and 1.
//' @return Boolean indicating acceptance.
bool metropolisAcceptanceStep(double acceptance_prob);



//' @title Squared exponential function
//' @description The squared exponential function as used in a covariance kernel.
//' @param amplitude The amplitude parameter (double)
//' @param length The length parameter (double)
//' @param i Time point (unsigned integer)
//' @param j Time point (unsigned integer)
//' @return Boolean indicating acceptance.
// [[Rcpp::export]]
double squaredExponentialFunction(double amplitude, double length, arma::uword i, arma::uword j);

bool doubleApproxEqual(double x, double y, double precision = 0.000002);

#endif /* GENFUN_H */

