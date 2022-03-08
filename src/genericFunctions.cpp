
# include "genericFunctions.h"

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp ;
using namespace arma ;

//' @title The Inverse Gamma Distribution
//' @description Random generation from the inverse Gamma distribution.
//' @param shape Shape parameter.
//' @param scale Scale parameter.
//' @return Sample from invGamma(shape, scale).
double rInvGamma(double shape, double scale) {
  double x = arma::randg( distr_param(shape, scale) );
  return (1 / x);
}

//' @title The Half-Cauchy Distribution
//' @description Random generation from the Half-Cauchy distribution.
//' See https://en.wikipedia.org/wiki/Cauchy_distribution#Related_distributions
//' @param mu Location parameter.
//' @param scale Scale parameter.
//' @return Sample from HalfCauchy(mu, scale).
double rHalfCauchy(double mu, double scale) {
  double x = 0.0, y = 0.0;
  x = arma::randn();
  if(x < 0.0) {
    x = 0.0;
  }
  y = rInvGamma(0.5, 0.5 * std::pow(scale, 2.0));
  return mu + x * std::sqrt(y);
}

//' @title The Half-Cauchy Distribution
//' @description Calculates the pdf of the Half-Cauchy distribution for value x.
//' See https://en.wikipedia.org/wiki/Cauchy_distribution#Related_distributions
//' @param x Value to calculate the probability density of.
//' @param mu Location parameter.
//' @param scale Scale parameter.
//' @return Sample from HalfCauchy(mu, scale).
double pHalfCauchy(double x, double mu, double scale) {
  double denom = 0.0;

  if(x < mu) {
    return 0;
  }
  denom = 1 + std::pow((x - mu) / scale, 2.0);
  return 2 / (M_PI * scale * denom);
}

//' @title The Beta Distribution
//' @description Random generation from the Beta distribution.
//' See https://en.wikipedia.org/wiki/Beta_distribution#Related_distributions.
//' Samples from a Beta distribution based using two independent gamma
//' distributions.
//' @param a Shape parameter.
//' @param b Shape parameter.
//' @return Sample from Beta(a, b).
double rBeta(double a, double b) { // double theta = 1.0) {
  double X = arma::randg( arma::distr_param(a, 1.0) );
  double Y = arma::randg( arma::distr_param(b, 1.0) );
  double beta = X / (double)(X + Y);
  return(beta);
}

//' @title The Beta Distribution
//' @description Random generation from the Beta distribution.
//' See https://en.wikipedia.org/wiki/Beta_distribution#Related_distributions.
//' Samples from a Beta distribution based using two independent gamma
//' distributions.
//' @param n The number of samples to draw.
//' @param a Shape parameter.
//' @param b Shape parameter.
//' @return Sample from Beta(a, b).
arma::vec rBeta(arma::uword n, double a, double b) {
  arma::vec X = arma::randg(n, arma::distr_param(a, 1.0) );
  arma::vec Y = arma::randg(n, arma::distr_param(b, 1.0) );
  arma::vec beta = X / (X + Y);
  return(beta);
}

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
) {

  mat sample_covariance = zeros<mat>(P, P);

  // If n > 0 (as this would crash for empty clusters), and for n = 1 the
  // sample covariance is 0
  if(N > 1){
    data.each_row() -= sample_mean.t();
    sample_covariance = data.t() * data;
  }
  return sample_covariance;
}
