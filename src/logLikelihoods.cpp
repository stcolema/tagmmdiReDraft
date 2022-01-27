// logLikelihoods.cpp
// =============================================================================
// included dependencies
# include <RcppArmadillo.h>
# include "logLikelihoods.h"

// [[Rcpp::depends(RcppArmadillo)]]

using namespace std ;

// =============================================================================

double gammaLogLikelihood(double x, double shape, double rate){
  return shape * log(rate) - lgamma(shape) + (shape - 1) * log(x) - rate * x;
};

double invGammaLogLikelihood(double x, double shape, double scale) {
  return shape * log(scale) - lgamma(shape) + (-shape - 1) * log(x) - scale / x;
};

double wishartLogLikelihood(arma::mat X, arma::mat V, double n, arma::uword P){
  return 0.5*((n - P - 1) * arma::log_det(X).real() 
                - trace(arma::inv_sympd(V) * X) 
                - n * arma::log_det(V).real()
  );
}

double invWishartLogLikelihood(arma::mat X, arma::mat Psi, double nu, arma::uword P) {
  return -0.5*(nu * arma::log_det(Psi).real()
                 + (nu + P + 1) * arma::log_det(X).real()
                 + arma::trace( Psi * arma::inv_sympd(X) ) 
  );
}

double mvtLogLikelihood(arma::vec x, arma::vec mu, arma::mat Sigma, double nu) {
    
    arma::uword P = x.n_rows;
    double exponent = 0.0, ll = 0.0;
    
    exponent = arma::as_scalar(
      (x - mu).t()
      * Sigma
      * (x - mu)
    );

    ll = lgamma(0.5 * (nu + P)) 
      - lgamma(0.5 * nu) 
      - 0.5 * P * log(nu * M_PI)
      - 0.5 * arma::log_det(Sigma).real()
      - ((nu + (double) P) / 2.0) * std::log(1.0 + (1.0 / nu) * exponent);
    
    return ll;
}
