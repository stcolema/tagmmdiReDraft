// density.cpp
// =============================================================================
// included dependencies
# include "density.h"

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp ;
using namespace arma ;

// =============================================================================
// virtual density class



// Parametrised class
density::density(
  arma::uword _K,
  arma::uvec _labels,
  arma::mat _X)
{
  
  K = _K;
  labels = _labels;
  
  X = _X;
  X_t = X.t();
  
  // Dimensions
  N = X.n_rows;
  P = X.n_cols;
  
  // Class populations
  N_k = zeros<uvec>(K);
  
  // Log likelihood (individual and model)
  ll = zeros<vec>(K);
  likelihood = zeros<vec>(N);

};
