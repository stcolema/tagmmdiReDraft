
# include "tagmMixture.h"

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp ;
using namespace arma ;


tagmMixture::tagmMixture(arma::uword _K,
  arma::uvec _labels,
  // arma::vec _concentration,
  arma::mat _X
) : mvnMixture(_K, _labels, _X),
tAdjustedMixture(_K, _labels, _X),
mixture(_K, _labels, _X){
};
  
void tagmMixture::calcBIC(){
    
  arma::uword n_param = (P + P * (P + 1) * 0.5) * (K_occ + 1);
  BIC = n_param * std::log(N) - 2 * model_likelihood;
  
};