
# include "mixture.h"

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp ;
using namespace arma ;


// Parametrised class
mixture::mixture(
  arma::uword _K,
  arma::uvec _labels,
  // arma::vec _concentration,
  arma::mat _X)
{
  
  K = _K;
  labels = _labels;
  
  // concentration = _concentration;
  X = _X;
  X_t = X.t();
  
  // Dimensions
  N = X.n_rows;
  P = X.n_cols;
  
  // Class populations
  N_k = zeros<uvec>(K);
  
  // std::cout << "\n\nN_k:\n" << N_k;
  
  // Weights
  // double x, y;
  // w = zeros<vec>(K);
  
  // Log likelihood (individual and model)
  ll = zeros<vec>(K);
  likelihood = zeros<vec>(N);
  
  // Class members
  members.set_size(N, K);
  members.zeros();
  
  // Allocation probability (not sensible in unsupervised)
  alloc.set_size(N, K);
  alloc.zeros();
  
  // not used in this, solely to enable t-adjusted mixtures in MDI.
  // 0 indicates not outlier, 1 indicates outlier within assigned cluster.
  // Outliers do not contribute to cluster parameters.
  outliers = zeros<uvec>(N);
  non_outliers = ones<uvec>(N);
  
  // Used a few times
  vec_of_ones = ones<uvec>(N);
  
  // For semi-supervised methods. A little inefficient to have here,
  // but unsupervised is not my priority.
  fixed = zeros<uvec>(N);
  fixed_ind = find(fixed == 1);
  unfixed_ind = find(fixed == 0);
};
  
void mixture::updateAllocation(arma::vec weights, arma::mat upweigths) {
  
  double u = 0.0;
  uvec uniqueK;
  vec comp_prob(K);
  
  for (auto& n : unfixed_ind) {
    // for(uword n = 0; n < N; n++){
    
    ll = itemLogLikelihood(X_t.col(n));
    
    // std::cout << "\n\nAllocation log likelihood: " << ll;
    // Update with weights
    comp_prob = ll + log(weights) + log(upweigths.col(n));
    
    // std::cout << "\nComparison probabilty:\n" << comp_prob;
    // std::cout << "\nLoglikelihood:\n" << ll;
    // std::cout << "\nWeights:\n" << log(weights);
    // std::cout << "\nCorrelation scaling:\n" << log(upweigths.col(n));
    
    likelihood(n) = accu(comp_prob);
    
    // Normalise and overflow
    comp_prob = exp(comp_prob - max(comp_prob));
    comp_prob = comp_prob / sum(comp_prob);
    
    // Save the allocation probabilities
    alloc.row(n) = comp_prob.t();
    
    // Prediction and update
    u = randu<double>( );
    
    labels(n) = sum(u > cumsum(comp_prob));
  }
  
  // The model log likelihood
  model_likelihood = accu(likelihood);
  
  // Number of occupied components (used in BIC calculation)
  uniqueK = unique(labels);
  K_occ = uniqueK.n_elem;
};
