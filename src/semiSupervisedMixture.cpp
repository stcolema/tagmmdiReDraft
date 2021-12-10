// semiSupervisedMixture.cpp
// =============================================================================
// included dependencies
# include "semiSupervisedMixture.h"

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp ;
using namespace arma ;

// =============================================================================
// virtual semiSupervisedMixture class

// Parametrised class
semiSupervisedMixture::semiSupervisedMixture(
  arma::uword _K,
  arma::uvec _labels,
  arma::uvec _fixed,
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
  
  // Pass the indicator vector for being fixed to the ``fixed`` object.
  fixed = _fixed;
  
  N_fixed = accu(fixed);
  
  fixed_ind = find(fixed == 1);
  unfixed_ind = find(fixed == 0);
  
  // std::cout << "\n\nNumber fixed: " << N_fixed;
  
  // Set the known label allocations to 1
  for(uword n = 0; n < N; n++) {
    if(fixed(n) == 1) {
      alloc(n, labels(n)) = 1.0;
    }
  }
  
  // Used a few times
  vec_of_ones = ones<uvec>(N);
  
  // Outlier vectors; not used really here
  non_outliers = ones<uvec>(N);
  outliers = zeros<uvec>(N);
};

void semiSupervisedMixture::updateAllocation(arma::vec weights,
  arma::mat upweigths
) {
  
  double u = 0.0;
  uvec uniqueK;
  vec comp_prob(K);
  
  complete_likelihood = 0.0;
  observed_likelihood = 0.0;
  
  // for (auto& n : unfixed_ind) {
  for (uword n = 0; n < N; n++) {
      
    // The mixture-specific log likelihood for each observation in each class
    ll = itemLogLikelihood(X_t.col(n));
    
    // Update with weights
    comp_prob = ll + log(weights) + log(upweigths.col(n));
    
    // Record the likelihood - this is used to calculate the observed likelihood
    // likelihood(n) = accu(comp_prob);
    observed_likelihood += accu(comp_prob);
    

    if(fixed(n) == 0) {
      // Handle overflow problems and then normalise to convert to probabilities
      comp_prob = exp(comp_prob - max(comp_prob));
      comp_prob = comp_prob / sum(comp_prob);
      
      // Save the allocation probabilities
      alloc.row(n) = comp_prob.t();
      
      // Prediction and update
      u = randu<double>( );
      
      labels(n) = sum(u > cumsum(comp_prob));
    }
    
    // Update the complete likelihood based on the new labelling
    complete_likelihood += ll(labels(n));
    
  }
  
  // Number of occupied components (used in BIC calculation)
  uniqueK = unique(labels);
  K_occ = uniqueK.n_elem;
};

void semiSupervisedMixture::initialiseMixture(
  arma::vec weights,
  arma::mat upweigths
) {
  
  uvec uniqueK;
  vec comp_prob(K);
  
  // Initialise the parameters by sampling from the prior distributions
  sampleFromPriors();
  
  complete_likelihood = 0.0;
  observed_likelihood = 0.0;
  
  // for (auto& n : unfixed_ind) {
  for (uword n = 0; n < N; n++) {
    
    // The mixture-specific log likelihood for each observation in each class
    ll = itemLogLikelihood(X_t.col(n));
    
    // Update with weights
    comp_prob = ll + log(weights) + log(upweigths.col(n));
    
    // Record the likelihood - this is used to calculate the observed likelihood
    // likelihood(n) = accu(comp_prob);
    observed_likelihood += accu(comp_prob);
    
    if(fixed(n) == 0) {
      // Handle overflow problems and then normalise to convert to probabilities
      comp_prob = exp(comp_prob - max(comp_prob));
      comp_prob = comp_prob / sum(comp_prob);
      
      // Save the allocation probabilities
      alloc.row(n) = comp_prob.t();
    }
    
    // Update the complete likelihood based on the new labelling
    complete_likelihood += ll(labels(n));
    
  }
  
  // Number of occupied components (used in BIC calculation)
  uniqueK = unique(labels);
  K_occ = uniqueK.n_elem;
  
}
