// mixtureModel.cpp
// =============================================================================
  // included dependencies
# include <RcppArmadillo.h>
# include "mixtureModel.h"

using namespace arma ;

// =============================================================================
// mixtureModel class

// Parametrised class
mixtureModel::mixtureModel(
  arma::uword _mixture_type,
  arma::uword _outlier_type,
  arma::uword _K,
  arma::uvec _labels,
  arma::uvec _fixed,
  arma::mat _X) {
  
  mixture_type = _mixture_type;
  outlier_type = _outlier_type;
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
  
  // Initialise the density
  initialiseMixture(mixture_type);
  n_param = density_ptr->n_param;
};
  
void mixtureModel::updateAllocation(
  arma::vec weights, 
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
  
void mixtureModel::initialiseMixture(arma::uword type) {
  
  densityFactory my_factory;
  
  // Convert the unsigned integer into a mixture type object
  densityFactory::densityType val = static_cast<densityFactory::densityType>(type);
  
  // Create a smart pointer to the correct type of model
  density_ptr = my_factory.createMixture(val, K, labels, X);
};
  
// Sample from the prior distribution of the density
void mixtureModel::sampleFromPriors() {
  density_ptr->sampleFromPriors();
};

void mixtureModel::sampleParameters() {
  density_ptr->sampleParameters(members, non_outliers);
};

// BIC currently ignores outlier parametes
void mixtureModel::calcBIC() {
  BIC = 2 * complete_likelihood - (n_param + 1) * K_occ * std::log(N);
}
arma::vec mixtureModel::itemLogLikelihood(arma::vec x) {
  return density_ptr->itemLogLikelihood(x);
};

double mixtureModel::logLikelihood(arma::vec x, arma::uword k) {
  return density_ptr->logLikelihood(x, k);
};

void mixtureModel::initialiseOutlierComponent(uword type) {
  outlierComponentFactory my_factory;

  // Convert the unsigned integer into a mixture type object
  outlierComponentFactory::outlierType val = static_cast<outlierComponentFactory::outlierType>(type);

  // Create a smart pointer to the correct type of model
  outlierComponent_ptr = my_factory.createOutlierComponent(val, X);
};
