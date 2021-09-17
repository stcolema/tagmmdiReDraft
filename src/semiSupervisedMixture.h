// semiSupervisedMixture.h
// =============================================================================
// include guard
#ifndef SEMISUPERVISEDMIXTURE_H
#define SEMISUPERVISEDMIXTURE_H

// =============================================================================
// included dependencies
# include <RcppArmadillo.h>
using namespace arma ;

// =============================================================================
// virtual semiSupervisedMixture class

class semiSupervisedMixture {
  
private:
  
public:
  
  uword 
  
    // The number of components modelled
    K, 
    
    // The number of components occupied (i.e. clusters/groups)
    K_occ, 
    
    // The dimensions of the dataset, samples and columns respectively
    N, P, 
  
    // The number of observed labels
    N_fixed = 0;
  double complete_likelihood = 0.0, observed_likelihood = 0.0, BIC = 0.0;
  
  uvec 
    
    // The cluster/class labels
    labels, 
    
    // The number of items in each class
    N_k, 
    
    // Vector of ones used ocasionally
    vec_of_ones,
    fixed,
    fixed_ind,
    unfixed_ind,
    
    // Outlier vectors (not really used here, declared so can be accessed in MDI class)
    outliers,
    non_outliers;
  
  vec concentration, w, ll, likelihood;
  umat members;
  mat X, X_t, alloc;
  
  // Parametrised class
  semiSupervisedMixture(
    arma::uword _K,
    arma::uvec _labels,
    arma::uvec _fixed,
    // arma::vec _concentration,
    arma::mat _X);
  
  
  // Destructor
  virtual ~semiSupervisedMixture() { };
  
  virtual void updateAllocation(arma::vec weights, arma::mat upweigths);
  
  // The virtual functions that will be defined in every subclasses
  virtual void sampleFromPriors() = 0;
  virtual void sampleParameters() = 0;
  virtual void calcBIC() = 0;
  virtual arma::vec itemLogLikelihood(arma::vec x) = 0;
  virtual double logLikelihood(arma::vec x, arma::uword k) = 0;
  
  // // Not every class needs to save matrix combinations, so this is not purely
  // // virtual
  // virtual void matrixCombinations() {};
};

#endif /* SEMISUPERVISEDMIXTURE_H */