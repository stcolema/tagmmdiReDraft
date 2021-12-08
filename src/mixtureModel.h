// mixtureModel.h
// =============================================================================
// include guard
#ifndef MIXTUREMODEL_H
#define MIXTUREMODEL_H

// =============================================================================
// included dependencies
# include <RcppArmadillo.h>
# include "density.h"
# include "densityFactory.h"
# include "outlierComponentFactory.h"
using namespace arma ;

// =============================================================================
// virtual mixtureModel class

class mixtureModel {
  
private:
  
public:
  
  uword 
    
    // The type of density modelled, must be 1 or 2
    mixture_type,
    
    // The type of outlier component modelled, must be 0 or 1
    outlier_type = 0,
    
    // The number of components modelled
    K, 
    
    // The number of components occupied (i.e. clusters/groups)
    K_occ, 
    
    // The dimensions of the dataset, samples and columns respectively
    N, P, 
    
    // The number of observed labels
    N_fixed = 0,
    
    n_param = 0;
  
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
  
  std::unique_ptr<density> density_ptr;
  std::unique_ptr<outlierComponent> outlierComponent_ptr;
  
  // Parametrised class
  mixtureModel(
    arma::uword _mixture_type,
    arma::uword _outlier_type,
    arma::uword _K,
    arma::uvec _labels,
    arma::uvec _fixed,
    arma::mat _X);
  
  
  // Destructor
  virtual ~mixtureModel() { };
  
  virtual void updateAllocation(arma::vec weights, arma::mat upweigths);
  
  // Initialise the density and outlier types
  virtual void initialiseMixture(arma::uword type);
  void initialiseOutlierComponent(arma::uword type);
  
  // The functions collected from the density
  virtual void sampleFromPriors();
  virtual void sampleParameters();
  virtual void calcBIC();
  virtual arma::vec itemLogLikelihood(arma::vec x);
  virtual double logLikelihood(arma::vec x, arma::uword k);
  
};

#endif /* MIXTUREMODEL_H */