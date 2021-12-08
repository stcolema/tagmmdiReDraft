// outlierComponent.h
// =============================================================================
// include guard
#ifndef OUTLIERCOMPONENT_H
#define OUTLIERCOMPONENT_H

// =============================================================================
// included dependencies
# include <RcppArmadillo.h>

using namespace arma ;

// =============================================================================
// virtual outlierComponent class

class outlierComponent {
  
private:
  
public:
  
  uword 
    // The number of items in the dataset
    N, 
    
    // Number of measurements
    P;
  
  double
    // Outlier component weights
    non_outlier_weight = 0.0, outlier_weight = 0.0,
  
    // Hyperparameters for outlier weights
    u = 2.0, v = 10.0, tau_1 = 0.0, tau_2 = 0.0;
    
  // Assume a global outlier likelihood with constant parameters
  vec outlier_likelihood;

  // The data and its transpose
  mat X, X_t;
  
  // Parametrised class
  outlierComponent(arma::mat _X);
  
  // Destructor
  virtual ~outlierComponent() { };

  // Calculate the likelihood of each item being an outlier
  void calculateAllLogLikelihoods();
  virtual double calculateItemLogLikelihood(vec x) = 0;
  
  // Update the outlier weights
  void updateWeights(uvec non_outliers, uvec outliers);
  
  // Sample if a given item is an outlier or not
  arma::uword sampleOutlier(double non_outlier_likelihood_n,
                            double outlier_likelihood_n);
  
};

#endif /* OUTLIERCOMPONENT_H */