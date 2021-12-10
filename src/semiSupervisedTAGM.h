
// semiSupervisedTAGM.h
// =============================================================================
// include guard
#ifndef SEMISUPERVISEDTAGM_H
#define SEMISUPERVISEDTAGM_H

// =============================================================================
// included dependencies
# include <RcppArmadillo.h>
# include "semiSupervisedMVN.h"
# include "genericFunctions.h"

using namespace arma ;

// =============================================================================
// virtual semiSupervisedTAGM class

//' @name semiSupervisedTAGM
//' @title Semi-Supervised Multivariate Normal mixture type
//' @description The semi-supervised Multivariate Normal mixture model.
//' @field new Constructor \itemize{
//' \item Parameter: K - the number of components to model
//' \item Parameter: labels - the initial clustering of the data
//' \item Parameter: fixed - indicator vector for which item labels are observed
//' \item Parameter: concentration - the vector for the prior concentration of
//' the Dirichlet distribution of the component weights
//' \item Parameter: X - the data to model
//' }
//' @field updateWeights Update the weights of each component based on current
//' clustering.
//' @field updateAllocation Sample a new clustering.
//' @field sampleFromPrior Sample from the priors for the multivariate normal
//' density.
//' @field calcBIC Calculate the BIC of the model.
//' @field logLikelihood Calculate the likelihood of a given data point in each
//' component. \itemize{
//' \item Parameter: point - a data point.
//' }
class semiSupervisedTAGM :
  virtual public semiSupervisedMVN
{
  
public:
  
  double global_log_det = 0.0,
    t_likelihood_const = 0.0,
    
    // T degrees of freedom
    df = 4.0, 
    
    // Hyperparameters  
    u = 2.0, v = 10.0, tau_1 = 0.0, tau_2 = 0.0,  // b = 0.0,
    
    // Outlier component weight
    non_outlier_weight = 0.0, outlier_weight = 0.0;
  
  // Vector indicating if the item is an outlier (value of 1) or not (value of 0)
  // arma::uvec outliers,
  //   non_outliers;
  
  vec 
    // The dataset mean
    global_mean, 
    
    // The outlier component likelihood
    t_ll;
  
  // The dataset covariance
  mat global_cov_inv;
  
  using semiSupervisedMVN::semiSupervisedMVN;
  
  semiSupervisedTAGM(arma::uword _K,
    arma::uvec _labels,
    arma::uvec _fixed,
    arma::mat _X
  );
  
  // Destructor
  virtual ~semiSupervisedTAGM() { };
  
  // virtual void calcBIC();
  
  // double calcTdistnLikelihood(arma::vec point) {
  double calcTdistnLikelihood(arma::uword n);
  
  void updateOutlierWeights();
  
  arma::uword sampleOutlier(arma::uword n);
  
  virtual void updateAllocation(arma::vec weights, arma::mat upweigths) override;
  
};


#endif /* SEMISUPERVISEDTAGM_H */