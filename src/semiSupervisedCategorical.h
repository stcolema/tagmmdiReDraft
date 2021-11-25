// semiSupervisedCategorical.h
// =============================================================================
// include guard
#ifndef SEMISUPERVISEDCATEGORICAL_H
#define SEMISUPERVISEDCATEGORICAL_H

// =============================================================================
// included dependencies
# include <RcppArmadillo.h>
# include "semiSupervisedMixture.h"
# include "genericFunctions.h"

using namespace arma ;

// =============================================================================
// virtual semiSupervisedCategorical class

//' @name semiSupervisedCategorical
//' @title Semi-Supervised Categorical mixture type
//' @description The semi-supervised Categorical mixture model.
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
class semiSupervisedCategorical :
  virtual public semiSupervisedMixture
{
  
public:
  
  // Each component has a weight, a mean vector and a symmetric covariance matrix.
  uword n_param = 0;
  
  // The number of categories in each measurement
  uvec n_cat;
  
  // This will hold the data converted to a matrix of integers
  umat Y;
  
  // The prior probability of being in a given category in each measurement
  field<vec> cat_prior_probability;
  
  // The probability of each class within category; it will be a N_cat x K x P
  // array
  arma::field<arma::mat> class_probabilities;
  
  using semiSupervisedMixture::semiSupervisedMixture;
  
  semiSupervisedCategorical(arma::uword _K,
                    arma::uvec _labels,
                    arma::uvec _fixed,
                    arma::mat _X
  );
  
  // Destructor
  virtual ~semiSupervisedCategorical() { };
  
  // Sampling from priors
  virtual void sampleFromPriors();
  
  // The log likelihood of a item belonging to each cluster
  virtual arma::vec itemLogLikelihood(arma::vec item);
  
  // The log likelihood of a item belonging to a specific cluster
  virtual double logLikelihood(arma::vec item, arma::uword k);
  
  virtual void calcBIC();
  
  virtual void sampleParameters();
  
  double posteriorPredictive(arma::vec x, arma::uvec indices);
  
};


#endif /* SEMISUPERVISEDCATEGORICAL_H */