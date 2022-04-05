// gp.h
// =============================================================================
// include guard
#ifndef GP_H
#define GP_H

// =============================================================================
// included dependencies
# include <RcppArmadillo.h>
# include "density.h"
# include "genericFunctions.h"
// # include "kernelFactory.h"

using namespace arma ;

// =============================================================================
// gp class

//' @name gp
//' @title Gaussian process density
//' @description Class for the GP density. Currenty allows only the squared
//' exponential covariance function.
//' @field new Constructor \itemize{
//' \item Parameter: K - the number of components to model
//' \item Parameter: labels - the initial clustering of the data
//' \item Parameter: X - the data to model
//' }
//' @field sampleFromPrior Sample from the priors for the Gaussian process
//' density.
//' @field calcBIC Calculate the BIC of the model.
//' @field logLikelihood Calculate the likelihood of a given data point in each
//' component. \itemize{
//' \item Parameter: point - a data point.
//' }
class gp : virtual public density
{
  
public:
  
  // Parameters and hyperparameters
  // double kappa, nu;
  
  uvec t_inds, density_non_outliers;
  vec amplitude, length, noise, cov_log_det, zero_vec;
  umat density_members;
  mat scale, mu, cov_comb_log_det, time_difference_mat, I_p;
  cube kernel_sub_block;
  field < uvec > repeated_time_indices;
  field < vec > repeated_mean_vector, flattened_component_data;
  field < mat > covariance_matrix, inverse_covariance;
  
  // kernelFactory my_factory;
  
  // std::unique_ptr<kernel> cov_kernel_ptr;

  using density::density;
  
  gp(arma::uword _K, arma::uvec _labels, arma::mat _X);
  
  // Destructor
  virtual ~gp() { };
  
  // Calculate the empirical hyperparameters 
  arma::vec empiricalMean();
  arma::mat empiricalScaleMatrix();
  // void empiricalBayesHyperparameters();
  
  // arma::vec meanFunction();
  // arma::mat covarianceKernel();

  // Sampling from priors
  // void sampleCovPrior();
  void sampleMuPrior();
  void sampleHyperParameterPriors();
  void sampleFromPriors();
  
  // Update the common matrix manipulations to avoid recalculating N times
  // void matrixCombinations();
  
  // Sampling and calculations related to the covarianc function/matrix
  void sampleHyperParameters();
  void calculateKernelSubBlock();
  // void constructCovarianceMatrix(uword n_k, uword k);
  mat constructCovarianceMatrix(uword n_k, uword k);
  double componentCovarianceDeterminant(uword k, uword n_k);
  arma::mat calculateCovarianceKernel(arma::uvec t_inds);
  mat invertComponentCovariance(uword k, uword n_k);
  double componentCovarianceLogDeterminant(uword k, uword n_k);
  void calculateInverseCovariance(umat members, uvec non_outliers);
  
  // Sample and calulcate objects related to sampling the mean posterior function
  // vec posteriorMeanParameter(uword k, uword n_k, vec data);
  vec posteriorMeanParameter(
      uword k,
      uword n_k, 
      vec data,
      mat covariance_matrix,
      mat inverse_covariance_matrix
  );
  
  // mat posteriorCovarianceParameter(uword k, uword n_k);
  mat posteriorCovarianceParameter(
      uword k, 
      uword n_k, 
      mat covariance_matrix,
      mat inverse_covariance_matrix);
  vec sampleMeanPosterior(uword k, uword n_k, vec data);
  
  void sampleKthComponentParameters(uword k, umat members, uvec non_outliers);
  void sampleParameters(arma::umat members, arma::uvec non_outliers);
  
  // The log likelihood of a item belonging to each cluster
  arma::vec itemLogLikelihood(arma::vec item);
  
  // The log likelihood of a item belonging to a specific cluster
  double logLikelihood(arma::vec item, arma::uword k);
  
  // Initialise the kernel function - allows for different choices
  // std::unique_ptr<kernel> initialiseKernel(uword kernel_type);
};

#endif /* GP_H */