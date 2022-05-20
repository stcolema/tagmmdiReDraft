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
# include "logLikelihoods.h"
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
  
  bool logNormPriorUsed = true;
  uword sampleHypersFrequency = 2, samplingCount = 0;
  
  double
    
    kernel_subblock_threshold = 1e-5,
    
    // amplitude_proposal_window = 0.02, 
    // length_proposal_window = 0.02, 
    // noise_proposal_window = 0.02;
    amplitude_proposal_window = 75,
    length_proposal_window = 75, 
    noise_proposal_window = 75;
  
  uvec t_inds, density_non_outliers,
    
    // Hold the count of acceptance for hyperparameters
    noise_acceptance_count,
    length_acceptance_count,
    amplitude_acceptance_count;
  
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

  double noisePriorLogDensity(double x, bool logNorm = false);
  double ampltiduePriorLogDensity(double x, bool logNorm = false);
  double lengthPriorLogDensity(double x, bool logNorm = false);
  
  double sampleAmplitudePriorDistribution(bool logNorm = false, double threshold = 1e-3);
  double sampleLengthPriorDistribution(bool logNorm = false, double threshold = 1e-3);
  double sampleNoisePriorDistribution(bool logNorm = false, double threshold = 1e-3);
  
  void sampleKthComponentHyperParameterPrior(uword k, bool logNorm = false);
  void sampleHyperParameterPriors();
  void sampleFromPriors();
  
  // Update the common matrix manipulations to avoid recalculating N times
  // void matrixCombinations();
  
  // Sampling and calculations related to the covarianc function/matrix
  void sampleHyperParameters();
  mat calculateKthComponentKernelSubBlock(double amplitude, double length);
  void calculateKernelSubBlock();
  // void constructCovarianceMatrix(uword n_k, uword k);
  mat constructCovarianceMatrix(uword n_k, uword k, mat kernel_sub_block);
  double componentCovarianceDeterminant(uword k, uword n_k);
  arma::mat calculateCovarianceKernel(arma::uvec t_inds);
  mat invertComponentCovariance(uword n_k, double noise, mat kernel_sub_block);
  // mat invertComponentCovariance(uword k, uword n_k);
  double componentCovarianceLogDeterminant(uword k, uword n_k);
  void calculateInverseCovariance(umat members, uvec non_outliers);
  
  // Sample and calulcate objects related to sampling the mean posterior function
  // vec posteriorMeanParameter(uword k, uword n_k, vec data);
  vec posteriorMeanParameter(
      vec data,
      mat covariance_matrix,
      mat inverse_covariance_matrix
  );
  
  // vec posteriorMeanParameter(
  //     uword n_k, 
  //     double noise, 
  //     mat data, 
  //     mat inv_cov_mat
  //   );
  
  mat constructCovMatrixUsingSymmetricBlockProperties(mat C, mat C_inv, uword n_k);
  
  uvec relevantIndices(uword ii, uword P);
  mat firstCovProduct(mat A, mat B, uword N);
  double blockVectorMultiplication(rowvec a, mat B, uword ii, uword jj, uword N, uword P);
  double blockVectorMultiplication(
      rowvec a, 
      mat B, 
      double lambda,
      uword ii, 
      uword jj, 
      uword N, 
      uword P
  );
  double findLambda(mat B, uword N, bool testLambdas = false);
  
  // mat posteriorCovarianceParameter(uword k, uword n_k);
  mat posteriorCovarianceParameter(
      mat covariance_matrix,
      mat inverse_covariance_matrix
  );
  void sampleMeanPosterior(uword k, uword n_k, mat data);
  
  void sampleKthComponentParameters(uword k, umat members, uvec non_outliers);
  void sampleParameters(arma::umat members, arma::uvec non_outliers);
  
  
  double proposeNewNonNegativeValue(double x, double window);
  double hyperParameterLogKernel(double hyper, vec mu_k, vec mu_tilde, mat cov_tilde, bool logNorm = false);
  void sampleLength(uword k, uword n_k, vec mu_tilde, vec component_data, mat cov_tilde);
  void sampleAmplitude(uword k, uword n_k, vec mu_tilde, vec component_data, mat cov_tilde);
  void sampleCovHypers(uword k, uword n_k, vec mu_tilde, vec component_data, mat cov_tilde);
  void sampleHyperParametersKthComponent(
      uword k, 
      uword n_k, 
      vec mu_tilde, 
      vec component_data,
      mat cov_tilde
  );
  
  double noiseLogKernel(uword n_k, double noise, vec mu, mat data);
  void sampleNoise(uword k, uword n_k, mat component_data);
  
  // The log likelihood of a item belonging to each cluster
  vec itemLogLikelihood(vec item);
  
  // The log likelihood of a item belonging to a specific cluster
  double logLikelihood(arma::vec item, arma::uword k);
  
  // Initialise the kernel function - allows for different choices
  // std::unique_ptr<kernel> initialiseKernel(uword kernel_type);
};

#endif /* GP_H */