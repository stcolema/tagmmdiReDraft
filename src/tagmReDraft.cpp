# include <RcppArmadillo.h>
# include <math.h>
# include <string>
# include <iostream>

# include "logLikelihoods.h"
# include "genericFunctions.h"

// # include "mixture.h"
// # include "tAdjustedMixture.h"
// # include "mvnMixture.h"
// # include "tagmMixture.h"

# include "semiSupervisedMixture.h"
# include "semiSupervisedMVN.h"
# include "semiSupervisedTAGM.h"

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp ;
using namespace arma ;

// PLAN
// mdiClass
//  phis /correlation
//  gammas / weights
//  mixture 1
//  mixture 2
//  label-swapping
//
// mixture class
//  allocation takes weights and phis as argument
// MVN mixture
// TAGM mixture
// Categorical mixture
// Semi-supervised
//
// sampler function - exported to R

// //' @title The Beta Distribution
// //' @description Random generation from the Beta distribution.
// //' See https://en.wikipedia.org/wiki/Beta_distribution#Related_distributions.
// //' Samples from a Beta distribution based using two independent gamma
// //' distributions.
// //' @param a Shape parameter.
// //' @param b Shape parameter.
// //' @return Sample from Beta(a, b).
// double rBeta(double a, double b) { // double theta = 1.0) {
//   double X = arma::randg( arma::distr_param(a, 1.0) );
//   double Y = arma::randg( arma::distr_param(b, 1.0) );
//   double beta = X / (double)(X + Y);
//   return(beta);
// }
// 
// //' @title The Beta Distribution
// //' @description Random generation from the Beta distribution.
// //' See https://en.wikipedia.org/wiki/Beta_distribution#Related_distributions.
// //' Samples from a Beta distribution based using two independent gamma
// //' distributions.
// //' @param n The number of samples to draw.
// //' @param a Shape parameter.
// //' @param b Shape parameter.
// //' @return Sample from Beta(a, b).
// arma::vec rBeta(arma::uword n, double a, double b) {
//   arma::vec X = arma::randg(n, arma::distr_param(a, 1.0) );
//   arma::vec Y = arma::randg(n, arma::distr_param(b, 1.0) );
//   arma::vec beta = X / (X + Y);
//   return(beta);
// }
// 
// //' @title Calculate sample covariance
// //' @description Returns the unnormalised sample covariance. Required as
// //' arma::cov() does not work for singletons.
// //' @param data Data in matrix format
// //' @param sample_mean Sample mean for data
// //' @param n The number of samples in data
// //' @param n_col The number of columns in data
// //' @return One of the parameters required to calculate the posterior of the
// //'  Multivariate normal with uknown mean and covariance (the unnormalised
// //'  sample covariance).
// arma::mat calcSampleCov(arma::mat data,
//                         arma::vec sample_mean,
//                         arma::uword N,
//                         arma::uword P
// ) {
// 
//   arma::mat sample_covariance = arma::zeros<arma::mat>(P, P);
// 
//   // If n > 0 (as this would crash for empty clusters), and for n = 1 the
//   // sample covariance is 0
//   if(N > 1){
//     data.each_row() -= sample_mean.t();
//     sample_covariance = data.t() * data;
//   }
//   return sample_covariance;
// }

// 
// class mixture {
// 
// private:
// 
// public:
// 
//   arma::uword K, N, P, K_occ;
//   double model_likelihood = 0.0, BIC = 0.0;
//   uvec labels, 
//     N_k, 
//     batch_vec, 
//     N_b, 
//     outliers, 
//     non_outliers, 
//     vec_of_ones,
//     fixed,
//     fixed_ind,
//     unfixed_ind;
//   
//   arma::vec concentration, w, ll, likelihood;
//   arma::umat members;
//   arma::mat X, X_t, alloc;
// 
//   // Parametrised class
//   mixture(
//     arma::uword _K,
//     arma::uvec _labels,
//     // arma::vec _concentration,
//     arma::mat _X)
//   {
// 
//     K = _K;
//     labels = _labels;
// 
//     // concentration = _concentration;
//     X = _X;
//     X_t = X.t();
// 
//     // Dimensions
//     N = X.n_rows;
//     P = X.n_cols;
// 
//     // Class populations
//     N_k = arma::zeros<arma::uvec>(K);
// 
//     // std::cout << "\n\nN_k:\n" << N_k;
//     
//     // Weights
//     // double x, y;
//     // w = arma::zeros<arma::vec>(K);
// 
//     // Log likelihood (individual and model)
//     ll = arma::zeros<arma::vec>(K);
//     likelihood = arma::zeros<arma::vec>(N);
// 
//     // Class members
//     members.set_size(N, K);
//     members.zeros();
//     
//     // Allocation probability (not sensible in unsupervised)
//     alloc.set_size(N, K);
//     alloc.zeros();
// 
//     // not used in this, solely to enable t-adjusted mixtures in MDI.
//     // 0 indicates not outlier, 1 indicates outlier within assigned cluster.
//     // Outliers do not contribute to cluster parameters.
//     outliers = arma::zeros<uvec>(N);
//     non_outliers = arma::ones<uvec>(N);
//     
//     // Used a few times
//     vec_of_ones = arma::ones<uvec>(N);
//   
//     // For semi-supervised methods. A little inefficient to have here,
//     // but unsupervised is not my priority.
//     fixed = arma::zeros<uvec>(N);
//     fixed_ind = find(fixed == 1);
//     unfixed_ind = find(fixed == 0);
//   };
//   
// 
//   // Destructor
//   virtual ~mixture() { };
// 
//   // Virtual functions are those that should actual point to the sub-class
//   // version of the function.
//   // Print the sampler type.
//   virtual void printType() {
//     std::cout << "\nType: NULL.\n";
//   };
// 
//   // // Functions required of all mixture models
//   // virtual void updateWeights(){
//   //
//   //   double a = 0.0;
//   //
//   //   for (arma::uword k = 0; k < K; k++) {
//   //
//   //     // Find how many labels have the value
//   //     members.col(k) = labels == k;
//   //     N_k(k) = arma::sum(members.col(k));
//   //
//   //     // Update weights by sampling from a Gamma distribution
//   //     a  = concentration(k) + N_k(k);
//   //     w(k) = arma::randg( arma::distr_param(a, 1.0) );
//   //   }
//   //
//   //   // Convert the cluster weights (previously gamma distributed) to Beta
//   //   // distributed by normalising
//   //   w = w / arma::accu(w);
//   //
//   // };
// 
//   virtual void updateAllocation(arma::vec weights, arma::mat upweigths) {
// 
//     double u = 0.0;
//     arma::uvec uniqueK;
//     arma::vec comp_prob(K);
// 
//     for (auto& n : unfixed_ind) {
//     // for(arma::uword n = 0; n < N; n++){
// 
//       ll = itemLogLikelihood(X_t.col(n));
// 
//       // std::cout << "\n\nAllocation log likelihood: " << ll;
//       // Update with weights
//       comp_prob = ll + log(weights) + log(upweigths.col(n));
// 
//       // std::cout << "\nComparison probabilty:\n" << comp_prob;
//       // std::cout << "\nLoglikelihood:\n" << ll;
//       // std::cout << "\nWeights:\n" << log(weights);
//       // std::cout << "\nCorrelation scaling:\n" << log(upweigths.col(n));
// 
//       likelihood(n) = arma::accu(comp_prob);
// 
//       // Normalise and overflow
//       comp_prob = exp(comp_prob - max(comp_prob));
//       comp_prob = comp_prob / sum(comp_prob);
// 
//       // Save the allocation probabilities
//       alloc.row(n) = comp_prob.t();
//       
//       // Prediction and update
//       u = arma::randu<double>( );
// 
//       labels(n) = sum(u > cumsum(comp_prob));
//     }
// 
//     // The model log likelihood
//     model_likelihood = arma::accu(likelihood);
// 
//     // Number of occupied components (used in BIC calculation)
//     uniqueK = arma::unique(labels);
//     K_occ = uniqueK.n_elem;
//   };
// 
//   // The virtual functions that will be defined in every subclasses
//   virtual void sampleFromPriors() = 0;
//   virtual void sampleParameters() = 0;
//   virtual void calcBIC() = 0;
//   virtual arma::vec itemLogLikelihood(arma::vec x) = 0;
// 
//   // Not every class needs to save matrix combinations, so this is not purely // [[Rcpp::export]]
//   // virtual
//   virtual void matrixCombinations() {};
// };
// 
// 
// 
// //' @name mvnMixture
// //' @title Multivariate Normal mixture type
// //' @description The sampler for the Multivariate Normal mixture model for batch effects.
// //' @field new Constructor \itemize{
// //' \item Parameter: K - the number of components to model
// //' \item Parameter: labels - the initial clustering of the data
// //' \item Parameter: concentration - the vector for the prior concentration of
// //' the Dirichlet distribution of the component weights
// //' \item Parameter: X - the data to model
// //' }
// //' @field printType Print the sampler type called.
// //' @field updateWeights Update the weights of each component based on current
// //' clustering.
// //' @field updateAllocation Sample a new clustering.
// //' @field sampleFromPrior Sample from the priors for the multivariate normal
// //' density.
// //' @field calcBIC Calculate the BIC of the model.
// //' @field logLikelihood Calculate the likelihood of a given data point in each
// //' component. \itemize{
// //' \item Parameter: point - a data point.
// //' }
// class mvnMixture: virtual public mixture {
// 
// public:
// 
//   // Each component has a weight, a mean vector and a symmetric covariance matrix.
//   arma::uword n_param_cluster = 0;
// 
//   double kappa, nu;
//   arma::vec xi, cov_log_det;
//   arma::mat scale, mu, cov_comb_log_det;
//   arma::cube cov, cov_inv;
// 
//   using mixture::mixture;
// 
//   mvnMixture(
//     arma::uword _K,
//     arma::uvec _labels,
//     // arma::vec _concentration,
//     arma::mat _X
//   ) : mixture(_K,
//   _labels,
//   // _concentration,
//   _X) {
// 
//     n_param_cluster = 1 + P + P * (P + 1) * 0.5;
// 
//     // Default values for hyperparameters
//     // Cluster hyperparameters for the Normal-inverse Wishart
//     // Prior shrinkage
//     kappa = 0.01;
//     // Degrees of freedom
//     nu = P + 2;
// 
//     // Mean
//     arma::mat mean_mat = arma::mean(_X, 0).t();
//     xi = mean_mat.col(0);
// 
//     // Empirical Bayes for a diagonal covariance matrix
//     arma::mat scale_param = _X.each_row() - xi.t();
//     arma::vec diag_entries(P);
//     // double scale_entry = arma::accu(scale_param % scale_param, 0) / (N * std::pow(K, 1.0 / (double) P));
// 
//     arma::mat global_cov = arma::cov(X);
//     double scale_entry = (arma::accu(global_cov.diag()) / P) / std::pow(K, 2.0 / (double) P);
// 
//     diag_entries.fill(scale_entry);
//     scale = arma::diagmat( diag_entries );
// 
//     // Set the size of the objects to hold the component specific parameters
//     mu.set_size(P, K);
//     mu.zeros();
// 
//     cov.set_size(P, P, K);
//     cov.zeros();
// 
//     // These will hold vertain matrix operations to avoid computational burden
//     // The log determinant of each cluster covariance
//     cov_log_det = arma::zeros<arma::vec>(K);
// 
//     // Inverse of the cluster covariance
//     cov_inv.set_size(P, P, K);
//     cov_inv.zeros();
//   };
// 
// 
//   // Destructor
//   virtual ~mvnMixture() { };
// 
//   // Print the sampler type.
//   virtual void printType() {
//     std::cout << "\nType: MVN.\n";
//   };
// 
//   virtual void sampleCovPrior() {
//     for(arma::uword k = 0; k < K; k++){
//       cov.slice(k) = arma::iwishrnd(scale, nu);
//       cov_inv.slice(k) = arma::inv_sympd(cov.slice(k));
//       cov_log_det(k) = arma::log_det(cov.slice(k)).real();
//     }
//   };
// 
//   virtual void sampleMuPrior() {
//     for(arma::uword k = 0; k < K; k++){
//       mu.col(k) = arma::mvnrnd(xi, (1.0/kappa) * cov.slice(k), 1);
//     }
//   }
// 
//   virtual void sampleFromPriors() {
//     sampleCovPrior();
//     sampleMuPrior();
//   };
// 
//   // Update the common matrix manipulations to avoid recalculating N times
//   virtual void matrixCombinations() {
//     for(arma::uword k = 0; k < K; k++) {
//       cov_inv.slice(k) = arma::inv_sympd(cov.slice(k));
//       cov_log_det(k) = arma::log_det(cov.slice(k)).real();
//     }
//   };
// 
//   // The log likelihood of a item belonging to each cluster given the batch label.
//   virtual arma::vec itemLogLikelihood(arma::vec item) {
// 
//     double exponent = 0.0;
//     arma::vec ll(K), dist_to_mean(P);
//     ll.zeros();
//     dist_to_mean.zeros();
// 
//     for(arma::uword k = 0; k < K; k++){
// 
//       // The exponent part of the MVN pdf
//       dist_to_mean = item - mu.col(k);
//       exponent = arma::as_scalar(dist_to_mean.t() * cov_inv.slice(k) * dist_to_mean);
// 
//       // Normal log likelihood
//       ll(k) = -0.5 *(cov_log_det(k) + exponent + (double) P * log(2.0 * M_PI));
//     }
//     return(ll);
//   };
// 
//   virtual void calcBIC(){
// 
//     // BIC = 2 * model_likelihood;
// 
//     BIC = 2 * model_likelihood - n_param_cluster * std::log(N);
// 
//     // for(arma::uword k = 0; k < K; k++) {
//     //   BIC -= n_param_cluster * std::log(N_k(k) + 1);
//     // }
// 
//   };
// 
// 
//   virtual void sampleParameters() {
// 
//     arma::uword n_k = 0;
//     arma::vec mu_n(P), sample_mean(P);
//     arma::mat sample_cov(P, P), dist_from_prior(P, P), scale_n(P, P);
// 
//     for (arma::uword k = 0; k < K; k++) {
// 
//       // std::cout << "\nN_k (wrong): " << accu(labels == k);
//       // std::cout << "\nN_k (should be right): " << N_k(k);
// 
// 
//       // Find how many labels have the value
//       n_k = N_k(k);
//       
//       // std::cout << "\n\nMembers:\n" << size(find(members.col(k)));
//       // std::cout << "\n\nOutliers:\n" << size(find(non_outliers == 1));
//       // std::cout << "\n\nNon-outlier members:\n" << size(find(members.col(k) && non_outliers == 1));
//       
//       if(n_k > 0){
// 
//         // std::cout << "\n\nSampling new value";
//         
//         // Component data
//         arma::mat component_data = X.rows( arma::find(members.col(k) && (non_outliers == 1)) );
// 
//         // std::cout << "\n\nComponent data subsetted.\n" << size(component_data);
//         
//         // Sample mean in the component data
//         sample_mean = arma::mean(component_data).t();
// 
//         // std::cout << "\n\nSampled mean acquired:\n" << sample_mean;
//         
//         sample_cov = calcSampleCov(component_data, sample_mean, n_k, P);
// 
//         // std::cout << "\n\nSampled cov acquired:\n" << sample_cov;
//         
//         // Calculate the distance of the sample mean from the prior
//         dist_from_prior = (sample_mean - xi) * (sample_mean - xi).t();
//         
//         // std::cout << "\n\nDistance from prior calculated:\n" << dist_from_prior;
//         
//         // Update the scale hyperparameter
//         scale_n = scale + sample_cov + ((kappa * n_k) / (double) (kappa + n_k)) * dist_from_prior;
//         
//         // std::cout << "\n\nDistance from prior calculated:\n" << dist_from_prior;
//         // 
//         // std::cout << "\n\nCovariacne:\n" << cov.slice(k);
//         // std::cout << "\n\nNu: " << nu;
//         // std::cout << "\nN_k: " << n_k;
//         
//         // Sample a new covariance matrix
//         cov.slice(k) = arma::iwishrnd(scale_n, nu + n_k);
//         
//         // std::cout << "\n\nCovariance sampled";
//         
//         // The weighted average of the prior mean and sample mean
//         mu_n = (kappa * xi + n_k * sample_mean) / (double)(kappa + n_k);
//         
//         // std::cout << "\nMu_n calculated";
//         
//         // Sample a new mean vector
//         mu.col(k) = arma::mvnrnd(mu_n, (1.0 / (double) (kappa + n_k)) * cov.slice(k), 1);
//         
//         // std::cout << "\nMean sampled";
//         // std::cout << "\n\nDistance from prior calculated:\n" << dist_from_prior;
//         
//       } else{
// 
//         // If no members in the component, draw from the prior distribution
//         cov.slice(k) = arma::iwishrnd(scale, nu);
//         mu.col(k) = arma::mvnrnd(xi, (1.0 / (double) kappa) * cov.slice(k), 1);
// 
//       }
//       
//       // Save the inverse and log determinant of the new covariance matrices
//       cov_inv.slice(k) = arma::inv_sympd(cov.slice(k));
//       cov_log_det(k) = arma::log_det(cov.slice(k)).real();
//       
//     }
//   }
// };


// //' @name gaussianSampler
// //' @title Gaussian mixture type
// //' @description The sampler for a mixture of Gaussians, where each feature is
// //' assumed to be independent (i.e. a multivariate Normal with a diagonal
// //' covariance matrix).
// //' @field new Constructor \itemize{
// //' \item Parameter: K - the number of components to model
// //' \item Parameter: labels - the initial clustering of the data
// //' \item Parameter: concentration - the vector for the prior concentration of
// //' the Dirichlet distribution of the component weights
// //' \item Parameter: X - the data to model
// //' }
// //' @field printType Print the sampler type called.
// //' @field updateWeights Update the weights of each component based on current
// //' clustering.
// //' @field updateAllocation Sample a new clustering.
// //' @field sampleFromPrior Sample from the priors for the Gaussian density.
// //' @field calcBIC Calculate the BIC of the model.
// //' @field logLikelihood Calculate the likelihood of a given data point in each
// //' component. \itemize{
// //' \item Parameter: point - a data point.
// //' }
// class gaussianMixture: virtual public mixture {
// 
// public:
// 
//   double xi, kappa, alpha, g, h, a;
//   arma::vec beta;
//   arma::mat mu, tau;
// 
//   using mixture::mixture;
// 
//   // Parametrised
//   gaussianMixture(
//     arma::uword _K,
//     arma::uvec _labels,
//     // arma::vec _concentration,
//     arma::mat _X
//   ) : mixture(_K, _labels, _X) {
// 
//     double data_range_inv = pow(1.0 / (X.max() - X.min()), 2);
// 
//     alpha = 2.0;
//     g = 0.2;
//     a = 10;
// 
//     xi = arma::accu(X)/(N * P);
// 
//     kappa = data_range_inv;
//     h = a * data_range_inv;
// 
//     beta.set_size(P);
//     beta.zeros();
// 
//     mu.set_size(P, K);
//     mu.zeros();
// 
//     tau.set_size(P, K);
//     tau.zeros();
// 
//   }
// 
//   gaussianMixture(
//     arma::uword _K,
//     arma::uvec _labels,
//     // arma::vec _concentration,
//     arma::mat _X,
//     double _xi,
//     double _kappa,
//     double _alpha,
//     double _g,
//     double _h,
//     double _a
//   ) : mixture {_K, _labels, _X} {
// 
//     xi = _xi;
//     kappa = _kappa;
//     alpha = _alpha;
//     g = _g;
//     h = _h;
//     a = _a;
// 
//     beta.set_size(P);
//     beta.zeros();
// 
//     mu.set_size(P, K);
//     mu.zeros();
// 
//     tau.set_size(P, K);
//     tau.zeros();
// 
//   }
// 
//   // Destructor
//   virtual ~gaussianMixture() { };
// 
//   // Print the sampler type.
//   virtual void printType() {
//     std::cout << "\nType: Gaussian.\n";
//   }
// 
//   // Parameters for the mixture model. The priors are empirical and follow the
//   // suggestions of Richardson and Green <https://doi.org/10.1111/1467-9868.00095>.
//   void sampleFromPriors() {
//     for(arma::uword p = 0; p < P; p++){
//       beta(p) = arma::randg<double>( arma::distr_param(g, 1.0 / h) );
//       for(arma::uword k = 0; k < K; k++){
//         mu(p, k) = (arma::randn<double>() / kappa) + xi;
//         tau(p, k) = arma::randg<double>( arma::distr_param(alpha, 1.0 / arma::as_scalar(beta(p))) );
//       }
// 
//     }
//   }
// 
// 
//   // Sample beta
//   void updateBeta(){
// 
//     double a = g + K * alpha;
//     double b = 0.0;
// 
//     for(arma::uword p = 0; p < P; p++){
//       b = h + arma::accu(tau.row(p));
//       beta(p) = arma::randg<double>( arma::distr_param(a, 1.0 / b) );
//     }
//   }
// 
//   // Sample mu
//   void updateMuTau() {
// 
//     arma::uword n_k = 0;
//     double _sd = 0, _mean = 0;
// 
//     double a, b;
//     arma::vec mu_k(P);
// 
//     for (arma::uword k = 0; k < K; k++) {
// 
//       // Find how many labels have the value
//       n_k = N_k(k);
//       if(n_k > 0){
// 
//         arma::mat component_data = X.rows( arma::find((labels == k) ) );
// 
//         for (arma::uword p = 0; p < P; p++){
// 
//           // The updated parameters for mu
//           _sd = 1.0/(tau(p, k) * n_k + kappa);
//           _mean = (tau(p, k) * arma::sum(component_data.col(p)) + kappa * xi) / (1.0/_sd) ;
// 
// 
//           // Sample a new value
//           mu(p, k) = arma::randn<double>() * _sd + _mean;
// 
//           // Parameters of the distribution for tau
//           a = alpha + 0.5 * n_k;
// 
//           arma::vec b_star = component_data.col(p) - mu(p, k);
//           b = beta(p) + 0.5 * arma::accu(b_star % b_star);
// 
//           // The updated parameters
//           tau(p, k) = arma::randg<double>(arma::distr_param(a, 1.0 / b) );
//         }
//       } else {
//         for (arma::uword p = 0; p < P; p++){
//           // Sample a new value from the priors
//           mu(p, k) = arma::randn<double>() * (1.0/kappa) + xi;
//           tau(p, k) = arma::randg<double>(arma::distr_param(alpha, 1.0 / beta(p)) );
//         }
//       }
//     }
//   };
// 
//   void sampleParameters() {
//     updateMuTau();
//     updateBeta();
//   }
// 
//   arma::vec itemLogLikelihood(arma::vec item) {
// 
//     arma::vec ll(K);
//     ll.zeros();
// 
//     for(arma::uword k = 0; k < K; k++){
//       for (arma::uword p = 0; p < P; p++){
//         ll(k) += -0.5*(std::log(2) + std::log(PI) - std::log(arma::as_scalar(tau(p, k)))) - arma::as_scalar(0.5 * tau(p, k) * pow(item(p) - mu(p, k), 2));
//       }
//     }
//     return ll;
//   };
// 
//   virtual void calcBIC(){
// 
//     arma::uword n_param = (P + P) * K_occ;
//     BIC = n_param * std::log(N) - 2 * model_likelihood;
// 
//   }
// 
// };

// 
// //' @name tAdjustedMixture
// //' @title Base class for adding a t-distribution to sweep up outliers in the
// //' model.
// //' @description The class that the specific TAGM types inherit from.
// //' @field new Constructor \itemize{
// //' \item Parameter: K - the number of components to model
// //' \item Parameter: labels - the initial clustering of the data
// //' \item Parameter: concentration - the vector for the prior concentration of
// //' the Dirichlet distribution of the component weights
// //' \item Parameter: X - the data to model
// //' }
// //' @field updateOutlierWeights Updates the weight of the outlier component.
// //' @field updateWeights Update the weights of each component based on current
// //' clustering, excluding the outliers.
// //' @field sampleOutlier Sample is the nth item is an outlier. \itemize{
// //' \item Parameter n: the index of the individual in the data matrix and
// //' allocation vector.
// //' }
// //' @field updateAllocation Sample a new clustering and simultaneously allocate
// //' items to the outlier distribution.
// //' @field calcTdistnLikelihood Virtual placeholder for the function that calculates
// //' the likelihood of a given point in a t-distribution. \itemize{
// //' \item Parameter: point - a data point.
// //' }
// class tAdjustedMixture : virtual public mixture {
// 
// private:
// 
// public:
//   // for use in the outlier distribution
//   double global_log_det = 0.0, t_likelihood_const = 0.0;
//   // arma::uvec outlier;
//   
//   arma::vec global_mean, t_ll;
//   arma::mat global_cov_inv;
//   double df = 4.0, u = 2.0, v = 10.0, b = 0.0, outlier_weight = 0.0;
// 
//   using mixture::mixture;
// 
//   tAdjustedMixture(arma::uword _K,
//                    arma::uvec _labels,
//                    arma::mat _X
//   ) : mixture(_K, _labels, _X) {
// 
//     // Doubles for leading coefficient of outlier distribution likelihood
//     double lgamma_df_p = 0.0, lgamma_df = 0.0, log_pi_df = 0.0;
//     vec dist_in_T(N), diff_data_global_mean(P);
//     
//     // for use in the outlier distribution
//     mat global_cov = 0.5 * cov(X);
//     
//     uword count_here = 0;
//     while( (rcond(global_cov) < 1e-5) && (count_here < 20) ) {
//       global_cov.diag() += 1 * 0.005;
//       count_here++;
//     }
//     
//     // std::cout << "\n\nGlobal covariance:\n" << global_cov;
//     
//     global_mean = (mean(X, 0)).t();
// 
//     // std::cout << "\n\nGlobal mean:\n" << global_mean;
//     
//     global_log_det = log_det(global_cov).real();
//     global_cov_inv = inv(global_cov);
// 
//     // std::cout << "\n\nGlobal mean: \n" << global_mean;
//     // std::cout << "\n\nGlobal cov: \n" << global_cov;
//     
//     // outliers = arma::zeros<arma::uvec>(N);
// 
//     b = N; // (double) sum(non_outliers);
//     outlier_weight = rBeta(b + u, N + v - b);
// 
//     // Components of the t-distribution likelihood that do not change
//     lgamma_df_p = std::lgamma((df + (double) P) / 2.0);
//     lgamma_df = std::lgamma(df / 2.0);
//     log_pi_df = (double) P / 2.0 * std::log(df * M_PI);
//     
//     // Constant in t likelihood
//     t_likelihood_const = lgamma_df_p - lgamma_df - log_pi_df - 0.5 * global_log_det;
//     
//     // std::cout << "\n\nT pdf leading coefficient: " << t_likelihood_const;
//     
//     // diff_data_global_mean = X_t.each_col() - global_mean;
//     
//     t_ll.set_size(N);
//     for(uword n = 0; n < N; n++){
//       
//       diff_data_global_mean = X_t.col(n) - global_mean;
//       
//       // dist_in_T(n) = as_scalar(diff_data_global_mean.col(n).t() 
//       //   * global_cov_inv 
//       //   * diff_data_global_mean.col(n)
//       // );
//       
//       dist_in_T(n) = as_scalar(diff_data_global_mean.t() 
//        * global_cov_inv 
//        * diff_data_global_mean
//       );
//       
//       // std::cout << "\n\nT-distance form centre: " << dist_in_T(n);
//       // 
//       // std::cout << "\n\nExponent of likelihood: " << ((df + (double) P) / 2.0) * std::log(1.0 + (1.0 / df) * dist_in_T(n));
//       
//       t_ll(n) = t_likelihood_const 
//         - ((df + (double) P) / 2.0) * std::log(1.0 + (1.0 / df) * dist_in_T(n));
//       
//       // std::cout << "\n\nOutlier likelihood: " << t_ll(n);
//       
//     }
//   };
// 
//   // Destructor
//   virtual ~tAdjustedMixture() { };
// 
//   // double calcTdistnLikelihood(arma::vec point) {
//   double calcTdistnLikelihood(arma::uword n) {
// 
//     double exponent = 0.0, ll = 0.0;
// 
//     exponent = as_scalar(
//       (X_t.col(n) - global_mean).t()
//       * global_cov_inv
//       * (X_t.col(n) - global_mean)
//     );
//     
//     // std::cout << "\nConstant: " << t_likelihood_const;
//     // std::cout << "\nGlobal covariance log determinant: " << 0.5 * global_log_det;
//     
//     // std::cout << dist_in_T(n);
// 
//     ll = t_likelihood_const 
//       - ((df + (double) P) / 2.0) * std::log(1.0 + (1.0 / df) * exponent);
// 
//     // std::cout << "\nLog likelihood: " << ll;
//       
//     return ll;
//   };
// 
//   void updateOutlierWeights(){
//     b = (double) sum(outliers);
//     outlier_weight = rBeta(b + u, N + v - b);
//     
//     // std::cout << "\n\nOutlier weight: " << outlier_weight;
//     
//   };
// 
//   // void updateWeights(){
//   //
//   //   double a = 0.0;
//   //
//   //   for (arma::uword k = 0; k < K; k++) {
//   //
//   //     // Find how many labels have the value
//   //     members.col(k) = (labels == k) % outliers;
//   //     N_k(k) = arma::sum(members.col(k));
//   //
//   //     // Update weights by sampling from a Gamma distribution
//   //     a  = concentration(k) + N_k(k);
//   //     w(k) = arma::randg( arma::distr_param(a, 1.0) );
//   //   }
//   //
//   //   // Convert the cluster weights (previously gamma distributed) to Beta
//   //   // distributed by normalising
//   //   w = w / arma::sum(w);
//   // };
// 
//   arma::uword sampleOutlier(arma::uword n) {
// 
//     uword pred_outlier = 0;
//     // arma::uword k = labels(n);
// 
//     double out_likelihood = 0.0;
//     arma::vec outlier_prob(2), point = X_t.col(n);
//     outlier_prob.zeros();
// 
//     // The likelihood of the point in the current cluster
//     outlier_prob(0) = likelihood(n) + log(1 - outlier_weight);
// 
//     // Calculate outlier likelihood
//     // out_likelihood = calcTdistnLikelihood(n); //calcTdistnLikelihood(point);
//     // 
//     // if(t_ll(n) != out_likelihood) {
//     //   std::cout << "\n\nOutlier ind ll: " << out_likelihood;
//     //   std::cout << "\nOutlier big ll: " << t_ll(n);
//     //   throw;
//     // }
//     
//     out_likelihood = t_ll(n) + log(outlier_weight);
//     outlier_prob(1) = out_likelihood;
// 
//     // std::cout << "\n\nOutlier probability:\n" << outlier_prob;
//     
//     // Normalise and overflow
//     outlier_prob = exp(outlier_prob - max(outlier_prob));
//     outlier_prob = outlier_prob / sum(outlier_prob);
// 
//     // Prediction and update
//     u = arma::randu<double>( );
//     pred_outlier = sum(u > cumsum(outlier_prob));
// 
//     return pred_outlier;
//   };
// 
//   
//   
//   virtual void updateAllocation(arma::vec weights, arma::mat upweigths) {
// 
//     double u = 0.0;
//     arma::uvec uniqueK;
//     arma::vec comp_prob(K);
// 
//     // First update the outlier parameters
//     updateOutlierWeights();
// 
//     for (auto& n : unfixed_ind) {
//     // for(arma::uword n = 0; n < N; n++){
//       ll = itemLogLikelihood(X_t.col(n));
// 
//       // Update with weights
//       comp_prob = ll + log(weights) + log(upweigths.col(n));
// 
//       // Normalise and overflow
//       comp_prob = exp(comp_prob - max(comp_prob));
//       comp_prob = comp_prob / sum(comp_prob);
// 
//       // Prediction and update
//       u = arma::randu<double>( );
//       labels(n) = sum(u > cumsum(comp_prob));
//       alloc.row(n) = comp_prob.t();
// 
//       // Record the likelihood of the item in it's allocated component
//       likelihood(n) = ll(labels(n));
// 
//       // Update if the point is an outlier or not
//       outliers(n) = sampleOutlier(n);
//     }
// 
//     // Update our vector indicating if we're not an outlier
//     non_outliers = 1 - outliers;
//     
//     // std::cout << "\n\nNumber outliers: " << sum(outliers);
//     // std::cout << "\nNumber non-outliers: " << sum(non_outliers);
//     
//     // The model log likelihood
//     model_likelihood = accu(likelihood);
// 
//     // Number of occupied components (used in BIC calculation)
//     uniqueK = unique(labels);
//     K_occ = uniqueK.n_elem;
//   };
// 
// };

// 
// //' @name tagmMVN
// //' @title T-ADjusted Gaussian Mixture (TAGM) type
// //' @description The sampler for the TAGM mixture model.
// //' @field new Constructor \itemize{
// //' \item Parameter: K - the number of components to model
// //' \item Parameter: labels - the initial clustering of the data
// //' \item Parameter: concentration - the vector for the prior concentration of
// //' the Dirichlet distribution of the component weights
// //' \item Parameter: X - the data to model
// //' }
// //' @field printType Print the sampler type called.
// //' @field calcBIC Calculate the BIC of the model.
// //' @field calcTdistnLikelihood Calculate the likelihood of a given data point
// //' the gloabl t-distirbution. \itemize{
// //' \item Parameter: point - a data point.
// //' }
// class tagmMixture : public tAdjustedMixture, public mvnMixture {
// 
// private:
// 
// public:
//   // for use in the outlier distribution
//   // arma::uvec outlier;
//   // arma::vec global_mean;
//   // arma::mat global_cov;
//   // double df = 4.0, u = 2.0, v = 10.0, b = 0.0, outlier_weight = 0.0;
// 
//   using mvnMixture::mvnMixture;
// 
//   tagmMixture(arma::uword _K,
//           arma::uvec _labels,
//           // arma::vec _concentration,
//           arma::mat _X
//   ) : mvnMixture(_K, _labels, _X),
//   tAdjustedMixture(_K, _labels, _X),
//   mixture(_K, _labels, _X){
//   };
// 
//   // Destructor
//   virtual ~tagmMixture() { };
// 
//   void printType() {
//     std::cout << "Type: TAGM.\n";
//   };
// 
//   virtual void calcBIC(){
// 
//     arma::uword n_param = (P + P * (P + 1) * 0.5) * (K_occ + 1);
//     BIC = n_param * std::log(N) - 2 * model_likelihood;
// 
//   }
// 
// };
// 
// // Factory for creating instances of samplers
// //' @name mixtureFactory
// //' @title Factory for different types of mixtures.
// //' @description The factory allows the type of mixture implemented to change
// //' based upon the user input.
// //' @field new Constructor \itemize{
// //' \item Parameter: samplerType - the density type to be modelled
// //' \item Parameter: K - the number of components to model
// //' \item Parameter: labels - the initial clustering of the data
// //' \item Parameter: concentration - the vector for the prior concentration of
// //' the Dirichlet distribution of the component weights
// //' \item Parameter: X - the data to model
// //' }
// class mixtureFactory
// {
// public:
// 
//   // empty contructor
//   mixtureFactory(){ };
// 
//   // destructor
//   virtual ~mixtureFactory(){ };
// 
//   // copy constructor
//   mixtureFactory(const mixtureFactory &L);
// 
//   // // assignment
//   // mixtureFactory & operator=(const mixtureFactory &L) {
//   //   if (this != &L) {
//   //   }
//   //   return *this;
//   // };
// 
// 
//   enum mixtureType {
//     // G = 0,
//     MVN = 1,
//     // C = 2,
//     TMVN = 3 //,
//     // TG = 4
//   };
// 
//   static std::unique_ptr<mixture> createMixture(mixtureType type,
//                                                 arma::uword K,
//                                                 arma::uvec labels,
//                                                 arma::mat X) {
//     switch (type) {
//     // case G: return std::make_unique<gaussianSampler>(K, labels, concentration, X);
//     case MVN: return std::make_unique<mvnMixture>(K, labels, X);
//     // case C: return std::make_unique<categoricalSampler>(K, labels, concentration, X);
//     case TMVN: return std::make_unique<tagmMixture>(K, labels, X);
//     // case TG: return std::make_unique<tagmGaussian>(K, labels, concentration, X);
//     default: throw std::invalid_argument( "invalid sampler type." );
//     }
// 
//   }
// 
// };


// // Wrapper to make a vector of mixtures
// //' @name VecWrapper
// //' @title Wrapper for a vector of different mixture types.
// //' @description Create a vector of different mixture types.
// //' @field new Constructor \itemize{
// //' \item Parameter: L - unsigned integer; the number of datasets (and thus mixtures)
// //' \item Parameter: K - vector of unsigned integers; the number of components to model in each mixture
// //' \item Parameter: labels - matrix of unsigned integers, the initial clustering of the data
// //' \item Parameter: X - field of matrices; the datasets.
// //' the Dirichlet distribution of the component weights
// //' \item Parameter: X - the data to model
// //' }
// class VecWrapper
// {
// public:
//
//   // constructor; initialize the list to be empty
//   // VecWrapper();
//
//   // destructor
//   ~VecWrapper();
//
//   // // copy constructor
//   // VecWrapper(const VecWrapper &L);
//
//   // // assignment
//   // VecWrapper & operator=(const VecWrapper &L);
//
//   // std::vector<gaussianMixture> ver;
//   std::vector<std::unique_ptr<mixture>>* ver;
//   // std::vector<std::unique_ptr<mixture>> ver;
//
//   // Vector* one = malloc(sizeof(*one))
//
//   // VecWrapper(uword L, uvec K, umat labels, field<mat> X)
//
//   // Constructor with arguments
//   VecWrapper(uword L, uvec K, uvec type, umat labels, field<mat> X)
//   {
//     // std::cout << "\n\nWe reach th wrapper.";
//     // ver->reserve(L);
//     ver.reserve(L);
//
//     mixtureFactory my_factory;
//
//     for(arma::uword l = 0; l < L; ++l ){
//
//       // Convert from an int to the samplerType variable for our Factory
//       mixtureFactory::mixtureType val = static_cast<mixtureFactory::mixtureType>(type(l));
//
//       // Make a pointer to the correct type of sampler
//       // (*ver)[l] = my_factory.createMixture(val,
//       //   K(l),
//       //   labels.col(l),
//       //   X(l)
//       // );
//
//       ver->push_back(my_factory.createMixture(val,
//         K(l),
//         labels.col(l),
//         X(l)
//       ));
//
//
//
//       // std::unique_ptr<mixture> mixture_ptr = my_factory.createMixture(val,
//       //   K(l),
//       //   labels.col(l),
//       //   X(l)
//       // );
//       // ver.push_back(mixture_ptr);
//
//       // ver->push_back(mixture_ptr);
//
//       // delete mixture_ptr;
//
//       // gaussianMixture my_sampler(K(l), labels.col(l), X(l));
//       // ver.push_back(my_sampler);
//     }
//     // free(ver);
//   }
// };
// 
// class semiSupervisedMixture : virtual public mixture {
// private:
//   
// public:
//   
//   semiSupervisedMixture(arma::uword _K,
//                    arma::uvec _labels,
//                    arma::uvec _fixed,
//                    arma::mat _X
//   ) : mixture(_K, _labels, _X) {
//     
//     // Pass the indicator vector for being fixed to the ``fixed`` object.
//     fixed = _fixed;
//     
//     uword N_fixed = accu(fixed);
//     
//     fixed_ind = find(fixed == 1);
//     unfixed_ind = find(fixed == 0);
//     
//     // Set the known label allocations to 1
//     for(uword n = 0; n < N; n++) {
//       if(fixed(n) == 1) {
//         alloc(n, labels(n)) = 1.0;
//       }
//     }
//     
//   };
//   
// };
// 

// class semiSupervisedMVN :
//   virtual public mvnMixture,
//   virtual public semiSupervisedMixture
// {
// 
// public:
// 
//   uvec outliers, non_outliers;
// 
//   using mvnMixture::mvnMixture;
// 
//   semiSupervisedMVN(arma::uword _K,
//                     arma::uvec _labels,
//                     arma::uvec _fixed,
//                     arma::mat _X) :
//     mixture(_K, _labels, _X),
//     mvnMixture(_K, _labels, _X),
//     semiSupervisedMixture(_K, _labels, _fixed, _X)
//      {
// 
// 
//     outliers = zeros<uvec>(N);
//     non_outliers = ones<uvec>(N);
// 
//     fixed = _fixed;
// 
//     uword N_fixed = accu(fixed);
// 
//     fixed_ind = find(fixed == 1);
//     unfixed_ind = find(fixed == 0);
// 
//     // std::cout << "\nNumber fixed: " << accu(fixed);
//     // std::cout << "\nFixed inds:\n" << fixed_ind.head(4);
//     // std::cout << "\nUnfixed inds:\n" << unfixed_ind.head(4);
//     //
//     // std::cout << "\nMembers:\n" << size(members);
//     //
//     // std::cout << "\nN_k:\n" << N_k;
// 
//   };
// 
//   // Destructor
//   virtual ~semiSupervisedMVN() { };
// 
// };

// 
// class semiSupervisedTAGM : 
//   virtual public tagmMixture,
//   virtual public semiSupervisedMixture
//   
// {
//   
// public:
//   
//   using tagmMixture::tagmMixture;
//   
//   semiSupervisedTAGM(arma::uword _K,
//               arma::uvec _labels,
//               arma::uvec _fixed,
//               arma::mat _X
//   ) : 
//     mixture(_K, _labels, _X),
//     semiSupervisedMixture(_K, _labels, _fixed, _X),
//     // mvnMixture(_K, _labels, _X),
//     tagmMixture(_K, _labels, _X){
//     
//     // In this case initialise the outlier members as the non-fixed points
//     outliers = 1 - fixed;
//     non_outliers = fixed;
//     
//     // std::cout << "\nNumber fixed: " << accu(fixed);
//     // std::cout << "\nFixed inds: " << fixed_ind.head(4);
//     // std::cout << "\nNumber fixed: " << unfixed_ind.head(4);
//     
//   };
//   
//   // Destructor
//   virtual ~semiSupervisedTAGM() { };
//   
// };


class semiSupervisedMixtureFactory {
  
public:
  
  
  // empty contructor
  semiSupervisedMixtureFactory(){ };
  
  // destructor
  virtual ~semiSupervisedMixtureFactory(){ };
  
  // copy constructor
  semiSupervisedMixtureFactory(const semiSupervisedMixtureFactory &L);
  
  enum mixtureType {
    // G = 0,
    MVN = 1,
    // C = 2,
    TMVN = 3 //,
    // TG = 4
  };
  
  static std::unique_ptr<semiSupervisedMixture> createMixture(mixtureType type,
                                                              arma::uword K,
                                                              arma::uvec labels,
                                                              arma::uvec fixed,
                                                              arma::mat X) {
    switch (type) {
    // case G: return std::make_unique<gaussianSampler>(K, labels, concentration, X);
    case MVN: return std::make_unique<semiSupervisedMVN>(K, labels, fixed, X);
      // case C: return std::make_unique<categoricalSampler>(K, labels, concentration, X);
    case TMVN: return std::make_unique<semiSupervisedTAGM>(K, labels, fixed, X);
      // case TG: return std::make_unique<tagmGaussian>(K, labels, concentration, X);
    default: throw std::invalid_argument( "invalid sampler type." );
    }
  }
};


class mdiModel {

private:

public:

  arma::uword N, L, LC2 = 1, K_max, K_prod, K_to_the_L, n_combinations;
  // int LC2 = 1;
  double 
    // Normalising constant
    Z = 0.0,
      
    // Strategic latent variable
    v = 0.0,
    
    // Prior hyperparameters for component weights
    w_shape_prior = 2.0,
    w_rate_prior = 2.0,
    
    // Prior hyperparameters for MDI phi parameters
    phi_shape_prior = 1.0,
    phi_rate_prior = 0.2;

  
  arma::uvec K,         // Number of clusters in each dataset
    one_to_K,           // [0, 1, ..., K]
    one_to_L,           // [0, 1, ..., L] 
    KL_powers,          // K to the power of the members of one_to_L
    rows_to_shed,       // rows initialised assuming symmetric K that are shed
    types,              // mixture types used 
    K_unfixed,          // Number of components not fixed
    K_fixed;            // Number of components fixed (i.e. at least one member has an observed label)

  arma::vec phis;

  // Weight combination indices
  arma::umat labels,

    // Various objects used to calculate MDI weights, normalising constant and phis
    comb_inds,
    phi_indicator,
    phi_ind_map,
    phi_indicator_t,
  
    // Class membership in each dataset
    N_k,
  
    // Indicator matrix for item n being an outlier in dataset l
    outliers,
  
    // Indicator matrix for item n being welll-described by its component
    // in dataset l
    non_outliers,
    
    fixed;

  // The labels, weights in each dataset and the dataset correlation
  arma::mat w, phi_indicator_t_mat;

  // Cube of cluster members
  arma::ucube members;

  arma::field<arma::uvec> fixed_ind;
  
  // The data can have varying numbers of columns
  arma::field<arma::mat> X;

  // The collection of mixtures
  // std::vector<gaussianMixture> mixtures;
  // std::vector< std::unique_ptr<mixture> > mixtures;
  std::vector< std::unique_ptr<semiSupervisedMixture> > mixtures;
  
  // // Cluster weights in each dataset. As possibly ragged need a field not a matrix.
  // arma::field<arma::vec> w;

  mdiModel(
    arma::field<arma::mat> _X,
    uvec _types,
    arma::uvec _K,
    arma::umat _labels,
    arma::umat _fixed
  ) {

    // These are used locally in some constructions
    arma::uword k = 0, col_ind = 0;

    // Mixture types
    types = _types;

    // std::cout << "\n\nTypes:\n" << types;
    
    // First allocate the inputs to their saved, class

    // The number of datasets modelled
    L = size(_X)(0);

    // std::cout << "\n\nNumber slices: " << _X.n_slices;
    // std::cout << "\n\nSize: " << size(_X);
    // std::cout << "\n\nSize 0: " << size(_X)(0);

    // The number of pairwise combinations
    // LC2 = 2;
    // throw (LC2);
    
    // LC2 = std::max(the_number_one, L * (L - 1) / 2);
    if(L > 1) {
      LC2 = L * (L - 1) / 2;
    }

    // std::cout << "\n\nLC2: " << LC2;
    // throw;
    
    // Used to check all datasets have matching number of rows
    arma::uvec N_check(L);

    // The number of components modelled in each dataset
    K = _K;
    K_max = max(K);
    // K_prod = prod(K);

    // Various objects that are used in the book-keeping for MDI. The weights,
    // normalising constant and correlation coefficients all involve some pretty
    // intense book-keeping.
    K_to_the_L = std::pow(K_max, L);

    // The count of members in each cluster
    N_k.set_size(K_max, L);
    N_k.zeros();
    
    // throw std::invalid_argument( "K declared?" );

    // std::cout << "\nCombination indicator failing?\nK to the L: "<< K_to_the_L;

    // We want to track which combinations should be unweighed and by what phi.
    // This object will be used in calculating the normalising constant (Z), the
    // cluster weights (gammas) and the correlation coefficient (phis)
    // along with the phi_indicator matrix.
    comb_inds.set_size(K_to_the_L, L);
    comb_inds.zeros();

    // The gammas combine in a form like (letting g denote gamma)
    // g_{11} g_{12} ... g_{1L}
    //          .
    //          .
    //          .
    // g_{K1} g_{12} ... g_{1L}
    // g_{11} g_{22} ... g_{1L}
    //          .
    //          .
    //          .
    // g_{K1} g_{K2} ... g_{KL}
    // 
    // Hence our matrix is initially of size K^L \times L
    
    // std::cout << "\nCombination indicator declared.\nK to the L: "<< K_to_the_L;

    one_to_K = linspace<uvec>(0, K_max - 1, K_max);
    one_to_L = linspace<uvec>(0, L - 1, L);

    // The matrix used to construct the rate for sampling the cluster weights
    KL_powers.set_size(L);
    for(uword l = 0; l < L; l++) {
      KL_powers(l) = std::pow(K_max, l);
      for(uword i = 0; i < K_to_the_L; i++){

        // We want the various combinations of the different gammas / cluster weights.
        // This format makes it relatively easy to figure out the upscaling too
        // (see phi_indicator below).
        k = (i / KL_powers(l));
        k = k % K_max;
        comb_inds(i, l) = k;
      }
    }
    
    // std::cout << "Combination indices:\n" << comb_inds << "\n\n";

    // Drop any rows that contain weights for clusters that shouldn't be 
    // modelled (i.e. if unique K_l are used)
    for(arma::uword l = 0; l < L; l++) {
      rows_to_shed = find(comb_inds.col(l) >= K(l));
      comb_inds.shed_rows(rows_to_shed);
    }

    // The final number of combinations
    n_combinations = comb_inds.n_rows;

    // std::cout << n_combinations;
    
    // This is used enough that we may as well define it
    // ones_mat.ones(LC2, n_combinations);

    // std::cout << "\n\nN combinations: " << n_combinations;

    // Now construct a matrix to record which phis are upweighing which weight
    // products, via an indicator matrix. This matrix has a column for each phi
    // (ncol = LC2) and a row for each combination (nrow = n_combinations).
    // This is working with the combi_inds object above. That indicated which 
    // weights to use, this indicates the corresponding up weights (e.g.,
    // we'd expect the first row to be all ones as all weights are for the first
    // component in each dataset, similarly for the last row).
    phi_indicator.set_size(n_combinations, LC2);
    phi_indicator.zeros();

    // std::cout << "\n\nPhi indicator declared.";

    // Map between a dataset pair and the column index. This will be a lower
    // triangular matrix of unsigned ints
    phi_ind_map.set_size(L, L);
    phi_ind_map.zeros();

    // Column index is, for some pair of datasets l and m,
    // \sum_{i = 0}^{l} (L - i - 1) + (m - l - 1). As this is awkward, let's
    // just use col_ind++.
    col_ind = 0;

    // Iterate over dataset pairings
    for(uword l = 0; l < (L - 1); l++) {
      for(uword m = l + 1; m < L; m++) {
        phi_indicator.col(col_ind) = (comb_inds.col(l) == comb_inds.col(m));

        // Record which column index maps to which phi
        phi_ind_map(m, l) = col_ind;

        // The column index is awkward, this is the  easiest solution
        col_ind++;

      }
    }

    // We use the transpose a surprising amount to ensure correct typing
    phi_indicator_t = phi_indicator.t();

    // And we often multiply by doubles, so convert to a matrix of doubles.
    phi_indicator_t_mat = conv_to<mat>::from(phi_indicator_t);

    // A weight vector for each dataset. Note that for ease of manipulations we
    // are using K_max and not K(l); this is to avoid ragged fields of vectors.
    w.set_size(K_max, L);
    w.zeros();



    // The ``correlation'' coefficient between each dataset clustering. Use the
    // size that best corresponds to phi_indicator. Possibly a row vector would
    // be easier.
    phis.set_size(LC2);
    phis.zeros();

    // Sample from prior?
    // phis = randg(LC2, distr_param(2.0, 1/2.0));

    // The initial labels
    labels = _labels;

    // std::cout << "\n\nAre we here?";

    X = _X;

    // Iterate over the number of datasets (currently must be 2)
    for(arma::uword l = 0; l < L; l++) {

      // std::cout << "\nl: " << l;

      // X(l) = _X(l);
      //
      // std::cout << "\nX assigned.";

      N_check(l) = _X(l).n_rows;

      // std::cout << "\nN_check assigned.";

      // // A weight for each component modelled
      // w(l).set_size(K(l));
    }

    if(arma::any(N_check != N_check(0))) {
      throw std::invalid_argument( "\n\nDatasets not matching in number of rows." );
    }

    // std::cout << "\n\nDo we reach this?";

    // The number of samples in each dataset
    N = N_check(0);

    // The members of each cluster across datasets. Each slice is a binary matrix
    // of the members of the kth class across the datasets.
    members.set_size(N, K_max, L);
    members.zeros();

    // for(uword l = 0; l < L; l++) {
    //   for(k = 0; k < K(l); k++) {
    //     members.slice(l).col(k) = labels.col(l) == k;
    //   }
    // }

    // std::cout << "\n\nHonestly this was an obvious expectation.\n";

    // These are used in t-adjusted mixtures. In all other models they should
    // never be changed.
    // outliers.set_size(N, L);
    // outliers.zeros();
    // 
    non_outliers.set_size(N, L);
    non_outliers.ones();
    
    fixed = _fixed;
    // fixed_ind.set_size(L);
    
    
    K_fixed.set_size(L);
    K_unfixed.set_size(L);

    uvec fixed_l, fixed_labels, labels_l, fixed_components;

    for(uword l = 0; l < L; l++){

      fixed_l = find(fixed.col(l) == 1);
      labels_l = labels.col(l);


      // fixed_ind(l) = fixed_l;
      fixed_labels = labels_l(fixed_l);
      fixed_components = arma::unique(fixed_labels);
      K_fixed(l) = fixed_components.n_elem;
      K_unfixed(l) = K(l) - K_fixed(l);
    }
    
    // throw std::invalid_argument("Throw reached.");
    
    // outliers = zeros<umat>(N, L); //1 - fixed;
    // non_outliers = ones<umat>(N, L);
    
    
  };


  // Destructor
  virtual ~mdiModel() { };

  // This wrapper to declare the mixtures at the dataset level is kept as its
  // own separate function to make a semi-supervised class easier
  void initialiseMixtures() {
    
    // std::cout << "\n\nInitialising mixtures?\n";
    
    // delete mixtures*;
    
    // std::vector< std::unique_ptr<semiSupervisedMixture> > mixtures;
    
    // Initialise the collection of mixtures (will need a vector of types too,, currently all are MVN)
    mixtures.reserve(L);
    
    semiSupervisedMixtureFactory my_factory;
    
    for(uword l = 0; l < L; l++) {
      semiSupervisedMixtureFactory::mixtureType val = static_cast<semiSupervisedMixtureFactory::mixtureType>(types(l));
      
      // Push it to the back of the vector
      mixtures.push_back(my_factory.createMixture(val,
          K(l),
          labels.col(l),
          fixed.col(l),
          X(l)
        )
      );
      
      // std::cout << "\n\nWhat:\n" << size(non_outliers) << "\n" << size(mixtures[l]->non_outliers);
      
      // We have to pass this back up
      non_outliers.col(l) = mixtures[l]->non_outliers;
    }
  };
  
  // virtual void initialiseMixtures() {
  //   // Initialise the collection of mixtures (will need a vector of types too,, currently all are MVN)
  //   // VecWrapper myMixtures(L, K, types, labels, X);
  //   // mixtures = myMixtures.ver;
  // 
  //   mixtures.reserve(L);
  // 
  //   mixtureFactory my_factory;
  // 
  //   for(uword l = 0; l < L; l++) {
  //     mixtureFactory::mixtureType val = static_cast<mixtureFactory::mixtureType>(types(l));
  // 
  //     // Make a pointer to the correct type of sampler
  //     // (*ver)[l] = my_factory.createMixture(val,
  //     //   K(l),
  //     //   labels.col(l),
  //     //   X(l)
  //     // );
  // 
  //     // Make a pointer to a mixture created by the factory to handle polymorphisms
  //     // std::unique_ptr<mixture> mix_l = my_factory.createMixture(val,
  //     //   K(l),
  //     //   labels.col(l),
  //     //   X(l)
  //     // );
  // 
  //     // Push it to the back of the vector
  //     mixtures.push_back(my_factory.createMixture(val,
  //         K(l),
  //         labels.col(l),
  //         X(l)
  //       )
  //     );
  //   }
  // 
  // };

  double calcWeightRate(uword lstar, uword kstar) {
    // The rate for the (k, l)th cluster weight is the sum across all of the clusters
    // in each dataset (excluding the lth) of the product of the kth cluster
    // weight in each of the L datasets (excluding the lth) upweigthed by
    // the pairwise correlation across all LC2 dataset combinations.

    // The rate we return.
    double rate = 0.0;

    // A chunk of these objects are used as our combinations matrix includes the
    // information for k != kstar and l != lstar, so we can shed some data.
    // Probably some of these can be dropped and calculated once at the class // [[Rcpp::export]]
    // level, but to allow K_l != K_m I suspect it will be different for each
    // dataset (I haven't done the maths though) and thus the objects might be
    // awkward
    uword n_used;
    uvec relevant_inds;
    vec weight_products, phi_products;
    umat relevant_combinations;
    mat relevant_phis, phi_prod_mat;

    relevant_inds = find(comb_inds.col(lstar) == kstar);
    relevant_combinations = comb_inds.rows(relevant_inds);
    relevant_phis =  phi_indicator_t_mat.cols(relevant_inds);
    n_used = relevant_combinations.n_rows;

    weight_products.ones(n_used);
    phi_products.ones(n_used);


    // The phi products (should be a matrix of 0's and phis)
    phi_prod_mat = relevant_phis.each_col() % phis;

    // Add 1 to each entry to have the object ready to be multiplied
    phi_prod_mat++;

    // Vector of the products, this should have the \prod (1 + \phi_{lm} ind(k_l = k_m))
    // ready to multiply by the weight products
    phi_products = prod(phi_prod_mat, 0).t();

    vec w_l(K_max);
    for(uword l = 0; l < L; l++) {
      if(l != lstar){
        w_l = w.col(l);
        weight_products = weight_products % w_l.elem(relevant_combinations.col(l));
      }
    }

    // The rate for the gammas
    rate = v * accu(weight_products % phi_products);

    return rate;
  };

  // The rate for the phi coefficient between the lth and mth datasets.
  double calcPhiRate(uword lstar, uword mstar) {

    // The rate we return.
    double rate = 0.0;
    arma::vec w_l;

    // A chunk of these objects are used as our combinations matrix includes the
    // information for l = lstar and l = mstar, so we can shed some data.
    uword n_used;
    uvec relevant_inds;
    vec weight_products, phi_products, relevant_phis;
    umat relevant_combinations;
    mat relevant_phi_inidicators, phi_prod_mat;

    relevant_inds = find(comb_inds.col(lstar) == comb_inds.col(mstar));
    relevant_combinations = comb_inds.rows(relevant_inds);

    // std::cout << "\n\nPhi indicator matrix (t):\n" << phi_indicator_t_mat;
    // std::cout << "\n\nRelevant indices:\n" << relevant_inds;

    // We only need the relevant phi indicators
    relevant_phi_inidicators = phi_indicator_t_mat.cols(relevant_inds);

    // std::cout << "\n\nRelevant indicators pre shedding:\n" << relevant_phi_inidicators;

    // Drop phi_{lstar, mstar} from both the indicator matrix and the phis vector
    relevant_phi_inidicators.shed_row(phi_ind_map(mstar, lstar));

    // std::cout << "\n\nRelevant indicators post shedding:\n" << relevant_phi_inidicators;

    relevant_phis = phis;
    relevant_phis.shed_row(phi_ind_map(mstar, lstar));

    n_used = relevant_combinations.n_rows;

    weight_products.ones(n_used);
    phi_products.ones(n_used);

    // The phi products (should be a matrix of 0's and phis)
    phi_prod_mat = relevant_phi_inidicators.each_col() % relevant_phis;

    // Add 1 to each entry to have the object ready to be multiplied
    phi_prod_mat++;

    // Vector of the products, this should have the \prod (1 + \phi_{lm} ind(k_l = k_m))
    // ready to multiply by the weight products
    phi_products = prod(phi_prod_mat, 0).t();

    // std::cout << "\n\nPhi product matrix:\n" << phi_prod_mat;
    // std::cout << "\n\nPhi products:\n" << phi_products;
    // std::cout << "\n\nRelevant combinations:\n" << relevant_combinations;

    for(uword l = 0; l < L; l++) {
      if(l != lstar){
        w_l = w.col(l);
        weight_products = weight_products % w_l.elem(relevant_combinations.col(l));

      }
    }

    // std::cout << "\n\nCalculate phi rate.\n";

    // The rate for the gammas
    rate = v * accu(weight_products % phi_products);

    return rate;
  };



  // Update the cluster weights
  void updateWeights() {

    // std::cout << "\n\nWeights before update:\n" << w;

    // std::cout << "\n\nMembers:\n" << members;

    double shape = 0.0, rate = 0.0;
    uvec members_lk(N);

    for(uword l = 0; l < L; l++) {

      // std::cout << "\nIn L loop: " << l;
      // std::cout << "\n\nMembers l:\n" << members.slice(l);

      for(uword k = 0; k < K(l); k++) {

        // std::cout << "\nIn K_loop: " << k;

        // std::cout << "\nMembers slice l col k:\n" << members.slice(l).col(k);

        // std::cout << "\nNumber of non-outleirs: \n" << accu(non_outliers);
        
        
        // std::cout << "\n\nLabels in dataset " << l << ":\n" << labels.col(l);
        // std::cout << "\n\nLabels in dataset " << l << " of class " << k << ":\n" << (labels.col(l) == k);
        // std::cout << "\n\nNon-outliers in dataset " << l << ":\n" << non_outliers.col(l);
        
        // Find how many labels have the value of k. We used to consider which
        // were outliers and which were not, but the non-outliers still 
        // contribute to the component weight, but not to the component parameters
        // and we use ot hand this down to the local mixture, mistakenly using 
        // the same value for N_k for the component parameters and the weights.
        members_lk = 1 * ((labels.col(l) == k) % non_outliers.col(l));
        // members_lk = 1 * ((labels.col(l) == k)); // % non_outliers.col(l));
        
        // std::cout << "\n\nN_k (before outliers): " << accu(members_lk);
        
        // members_lk = members_lk % non_outliers.col(l);
        
        // std::cout << "\n\nN_k (after outliers): " << accu(members_lk);
        
        // std::cout << "\n\nNon_outliers (MDI):\n" << non_outliers.col(l);
        // std::cout << "\n\nNon_outliers (Mixture):\n" << mixtures[l]->non_outliers;
        
        // if(accu(non_outliers.col(l) != mixtures[l]->non_outliers) > 0) {
        //   std::cout << "\n\nDataset: " << l << "\nDisagreement non_outliers:\n" << accu(non_outliers.col(l) != mixtures[l]->non_outliers);
        // }
        // std::cout << "\n\nDataset: " << l << "\nCluster: " << k << "\nMembers:\n" << accu(members_lk);
        
        
        // members(span::all, span(l, l), span(k, k) = members_lk;
        members.slice(l).col(k) = members_lk;

        // std::cout << "\nMembers accessed.";

        N_k(k, l) = accu(members_lk);

        // std::cout << "\nN_k updated.";

        // The hyperparameters
        shape = 1 + N_k(k, l);
        
        // throw std::invalid_argument( "\nMy inverses diverged." );
        
        
        rate = calcWeightRate(l, k);

        // std::cout << "\n\nShape:" << shape;
        // std::cout << "\nRate:" << rate;

        // Sample a new weight
        w(k, l) = randg(distr_param(w_shape_prior + shape,
            1.0 / (w_rate_prior + rate)
          )
        );


        // throw std::invalid_argument( "\nMy inverses diverged." );
        // std::cout << "\nIssue";

        // Pass the allocation count down to the mixture
        // (this is used in the parameter sampling)
        // mixtures[l].N_k = N_k(span(0, K(l) - 1), l);
        // mixtures[l].members.col(k) = members_lk;
        // 
        // (*mixtures)[l]->N_k = N_k(span(0, K(l) - 1), l);
        // (*mixtures)[l]->members.col(k) = members_lk;
        
        
        // std::cout << "\n\nMembers (MDI) dim:n" << size(members_lk);
        // std::cout << "\n\nMembers (MDI) :n" << dim(members_lk);
        
        
        // std::cout << "\n\nMembers (Mixture) dim:n" << size(mixtures[l]->members);
        // std::cout << "\n\nMembers (Mixture) :n" << dim(members_lk);
        
        // throw std::invalid_argument( "\nCraic." );
        
        mixtures[l]->members.col(k) = members_lk;

        
        // std::cout << "\nWeight updated.";
      }
      
      // std::cout << "\n\nN_k (MDI):\n" << N_k;
      // std::cout << "\n\nN_k (mixture):\n" << mixtures[l]->N_k;
      
      // throw std::invalid_argument( "\nMy inverses diverged." );
      
      mixtures[l]->N_k = N_k(span(0, K(l) - 1), l);
      
      // If we only have one dataset, flip back to normalised weights
      if(L == 1) {
        w = w / accu(w) ;
      }
      
      // std::cout << "\n\nN_kl:\n" << mixtures[l]->N_k;
      
    }

    // std::cout << "\n\nN_k:\n" << N_k;
    
    // std::cout << "\n\nWeights after update:\n" << w;
  };

  double samplePhiShape(arma::uword l, arma::uword m, double rate) {
    bool rTooSmall = false, priorShapeTooSmall = false;
    
    uword r = 0, N_lm = 0;
    double shape = 0.0,
      u = 0.0,
      prod_to_phi_shape = 0.0, 
      prod_to_r_less_1 = 0.0;
    
    uvec rel_inds_l(N), rel_inds_m(N);
    
    vec weights;
    
    rel_inds_l = labels.col(l) % non_outliers.col(l);
    rel_inds_m = labels.col(m) % non_outliers.col(m);

    N_lm = accu(rel_inds_l == rel_inds_m);
    // N_lm = accu(labels.col(l) == labels.col(m));
    rate = calcPhiRate(l, m);
    weights = zeros<vec>(N_lm + 1);
    
    if(phi_shape_prior < 2) {
      priorShapeTooSmall = true;
      prod_to_phi_shape = 1.0; 
    }
    
    for(uword r = 0; r < (N_lm + 1); r++) {
      
      // Some of the products can be ``backwards'', i.e. the top index is less 
      // than (or equal to) the bottom index. If this occurs we want to keep the 
      // contribution of these products as the identity.
      
      // Reset vlaues that might have changed in the previous iteration
      rTooSmall = false;
      prod_to_phi_shape = 1.0; 
      prod_to_r_less_1= 1.0;
      
      if(r < 2) {
        rTooSmall = true;
      }
      
      if(! rTooSmall) {
        for(uword i = 0; i < r; i++) {
          prod_to_r_less_1 *= (N_lm - i);
        }
      }
      
      if(! priorShapeTooSmall) {
        for(uword j = 1; j < phi_shape_prior; j++) {
          prod_to_phi_shape *= (r + j);
        }
      }
      
      weights(r) = prod_to_r_less_1 * prod_to_phi_shape / std::pow(rate + phi_rate_prior, r + 1);
      

    }
    
    // Normalise the weights
    weights = weights / accu(weights);
    
    // Prediction and update
    u = randu<double>( );
    
    shape = sum(u > cumsum(weights)) ;
   
   return shape; 
  }
  
  // void updatePhis() {
  //   uword r = 0;
  //   double shape = 0.0, rate = 0.0;
  //   for(uword l = 0; l < (L - 1); l++) {
  //     for(uword m = l + 1; m < L; m++) {
  // 
  //       // Find the parameters based on the likelihood
  //       rate = calcPhiRate(l, m);
  //       shape = samplePhiShape(l, m, rate);
  // 
  // 
  //       // std::cout << "\n\nShape:" << shape;
  //       // std::cout << "\nRate:" << rate;
  // 
  //       phis(phi_ind_map(m, l)) = randg(distr_param(
  //         phi_shape_prior + shape,
  //         1.0 / (phi_rate_prior + rate)
  //       )
  //       );
  //     }
  //   }
  // }
  // 
  // Update the context similarity parameters
void updatePhis() {

  // std::cout << "\n\nPhis before update:\n" << phis;
  uword N_lm = 0;
  double shape = 0.0, rate = 0.0;
  uvec rel_inds_l(N), rel_inds_m(N);
  
  for(uword l = 0; l < (L - 1); l++) {
    for(uword m = l + 1; m < L; m++) {
      rel_inds_l = labels.col(l) % non_outliers.col(l);
      rel_inds_m = labels.col(m) % non_outliers.col(m);

      N_lm = accu(rel_inds_l == rel_inds_m);
      shape = 1 + N_lm;
      
      // shape = 1 + accu(labels.col(l) == labels.col(m));
      rate = calcPhiRate(l, m);

      // std::cout << "\n\nShape:" << shape;
      // std::cout << "\nRate:" << rate;

      phis(phi_ind_map(m, l)) = randg(distr_param(
          phi_shape_prior + shape,
          1.0 / (phi_rate_prior + rate)
        )
      );
    }
  }

    // std::cout << "\n\nPhis after update:\n" << phis;

  };

  // Updates the normalising constant for the posterior
  void updateNormalisingConst() {

    arma::vec w_l;

    // A chunk of these objects are used as our combinations matrix includes the
    // information for l = lstar and l = mstar, so we can shed some data.
    vec weight_products, phi_products;
    mat phi_prod_mat;

    weight_products.ones(n_combinations);
    phi_products.ones(n_combinations);

    // The phi products (should be a matrix of 0's and phis)
    phi_prod_mat = phi_indicator_t_mat.each_col() % phis;

    // Add 1 to each entry to have the object ready to be multiplied
    phi_prod_mat++;

    // Vector of the products, this should have the \prod (1 + \phi_{lm} ind(k_l = k_m))
    // ready to multiply by the weight products
    phi_products = prod(phi_prod_mat, 0).t();


    for(uword l = 0; l < L; l++) {
      w_l = w.col(l);

      weight_products = weight_products % w_l.elem(comb_inds.col(l));
    }

    // The rate for the gammas
    Z = accu(weight_products % phi_products);
  };

  void sampleStrategicLatentVariable() {
    v = arma::randg(arma::distr_param(N, 1.0 / Z));
  }

  void sampleFromPriors() {
    phis = randg(LC2, distr_param(2.0 , 1.0 / 2));

    for(arma::uword l = 0; l < L; l++) {
      for(uword k = 0; k < K(l); k++) {
        w(k, l) = randg(distr_param(1.0 / (double)K(l), 1.0));
      }
    }
  };
  
  mat calculateUpweights(uword l) {
    uvec matching_labels(N);
    mat upweights(N, K(l));
    upweights.zeros();
    
    for(uword m = 0; m < L; m++) {
      
      if(m != l){
        
        for(uword k = 0; k < K(l); k++) {
          
          matching_labels = (labels.col(m) == k);
          
          // Recall that the map assumes l < m; so account for that
          if(l < m) {
            upweights.col(k) = phis(phi_ind_map(m, l)) * conv_to<vec>::from(matching_labels);
          } else {
            upweights.col(k) = phis(phi_ind_map(l, m)) * conv_to<vec>::from(matching_labels);
          }
        }
      }
    }
    
    upweights++;
    
    return upweights;
  };
  
  void initialiseMDI() {
    // uvec matching_labels(N);
    mat upweights;
    
    sampleFromPriors();
    initialiseMixtures();
    
    for(uword l = 0; l < L; l++) {
      upweights = calculateUpweights(l);
    
      
      // Update the allocation within the mixture using MDI level weights and phis
      // mixtures[l].updateAllocation(w(span(0, K(l) - 1), l), upweigths.t());
      // (*mixtures)[l]->updateAllocation(w(span(0, K(l) - 1), l), upweigths.t());
      mixtures[l]->initialiseMixture(w(span(0, K(l) - 1), l), upweights.t());
      
      labels.col(l) = mixtures[l]->labels;
      non_outliers.col(l) = mixtures[l]->non_outliers;
      
    }
    
  };

  void updateAllocation() {

    // uvec matching_labels(N);
    mat upweights; // (N, K_max);

    // throw std::invalid_argument( "in MDI allocation." );
    
    for(uword l = 0; l < L; l++) {
      upweights = calculateUpweights(l);
      
      // upweights.set_size(N, K(l));
      // upweights.zeros();
      // 
      // for(uword m = 0; m < L; m++) {
      //   
      //   
      //   if(m != l){
      //     Rcpp::Rcout << "\n\nDo we enter this loop when L = 1?\n";
      //     
      //     
      //     for(uword k = 0; k < K(l); k++) {
      // 
      //       matching_labels = (labels.col(m) == k);
      // 
      //       // Recall that the map assumes l < m; so account for that
      //       if(l < m) {
      //         upweights.col(k) = phis(phi_ind_map(m, l)) * conv_to<vec>::from(matching_labels);
      //       } else {
      //         upweights.col(k) = phis(phi_ind_map(l, m)) * conv_to<vec>::from(matching_labels);
      //       }
      //     }
      //   }
      // }
      // 
      // upweights++;

      // throw std::invalid_argument( "in MDI allocation." );
      
      
      // std::cout << "\n\nUpdate allocations in mixture.\n";
      //
      // std::cout << "\n\nUpweights:\n" << upweigths;
      // std::cout << "\n\nWeights:\n" << w;
      // std::cout << "\n\nWeights in lth dataset:\n" << w(span(0, K(l) - 1), l);


      // Update the allocation within the mixture using MDI level weights and phis
      // mixtures[l].updateAllocation(w(span(0, K(l) - 1), l), upweigths.t());
      // (*mixtures)[l]->updateAllocation(w(span(0, K(l) - 1), l), upweigths.t());
      mixtures[l]->updateAllocation(w(span(0, K(l) - 1), l), upweights.t());
      
      // Pass the new labels from the mixture level back to the MDI level.
      // labels.col(l) = mixtures[l].labels;
      // non_outliers.col(l) = mixtures[l].non_outliers;

      // labels.col(l) = (*mixtures)[l]->labels;
      // non_outliers.col(l) = (*mixtures)[l]->non_outliers;
      
      // std::cout << "\n\nPass from mixtures back to MDI level.\n";
      
      labels.col(l) = mixtures[l]->labels;
      non_outliers.col(l) = mixtures[l]->non_outliers;
      

    }
  };

  
  // This is used to consider possible label swaps
  double sampleLabel(arma::uword k, arma::vec K_inv_cum) {
    
    // Need to account for the fixed labels
    // Need the non-fixed classes (e.g., we now need, K, K_fixed and K_unfixed)
    // uvec fixed_classes = unique(labels(fixed_ind));
    // uword K_fixed = length(fixed_classes)
    
    
    // Select another label randomly
    double u = randu();
    uword k_prime = sum(u > K_inv_cum);
    
    // If it is >= than the current label under consideration, add one
    if(k_prime >= k) {
      k_prime++;
    }
    return k_prime;
  }
  
  double calcScore(arma::uword l, arma::umat labels) {
    
    bool not_current_context = true;
    double score = 0.0;
    uvec agreeing_labels;
    
    for(uword m = 0; m < L; m++) {
      
      // Skip the current context (the lth context)
      not_current_context = m != l;
      if(not_current_context) {
        
        // Find which labels agree between datasets
        agreeing_labels = 1 * (labels.col(m) == labels.col(l));
        
        // Update the score based on the phi's
        score += accu(log(1 + phis(phi_ind_map(m, l)) * agreeing_labels));
      }
    }
    
    // // Find which labels match in the other contexts
    // umat matching_labels(N, L - 1);
    // matching_labels = dummy_labels.each_col([this, l](uvec i)
    // {
    //   return 1 * (i == labels.col(l));
    // }); 
    
    return score;
  }
  
  // Check if labels should be swapped to improve correlation of clustering
  // across datasets via random sampling.
  arma::umat swapLabels(arma::uword l, arma::uword k, arma::uword k_prime) {
    
    // The labels in the current context, which will be changed
    uvec loc_labs = labels.col(l), 
      
      // The indices for the clusters labelled k and k prime
      cluster_k,
      cluster_k_prime;
    
    // The labels for the other contexts
    umat dummy_labels = labels;
    
    // Find which indices are to be swapped
    cluster_k = find(loc_labs == k);
    cluster_k_prime = find(loc_labs == k_prime);
    
    // Swap the label associated with the two clusters
    loc_labs.elem(cluster_k).fill(k_prime);
    loc_labs.elem(cluster_k_prime).fill(k);
    
    dummy_labels.col(l) = loc_labs;
    return dummy_labels;
  }
  
  // Check if labels should be swapped to improve correlation of clustering
  // across datasets via random sampling.
  void updateLabels() {
    
    bool multipleUnfixedComponents = true;
    
    // The other component considered
    uword k_prime = 0;
    
    // Random uniform number
    double u = 0.0,
      
      // The current likelihood
      curr_score = 0.0,
      
      // The competitor
      alt_score = 0.0,
      
      // The accpetance probability
      accept = 1.0,
      log_accept = 0.0,
      
      // The weight of the kth component if we do accept the swap
      old_weight = 0.0;

      // Vector of entries equal to 1/(K - 1) (as we exclude the current label) and
      // its cumulative sum, used to sample another label to consider swapping.
      vec K_inv, K_inv_cum;
      
      umat swapped_labels(N, L);
      
      for(uword l = 0; l < L; l++) {
    
        multipleUnfixedComponents = (K_unfixed(l) > 1);
        if(multipleUnfixedComponents) {
          // K_inv = ones<vec>(K(l) - 1) * 1 / (K(l) - 1);
          K_inv = ones<vec>(K_unfixed(l) - 1) * 1 / (K_unfixed(l) - 1);
          K_inv_cum = cumsum(K_inv);
          
          // The score associated with the current labelling
          curr_score = calcScore(l, labels);
          
          for(uword k = K_fixed(l); k < K(l); k++) {
            
            // Select another label randomly
            k_prime = sampleLabel(k, K_inv_cum) + K_fixed(l);
            
            // The label matrix updated with the swapped labels
            swapped_labels = swapLabels(l, k, k_prime);
            
            // The score for the swap
            alt_score = calcScore(l, swapped_labels);
            
            // The log acceptance probability
            log_accept = alt_score - curr_score;
            
            if(log_accept < 0) {
              accept = std::exp(log_accept);
            }
            
            // If we accept the label swap, update labels, weights and score
            if(randu() < accept) {
              
              // Update the current score
              curr_score = alt_score;
              labels = swapped_labels;
              
              // Update the component weights
              old_weight = w(k, l);
              w(k, l) = w(k_prime, l);
              w(k_prime, l) = old_weight;
              
            } 
            
          } 
        }
      }
    }
  
  
  // double updateWeightHyperParameter(arma::uword k) {
  //
  //   // Initialise b, the rate
  //   double b = 0.0;
  //
  //   if(k < n_clust){
  //     b = v * (arma::sum(cluster_weights) + phi * cluster_weights(cluster_index));
  //   } else {
  //     b = v * (arma::sum(cluster_weights));
  //   }
  //   return b;
  // }
  //
  // // Calculate the rate for the gamma distribution for the class weights for MDI
  // // This is defined as the sume of the cluster weights (upweighted by the
  // // correlation parameter, phi, if the labels match) multiplied by the variable v
  // // Old name: mdi_phi_rate
  // double calcPhiRate(double v,
  //                    arma::uword n_clust,
  //                    arma::vec cluster_weights_1,
  //                    arma::vec cluster_weights_2
  // ) {
  //   // Initialise b, the rate
  //   double b = 0.0;
  //
  //   // Find the subset of weights to use in calculating b
  //   arma::vec sub_weights_1(n_clust);
  //   arma::vec sub_weights_2(n_clust);
  //
  //   sub_weights_1 = cluster_weights_1(arma::span(0, n_clust - 1) );
  //   sub_weights_2 = cluster_weights_2(arma::span(0, n_clust - 1) );
  //
  //   // Calculate b
  //   b = v * sum(sub_weights_1 % sub_weights_2);
  //
  //   return b;
  //
  // }



  // // Calculate the weights for each Gamma distribution ohi may be sampled from
  // // Old name: phi_weights
  // arma::vec calcMDIPhiWeights(arma::uword n,
  //                             double a_0,
  //                             double b_n){
  //   arma::vec phi_weights(n);
  //   phi_weights.zeros();
  //
  //   for(arma::uword i = 0; i < n; i++){
  //     // this is the weight of which gamma to sample for the phi
  //     phi_weights(i) = logFactorial(n)
  //     - logFactorial(i)
  //     - logFactorial(n - i)
  //     + logFactorial(i + a_0 - 1)
  //     - (i + a_0)*log(b_n);
  //   }
  //   return phi_weights;
  // }
  //
  // // samples a gamma distn for the current iterations context similarity parameter
  // // (phi in the original 2012 paper).
  // // Old name: sample_phi
  // double sampleMDIPhi(arma::uvec cl_1,
  //                     arma::uvec cl_2,
  //                     arma::vec cl_wgts_1,
  //                     arma::vec cl_wgts_2,
  //                     double v,
  //                     arma::uword n,
  //                     arma::uword min_n_clust,
  //                     double a_0,
  //                     double b_0
  // ) {
  //
  //   // The predicted index of the weighted sum to use
  //   arma::uword count_same_cluster = 0;
  //   arma::uword pred_ind = 0;
  //   double b = 0.0; // the rate
  //   double phi = 0.0; //output, the clustering correlation parameter (phi)
  //   arma::vec prob_vec(count_same_cluster);
  //
  //   // calculate the shape of the relevant gamma function (this is the count of
  //   // the points with a common label across contexts)
  //   count_same_cluster = countCommonLabel(cl_1, cl_2, n);
  //   prob_vec.zeros();
  //
  //   // calculate the rate
  //   b = calcRateMDIPhi(v, min_n_clust, cl_wgts_1, cl_wgts_2) + b_0;
  //
  //   // phi is a weighted sum of gammas; see section 1.5 and 1.51 from:
  //   // https://github.com/stcolema/tagmmdi_notes/blob/master/notes/mdi_olly.pdf
  //   if(count_same_cluster > 0){
  //
  //     // Calculated the weight for each Gamma distribution
  //     prob_vec = calcMDIPhiWeights(count_same_cluster, a_0, b);
  //
  //     // Predict which to use based on prob_vec
  //     pred_ind = predictCluster(prob_vec);
  //
  //     // Sample phi
  //     phi = arma::randg( arma::distr_param(pred_ind + a_0, 1.0/b) );
  //
  //   } else {
  //     // Sample from the prior
  //     phi = arma::randg( arma::distr_param(a_0, 1.0/b) );
  //   }
  //
  //   return phi;
  // }
  //
  //
  //
  //
  //
  // // Sample the cluster membership of a categorical sample for MDI
  // // Old name: mdi_cat_clust_prob
  // arma::vec sampleMDICatClustProb(arma::uword row_index,
  //                                 arma::umat data,
  //                                 arma::field<arma::mat> class_probs,
  //                                 arma::uword num_clusters,
  //                                 arma::uword n_col,
  //                                 double phi,
  //                                 arma::vec cluster_weights,
  //                                 arma::uvec clust_labels,
  //                                 arma::uvec clust_labels_comp
  // ) {
  //
  //   // cluster_labels_comparison is the labels of the data in the other context
  //   arma::uword common_cluster = 0;
  //   double curr_weight = 0.0;
  //   double similarity_upweight = 0.0; // Upweight for similarity of contexts
  //   arma::urowvec point = data.row(row_index);
  //   arma::vec prob_vec = arma::zeros<arma::vec>(num_clusters);
  //
  //   // std::cout << "\nCluster weights:\n" << cluster_weights.t() << "\n";
  //   // std::cout << "\nK:\n" << num_clusters << "\n";
  //
  //   for(arma::uword i = 0; i < num_clusters; i++){
  //
  //     // std::cout << "In loop: " << i << "\n";
  //
  //     // calculate the log-weights for the context specific cluster and the across
  //     // context similarity
  //     // pretty much this is the product of probabilities possibly up-weighted by
  //     // being in the same cluster in a different context and weighted by the cluster
  //     // weight in the current context
  //     curr_weight = log(cluster_weights(i));
  //
  //     // std::cout << "\nWeight calculated\n";
  //
  //     // Check if in the same cluster in both contexts
  //     common_cluster = 1 * (clust_labels_comp(row_index) == clust_labels(row_index));
  //
  //     // std::cout << "\nIndicator funciton done.\n";
  //
  //     similarity_upweight = log(1 + phi * common_cluster);
  //
  //     // std::cout << "\nUpweighed.\n";
  //
  //     for(arma::uword j = 0; j < n_col; j++){
  //
  //       // std::cout << "\nLoop over columns: " << j << "\n";
  //       prob_vec(i) = prob_vec(i) + std::log(class_probs(j)(i, point(j)));
  //
  //     }
  //
  //     // std::cout << "\nOut of loop, final calculation.\n";
  //
  //     // As logs can sum rather than multiply the components
  //     prob_vec(i) = curr_weight + prob_vec(i) + similarity_upweight;
  //   }
  //
  //   // std::cout << "\nFinished.\n";
  //
  //   // // to handle overflowing
  //   // prob_vec = exp(prob_vec - max(prob_vec));
  //   //
  //   // // normalise
  //   // prob_vec = prob_vec / sum(prob_vec);
  //
  //   return prob_vec;
  // }
  //
  //
  //
  // // Old name: mdi_cluster_weights
  // //' Returns the cluster weights for MDI.
  // //' @param shape_0 The prior on the shape for the cluster weights.
  // //' @param rate_0 The prior on the rate for the cluster weights.
  // //' @param v The strategic latent variable (as in Nieto-Barajas et al., 2004) to
  // //' ensure that the posterior of most of the model parameters are Gamma
  // //' distributions.
  // //' @param n_clust The number of clusters present in the current dataset.
  // //' @param n_clust_comp The number of clusters present in the other dataset.
  // //' @param cluster_weights_comp The mixture weights for the other dataset.
  // //' @param cluster_labels The membership vector for the clustering in the
  // //' current dataset.
  // //' @param cluster_labels_comp The membership vector for the clustering in the
  // //' other dataset.
  // //' @param phi The context similarity parameter from MDI; this is a measure of
  // //' the similarity of clustering between datasets.
  // //'
  // //' @return A vector of mixture weights.
  // arma::vec sampleMDIClusterWeights(arma::vec shape_0,
  //                                   arma::vec rate_0,
  //                                   double v,
  //                                   arma::uword n_clust,
  //                                   arma::uword n_clust_comp,
  //                                   arma::vec cluster_weights_comp,
  //                                   arma::uvec cluster_labels,
  //                                   arma::uvec cluster_labels_comp,
  //                                   double phi){
  //
  //   // The number of clusters relevant to MDI is the minimum of the number of
  //   // clusters present in either dataset
  //   // Initialise the cluster weights, the rate and shape
  //   arma::uword n_rel = std::min(n_clust, n_clust_comp);
  //   arma::uword r_class_start = 1;
  //   double b = 0.0;
  //   double b_n = 0.0;
  //   arma::vec cluster_weight = arma::zeros<arma::vec>(n_clust);
  //   arma::vec shape_n = arma::zeros<arma::vec>(n_rel);
  //
  //   // Calculate the concentration parameter for each cluster
  //   shape_n = updateConcentration(shape_0,
  //                                 cluster_labels,
  //                                 n_clust,
  //                                 r_class_start) + 1;
  //
  //   for (arma::uword i = 0; i < n_clust; i++) {
  //
  //     // Calculate the rate based upon the current clustering
  //     b = calcRateMDIClassWeights(v,
  //                                 n_rel,
  //                                 i,
  //                                 cluster_weights_comp,
  //                                 phi);
  //
  //     // Update the prior
  //     b_n = b + rate_0(i);
  //
  //     // Sample the weights from a gamma distribution
  //     cluster_weight(i) = arma::randg(arma::distr_param(shape_n(i), 1 / b_n));
  //
  //   }
  //   return cluster_weight;
  // }
  //
  //
  //
  //
};

// class semisupervisedMixture : public virtual mixture {
// private:
//
// public:
//
//   arma::uword N_fixed = 0;
//   arma::uvec fixed, unfixed_ind;
//   arma::mat alloc_prob;
//
//   using mixture::mixture;
//
//   semisupervisedMixture(
//     arma::uword _K,
//     arma::uvec _labels,
//     arma::vec _concentration,
//     arma::mat _X,
//     arma::uvec _fixed
//   ) :
//     mixture(_K, _labels, _concentration, _X) {
//
//     arma::uvec fixed_ind(N);
//
//     fixed = _fixed;
//     N_fixed = arma::sum(fixed);
//     fixed_ind = arma::find(_fixed == 1);
//     unfixed_ind = find(fixed == 0);
//
//     alloc_prob.set_size(N, K);
//     alloc_prob.zeros();
//
//     for (auto& n : fixed_ind) {
//       alloc_prob(n, labels(n)) = 1.0;
//     }
//   };
//
//   // Destructor
//   virtual ~semisupervisedMixture() { };
//
//   virtual void updateAllocation() {
//
//     double u = 0.0;
//     arma::uvec uniqueK;
//     arma::vec comp_prob(K);
//
//     for (auto& n : unfixed_ind) {
//
//       ll = itemLogLikelihood(X_t.col(n));
//
//       // Update with weights
//       comp_prob = ll + log(w);
//
//       likelihood(n) = arma::accu(comp_prob);
//
//       // Normalise and overflow
//       comp_prob = exp(comp_prob - max(comp_prob));
//       comp_prob = comp_prob / sum(comp_prob);
//
//       // Save the allocation probabilities
//       alloc_prob.row(n) = comp_prob.t();
//
//       // Prediction and update
//       u = arma::randu<double>( );
//
//       labels(n) = sum(u > cumsum(comp_prob));
//       alloc.row(n) = comp_prob.t();
//
//       // model_likelihood_alt += ll(labels(n));
//
//       // Record the log likelihood of the item in it's allocated component
//       // likelihood(n) = ll(labels(n));
//     }
//
//     // The model log likelihood
//     model_likelihood = arma::accu(likelihood);
//
//     // Number of occupied components (used in BIC calculation)
//     uniqueK = arma::unique(labels);
//     K_occ = uniqueK.n_elem;
//   };
//
// };


// class mvnPredictive : public mvnSampler, public semisupervisedSampler {
//
// private:
//
// public:
//
//   using mvnSampler::mvnSampler;
//
//   mvnPredictive(
//     arma::uword _K,
//     arma::uword _B,
//     double _mu_proposal_window,
//     double _cov_proposal_window,
//     double _m_proposal_window,
//     double _S_proposal_window,
//     double _rho,
//     double _theta,
//     double _lambda,
//     arma::uvec _labels,
//     arma::uvec _batch_vec,
//     arma::vec _concentration,
//     arma::mat _X,
//     arma::uvec _fixed
//   ) :
//     sampler(_K, _B, _labels, _batch_vec, _concentration, _X),
//     mvnSampler(_K,
//                _B,
//                _mu_proposal_window,
//                _cov_proposal_window,
//                _m_proposal_window,
//                _S_proposal_window,
//                _rho,
//                _theta,
//                _lambda,
//                _labels,
//                _batch_vec,
//                _concentration,
//                _X),
//                semisupervisedSampler(_K, _B, _labels, _batch_vec, _concentration, _X, _fixed)
//   {
//   };
//
//   virtual ~mvnPredictive() { };
//
//   // virtual void sampleFromPriors() {
//   //
//   //   arma::mat X_k;
//   //
//   //   for(arma::uword k = 0; k < K; k++){
//   //     X_k = X.rows(arma::find(labels == k && fixed == 1));
//   //     cov.slice(k) = arma::diagmat(arma::stddev(X_k).t());
//   //     mu.col(k) = arma::mean(X_k).t();
//   //   }
//   //   for(arma::uword b = 0; b < B; b++){
//   //     for(arma::uword p = 0; p < P; p++){
//   //
//   //       // Fix the 0th batch at no effect; all other batches have an effect
//   //       // relative to this
//   //       // if(b == 0){
//   //       S(p, b) = 1.0;
//   //       m(p, b) = 0.0;
//   //       // } else {
//   //       // S(p, b) = 1.0 / arma::randg<double>( arma::distr_param(rho, 1.0 / theta ) );
//   //       // m(p, b) = arma::randn<double>() * S(p, b) / lambda + delta(p);
//   //       // }
//   //     }
//   //   }
//   // };
//
// };
//
//
//
// class mvtSampler: virtual public mvnSampler {
//
// public:
//
//   // arma::uword t_df = 4;
//   arma::uword n_param_cluster = 2 + P + P * (P + 1) * 0.5, n_param_batch = 2 * P;
//   double psi = 2.0, chi = 0.01, t_df_proposal_window = 0.0, pdf_const = 0.0, t_loc = 2.0;
//   arma::uvec t_df_count;
//   arma::vec t_df, pdf_coef;
//
//
//   using mvnSampler::mvnSampler;
//
//   mvtSampler(
//     arma::uword _K,
//     arma::uword _B,
//     double _mu_proposal_window,
//     double _cov_proposal_window,
//     double _m_proposal_window,
//     double _S_proposal_window,
//     double _t_df_proposal_window,
//     double _rho,
//     double _theta,
//     double _lambda,
//     arma::uvec _labels,
//     arma::uvec _batch_vec,
//     arma::vec _concentration,
//     arma::mat _X
//   ) : sampler(_K,
//   _B,
//   _labels,
//   _batch_vec,
//   _concentration,
//   _X), mvnSampler(
//       _K,
//       _B,
//       _mu_proposal_window,
//       _cov_proposal_window,
//       _m_proposal_window,
//       _S_proposal_window,
//       _rho,
//       _theta,
//       _lambda,
//       _labels,
//       _batch_vec,
//       _concentration,
//       _X
//   ) {
//
//     // Hyperparameter for the d.o.f for the t-distn
//     // psi = 0.5;
//     // chi = 0.5;
//
//     t_df.set_size(K);
//     t_df.zeros();
//
//     pdf_coef.set_size(K);
//     pdf_coef.zeros();
//
//     t_df_count.set_size(K);
//     t_df_count.zeros();
//
//     // The shape of the skew normal
//     // phi.set_size(P, K);
//     // phi.zeros();
//
//     // Count the number of times proposed values are accepted
//     // phi_count = arma::zeros<arma::uvec>(K);
//
//     // The proposal windows for the cluster and batch parameters
//     t_df_proposal_window = _t_df_proposal_window;
//
//     // The constant for the item likelihood (changes if t_df != const)
//     // pdf_const = logGamma(0.5 * (t_df + P)) - logGamma(0.5 * t_df) - 0.5 * P * log(t_df);
//   };
//
//
//   // Destructor
//   virtual ~mvtSampler() { };
//
//   // Print the sampler type.
//   virtual void printType() {
//     std::cout << "\nType: Multivariate T.\n";
//   };
//
//   double calcPDFCoef(double t_df){
//     double x = logGamma(0.5 * (t_df + P)) - logGamma(0.5 * t_df) - 0.5 * P * log(t_df);
//     return x;
//   };
//
//   virtual void sampleDFPrior() {
//     for(arma::uword k = 0; k < K; k++){
//       // Draw from a shifted gamma distribution (i.e. gamma with location parameter)
//       t_df(k) = t_loc + arma::randg<double>( arma::distr_param(psi, 1.0 / chi));
//     }
//   };
//
//   virtual void sampleFromPriors() {
//
//     sampleCovPrior();
//     sampleMuPrior();
//     sampleDFPrior();
//     sampleSPrior();
//     sampleMPrior();
//   };
//
//   // virtual void sampleFromPriors() {
//   //
//   //   for(arma::uword k = 0; k < K; k++){
//   //     cov.slice(k) = arma::iwishrnd(scale, nu);
//   //     mu.col(k) = arma::mvnrnd(xi, (1.0/kappa) * cov.slice(k), 1);
//   //
//   //     // Draw from a shifted gamma distribution (i.e. gamma with location parameter)
//   //     t_df(k) = t_loc + arma::randg<double>( arma::distr_param(psi, 1.0 / chi));
//   //   }
//   //   for(arma::uword b = 0; b < B; b++){
//   //     for(arma::uword p = 0; p < P; p++){
//   //       S(p, b) = S_loc + 1.0 / arma::randg<double>( arma::distr_param(rho, 1.0 / theta ) );
//   //       m(p, b) = arma::randn<double>() * S(p, b) / lambda + delta(p);
//   //     }
//   //   }
//   // };
//
//   // Update the common matrix manipulations to avoid recalculating N times
//   virtual void matrixCombinations() {
//
//     for(arma::uword k = 0; k < K; k++) {
//       pdf_coef(k) = calcPDFCoef(t_df(k));
//       cov_inv.slice(k) = arma::inv_sympd(cov.slice(k));
//       cov_log_det(k) = arma::log_det(cov.slice(k)).real();
//
//       for(arma::uword b = 0; b < B; b++) {
//         cov_comb.slice(k * B + b) = cov.slice(k); // + arma::diagmat(S.col(b))
//         for(arma::uword p = 0; p < P; p++) {
//           cov_comb.slice(k * B + b)(p, p) *= S(p, b);
//         }
//         cov_comb_log_det(k, b) = arma::log_det(cov_comb.slice(k * B + b)).real();
//         cov_comb_inv.slice(k * B + b) = arma::inv_sympd(cov_comb.slice(k * B + b));
//
//         mean_sum.col(k * B + b) = mu.col(k) + m.col(b);
//       }
//     }
//   };
//
//
//   // The log likelihood of a item belonging to each cluster given the batch label.
//   arma::vec itemLogLikelihood(arma::vec item, arma::uword b) {
//
//     double x = 0.0, y = 0.0, my_det = 0.0;
//     arma::vec ll(K), dist_to_mean(P);
//     ll.zeros();
//     dist_to_mean.zeros();
//     arma::mat my_cov_comv_inv(P, P), my_inv(P, P), my_cov_comb(P, P);
//     my_cov_comv_inv.zeros();
//     my_inv.zeros();
//     my_cov_comb.zeros();
//
//     double cov_correction = 0.0;
//
//     for(arma::uword k = 0; k < K; k++){
//
//       // gamma(0.5 * (nu + P)) / (gamma(0.5 * nu) * nu ^ (0.5 * P) * pi ^ (0.5 * P)  * det(cov) ^ 0.5) * (1 + (1 / nu) * (x - mu)^t * inv(cov) * (x - mu)) ^ (-0.5 * (nu + P))
//       // logGamma(0.5 * (nu + P)) - logGamma(0.5 * nu) - (0.5 * P) * log(nu) - 0.5 * P * log(pi) - 0.5 * logDet(cov) -0.5 * (nu + P) * log(1 + (1 / nu) * (x - mu)^t * inv(cov) * (x - mu))
//
//       // my_cov_comv_inv = cov.slice(k);
//       // for(arma::uword p = 0; p < P; p++) {
//       //   my_cov_comv_inv(p, p) *= S(p, b);
//       // }
//
//       // cov_correction = t_df(k) / (t_df(k) - 2.0);
//
//
//       // my_cov_comb = cov.slice(k);
//       //
//       // for(arma::uword p = 0; p < P; p++) {
//       //   my_cov_comb(p, p) = my_cov_comb(p, p) * S(p, b);
//       // }
//       //
//       // // my_cov_comb = my_cov_comb / cov_correction;
//       //
//       // // std::cout << "\nThe invariance.";
//       //
//       // my_inv = arma::inv_sympd(my_cov_comb);
//       //
//       // // std::cout << "\nDeterminant.";
//       // my_det = arma::log_det(my_cov_comb).real();
//
//       // The exponent part of the MVN pdf
//       dist_to_mean = item - mean_sum.col(k * B + b);
//       x = arma::as_scalar(dist_to_mean.t() * cov_comb_inv.slice(k * B + b) * dist_to_mean);
//       // x = arma::as_scalar(dist_to_mean.t() * my_inv * dist_to_mean);
//       y = (t_df(k) + P) * log(1.0 + (1/t_df(k)) * x);
//
//       ll(k) = pdf_coef(k) - 0.5 * (cov_comb_log_det(k, b) + y + P * log(PI));
//       // ll(k) = pdf_coef(k) - 0.5 * (my_det + y + P * log(PI));
//
//       // std::cout << "\nCheck.";
//
//       // if(! arma::approx_equal(mean_sum.col(k * B + b), (mu.col(k) + m.col(b)), "absdiff", 0.001)) {
//       //   std::cout << "\n\nMean sum has deviated from expected.";
//       // }
//       //
//       // if(! arma::approx_equal(cov_comb_inv.slice(k * B + b), my_inv, "absdiff", 0.001)) {
//       //   std::cout << "\n\nCovariance inverse has deviated from expected.";
//       //   std::cout << "\n\nExpected:\n" << cov_comb_inv.slice(k * B + b) <<
//       //     "\n\nCalculated:\n" << my_inv;
//       //
//       //   throw std::invalid_argument( "\nMy inverses diverged." );
//       // }
//       //
//       // if(isnan(ll(k))) {
//       //   std::cout << "\nNaN!\n";
//       //
//       //   double new_x = (1/t_df(k)) * arma::as_scalar((item - mu.col(k) - m.col(b)).t() * my_inv * (item - mu.col(k) - m.col(b)));
//       //
//       //   std::cout << "\n\nItem likelihood:\n" << ll(k) <<
//       //     "\nPDF coefficient: " << pdf_coef(k) << "\nLog determinant: " <<
//       //       cov_comb_log_det(k, b) << "\nX: " << x << "\nY: " << y <<
//       //         "\nLog comp of y: " << 1.0 + (1/t_df(k)) * x <<
//       //           "\nLogged: " << log(1.0 + (1/t_df(k)) * x) <<
//       //             "\nt_df(k): " << t_df(k) << "\n" <<
//       //               "\nMy new x" << new_x << "\nLL alt: " <<
//       //                 pdf_coef(k) - 0.5 * (my_det + (t_df(k) + P) * log(1.0 + new_x) + P * log(PI)) <<
//       //                   "\n\nCov combined expected:\n" << cov_comb_inv.slice(k * B + b) <<
//       //                     "\n\nCov combined real:\n" << my_inv;
//       //
//       //   throw std::invalid_argument( "\nNaN returned from likelihood." );
//       //
//       // }
//
//
//     }
//
//     return(ll);
//   };
//
//   void calcBIC(){
//
//     // Each component has a weight, a mean vector, a symmetric covariance matrix and a
//     // degree of freedom parameter. Each batch has a mean and standard
//     // deviations vector.
//     // arma::uword n_param = (P + P * (P + 1) * 0.5 + 1) * K_occ + (2 * P) * B;
//     // BIC = n_param * std::log(N) - 2 * model_likelihood;
//
//     // arma::uword n_param_cluster = 2 + P + P * (P + 1) * 0.5;
//     // arma::uword n_param_batch = 2 * P;
//
//     BIC = 2 * model_likelihood - (n_param_batch + n_param_batch) * std::log(N);
//
//     // for(arma::uword k = 0; k < K; k++) {
//     //   BIC -= n_param_cluster * std::log(N_k(k)+ 1);
//     // }
//     // for(arma::uword b = 0; b < B; b++) {
//     //   BIC -= n_param_batch * std::log(N_b(b)+ 1);
//     // }
//
//   };
//
//   double clusterLikelihood(
//       double t_df,
//       arma::uvec cluster_ind,
//       arma::vec cov_det,
//       arma::mat mean_sum,
//       arma::cube cov_inv
//   ) {
//
//     arma::uword b = 0;
//     double score = 0.0;
//     arma::vec dist_from_mean(P);
//
//     for (auto& n : cluster_ind) {
//       b = batch_vec(n);
//       dist_from_mean = X_t.col(n) - mean_sum.col(b);
//       score += cov_det(b) + (t_df + P) * log(1 + (1/t_df) * arma::as_scalar(dist_from_mean.t() * cov_inv.slice(b) * dist_from_mean));
//     }
//     //
//     // std::cout << "\nScore before halving: " << score << "\nT DF: " << t_df <<
//     //   "\n\nCov log det:\n" << cov_det << "\n\nCov inverse:\n " << cov_inv;
//     //
//     //
//
//     return (-0.5 * score);
//   }
//
//   double batchLikelihood(
//       arma::uvec batch_inds,
//       arma::uvec labels,
//       arma::vec cov_det,
//       arma::vec t_df,
//       arma::mat mean_sum,
//       arma::cube cov_inv){
//
//     arma::uword k = 0;
//     double score = 0.0;
//     arma::vec dist_from_mean(P);
//
//     for (auto& n : batch_inds) {
//       k = labels(n);
//       dist_from_mean = X_t.col(n) - mean_sum.col(k);
//       score += cov_det(k) + (t_df(k) + P) * log(1 + (1/t_df(k)) * arma::as_scalar(dist_from_mean.t() * cov_inv.slice(k) * dist_from_mean));
//     }
//     return (-0.5 * score);
//   }
//
//   double mLogKernel(arma::uword b, arma::vec m_b, arma::mat mean_sum) {
//
//     arma::uword k = 0;
//     double score = 0.0, score_alt = 0.0;
//     arma::vec dist_from_mean(P);
//     dist_from_mean.zeros();
//
//     score = batchLikelihood(batch_ind(b),
//                             labels,
//                             cov_comb_log_det.col(b),
//                             t_df,
//                             mean_sum,
//                             cov_comb_inv.slices(KB_inds + b)
//     );
//
//     // for (auto& n : batch_ind(b)) {
//     //   k = labels(n);
//     //   dist_from_mean = X_t.col(n) - mean_sum.col(k);
//     //   score_alt += cov_comb_log_det(k, b) + (t_df(k) + P) * log(1 + (1/t_df(k)) * arma::as_scalar(dist_from_mean.t() * cov_comb_inv.slice(k * B + b) * dist_from_mean));
//     // }
//     //
//     // score_alt *= -0.5;
//     //
//     // if(std::abs(score_alt - score) > 1e-6) {
//     //   std::cout << "\nProblem in m kernel function.\nOld score: " <<
//     //     score << "\nAlternative score: " << score_alt;
//     //   throw std::invalid_argument( "\n" );
//     // }
//
//     for(arma::uword p = 0; p < P; p++) {
//       score += -0.5 * lambda * std::pow(m_b(p) - delta(p), 2.0) / S(p, b);
//     }
//
//     // score *= -0.5;
//
//     return score;
//   };
//
//   double sLogKernel(arma::uword b,
//                     arma::vec S_b,
//                     arma::vec cov_comb_log_det,
//                     arma::cube cov_comb_inv) {
//
//     arma::uword k = 0;
//     double score = 0.0, score_alt = 0.0;
//     arma::vec dist_from_mean(P);
//     dist_from_mean.zeros();
//     arma::mat curr_sum(P, P);
//
//     score = batchLikelihood(batch_ind(b),
//                             labels,
//                             cov_comb_log_det,
//                             t_df,
//                             mean_sum.cols(KB_inds + b),
//                             cov_comb_inv
//     );
//
//     // for (auto& n : batch_ind(b)) {
//     //   k = labels(n);
//     //   dist_from_mean = X_t.col(n) - mean_sum.col(k * B + b);
//     //   score_alt += (cov_comb_log_det(k) + (t_df(k) + P) * log(1 + (1/t_df(k)) * arma::as_scalar(dist_from_mean.t() * cov_comb_inv.slice(k) * dist_from_mean)));
//     // }
//     //
//     // score_alt *= -0.5;
//     //
//     // if(std::abs(score_alt - score) > 1e-6) {
//     //   std::cout << "\nProblem in S kernel function.\nOld score: " <<
//     //     score << "\nAlternative score: " << score_alt;
//     //   throw std::invalid_argument( "\n" );
//     // }
//
//     for(arma::uword p = 0; p < P; p++) {
//       score += -0.5 * ((2 * rho + 3) * std::log(S_b(p) - S_loc)
//                          + 2 * theta / (S_b(p) - S_loc)
//                          + lambda * std::pow(m(p,b) - delta(p), 2.0) / S_b(p));
//
//                          // score +=   (0.5 - 1) * std::log(S(p,b) - S_loc)
//                          //   - 0.5 * (S(p, b) - S_loc)
//                          //   - 0.5 * lambda * std::pow(m(p,b) - delta(p), 2.0) / S_b(p);
//     }
//
//     // score *= -0.5;
//     return score;
//   };
//
//   double muLogKernel(arma::uword k, arma::vec mu_k, arma::mat mean_sum) {
//
//     arma::uword b = 0;
//     double score = 0.0, score_alt = 0.0;
//     arma::uvec cluster_ind = arma::find(labels == k);
//     arma::vec dist_from_mean(P);
//
//     score = clusterLikelihood(
//       t_df(k),
//       cluster_ind,
//       cov_comb_log_det.row(k).t(),
//       mean_sum,
//       cov_comb_inv.slices(k * B + B_inds)
//     );
//
//     // for (auto& n : cluster_ind) {
//     //   b = batch_vec(n);
//     //   dist_from_mean = X_t.col(n) - mean_sum.col(b);
//     //   score_alt += cov_comb_log_det(k, b) +  (t_df(k) + P) * log(1 + (1/t_df(k)) * arma::as_scalar(dist_from_mean.t() * cov_comb_inv.slice(k * B + b) * dist_from_mean));
//     // }
//     //
//     // score_alt *= -0.5;
//     //
//     // if(std::abs(score_alt - score) > 1e-6) {
//     //   std::cout << "\nProblem in mu kernel function.\nOld score: " <<
//     //     score << "\nAlternative score: " << score_alt;
//     //   throw std::invalid_argument( "\n" );
//     // }
//
//     score += -0.5 * arma::as_scalar(kappa * ((mu_k - xi).t() *  cov_inv.slice(k) * (mu_k - xi)));
//     // score *= -0.5;
//
//     return score;
//   };
//
//   double covLogKernel(arma::uword k,
//                       arma::mat cov_k,
//                       double cov_log_det,
//                       arma::mat cov_inv,
//                       arma::vec cov_comb_log_det,
//                       arma::cube cov_comb_inv) {
//
//     arma::uword b = 0;
//     double score = 0.0, score_alt = 0.0;
//     arma::uvec cluster_ind = arma::find(labels == k);
//     arma::vec dist_from_mean(P);
//
//     score = clusterLikelihood(
//       t_df(k),
//       cluster_ind,
//       cov_comb_log_det,
//       mean_sum.cols(k * B + B_inds),
//       cov_comb_inv
//     );
//
//
//     // for (auto& n : cluster_ind) {
//     //   b = batch_vec(n);
//     //   dist_from_mean = X_t.col(n) - mean_sum.col(k * B + b);
//     //   score_alt += cov_comb_log_det(b) + (t_df(k) + P) * log(1 + (1/t_df(k)) * arma::as_scalar(dist_from_mean.t() * cov_comb_inv.slice(b) * dist_from_mean));
//     // }
//     //
//     // score_alt *= -0.5;
//     //
//     // if(std::abs(score_alt - score) > 1e-6) {
//     //   std::cout << "\nProblem in cov kernel function.\nOld score: " <<
//     //     score << "\nAlternative score: " << score_alt;
//     //
//     //   std::cout << "\nT DF: " << t_df(k) << "\n\nCov log det:\n" << cov_comb_log_det <<
//     //     "\n\nCov inverse:\n " << cov_comb_inv;
//     //
//     //   // std::cout << "\n\nMean sums:\n";
//     //   // for(arma::uword b = 0;b < B; b++){
//     //   //   std::cout << mean_sum.col(k * B + b) << "\n\n";
//     //   // }
//     //   // std::cout << "\n\nMean sums:\n" << mean_sum.cols(k * B + B_inds);
//     //   throw std::invalid_argument( "\n" );
//     // }
//
//     score += -0.5 *( arma::as_scalar((nu + P + 2) * cov_log_det
//                                        + kappa * ((mu.col(k) - xi).t() * cov_inv * (mu.col(k) - xi))
//                                        + arma::trace(scale * cov_inv)));
//                                        // score *= -0.5;
//
//                                        return score;
//   };
//
//   double dfLogKernel(arma::uword k,
//                      double t_df,
//                      double pdf_coef) {
//
//     arma::uword b = 0;
//     double score = 0.0;
//     arma::uvec cluster_ind = arma::find(labels == k);
//     arma::vec dist_from_mean(P);
//     for (auto& n : cluster_ind) {
//       b = batch_vec(n);
//       dist_from_mean = X_t.col(n) - mean_sum.col(k * B + b);
//       score += pdf_coef - 0.5 * (t_df + P) * log(1 + (1/t_df) * arma::as_scalar(dist_from_mean.t() * cov_comb_inv.slice(k * B + b) * dist_from_mean));
//     }
//     // score += (psi - 1) * log(t_df - t_loc) - (t_df - t_loc) / chi;
//     score += (psi - 1) * log(t_df - t_loc) - chi * (t_df - t_loc);
//     return score;
//   };
//
//   void clusterDFMetropolis() {
//     double u = 0.0, proposed_model_score = 0.0, acceptance_prob = 0.0, current_model_score = 0.0, t_df_proposed = 0.0, proposed_pdf_coef = 0.0;
//
//     for(arma::uword k = 0; k < K ; k++) {
//       proposed_model_score = 0.0, acceptance_prob = 0.0, current_model_score = 0.0, t_df_proposed = 0.0, proposed_pdf_coef = 0.0;
//       if(N_k(k) == 0){
//         t_df_proposed = t_loc + arma::randg<double>( arma::distr_param(psi, 1.0 / chi));
//         proposed_pdf_coef = calcPDFCoef(t_df_proposed);
//       } else {
//
//         // std::cout << "\n\nT df.\nPsi: " << psi << "\nChi: " << chi
//         // << "\nWindow: " << t_df_proposal_window << "\nCurrent: " << t_df(k);
//
//         t_df_proposed = t_loc + arma::randg( arma::distr_param( (t_df(k) - t_loc) * t_df_proposal_window, 1.0 / t_df_proposal_window) );
//
//         // t_df_proposed = t_loc + std::exp(arma::randn() * t_df_proposal_window + log(t_df(k) - t_loc) );
//
//         // proposed_model_score = logNormalLogProbability(t_df(k) - t_loc, t_df_proposed - t_loc, t_df_proposal_window);
//         // current_model_score = logNormalLogProbability(t_df_proposed - t_loc, t_df(k) - t_loc, t_df_proposal_window);
//         //
//         // std::cout  << "\nProposed score: " << proposed_model_score << "\nCurrent score: " << current_model_score;
//
//         // t_df_proposed = t_loc + std::exp((arma::randn() * t_df_proposal_window) + t_df(k) - t_loc);
//
//         // // Log probability under the proposal density
//         // proposed_model_score = logNormalLogProbability(t_df(k) - t_loc, (t_df_proposed - t_loc), t_df_proposal_window);
//         // current_model_score = logNormalLogProbability(t_df_proposed - t_loc, (t_df(k) - t_loc), t_df_proposal_window);
//
//         // Proposed value
//         // t_df_proposed = t_loc + arma::randg( arma::distr_param( (t_df(k) - t_loc) * t_df_proposal_window, 1.0 / t_df_proposal_window) );
//         proposed_pdf_coef = calcPDFCoef(t_df_proposed);
//
//         // std::cout << "\n\nDF: " << t_df(k) << "\nProposed DF: " << t_df_proposed;
//
//         // Asymmetric proposal density
//         proposed_model_score = gammaLogLikelihood(t_df(k) - t_loc, (t_df_proposed - t_loc) * t_df_proposal_window, t_df_proposal_window);
//         current_model_score = gammaLogLikelihood(t_df_proposed - t_loc, (t_df(k) - t_loc) * t_df_proposal_window, t_df_proposal_window);
//
//         // The prior is included in the kernel
//         proposed_model_score = dfLogKernel(k, t_df_proposed, proposed_pdf_coef);
//         current_model_score = dfLogKernel(k, t_df(k), pdf_coef(k));
//
//         u = arma::randu();
//         acceptance_prob = std::min(1.0, std::exp(proposed_model_score - current_model_score));
//
//       }
//
//       if((u < acceptance_prob) || (N_k(k) == 0)) {
//         t_df(k) = t_df_proposed;
//         t_df_count(k)++;
//         pdf_coef(k) = proposed_pdf_coef;
//       }
//     }
//   }
//
//   virtual void metropolisStep() {
//
//     // Metropolis step for cluster parameters
//     clusterCovarianceMetropolis();
//
//     // std::cout << "\n\nCluster covariance.";
//
//     // matrixCombinations();
//
//     clusterMeanMetropolis();
//
//     // std::cout << "\n\nCluster mean.";
//
//     // matrixCombinations();
//
//     // Update the shape parameter of the skew normal
//     clusterDFMetropolis();
//
//     // std::cout << "\n\nCluster df.";
//
//     // matrixCombinations();
//
//     // Metropolis step for batch parameters if more than 1 batch
//     // if(B > 1){
//     batchScaleMetropolis();
//
//     // std::cout << "\n\nBatch scale.";
//
//     // matrixCombinations();
//
//     batchShiftMetorpolis();
//
//     // std::cout << "\n\nBatch mean.";
//
//     // matrixCombinations();
//
//     // }
//   };
//
// };
//
// class mvtPredictive : public mvtSampler, public semisupervisedSampler {
//
// private:
//
// public:
//
//   using mvtSampler::mvtSampler;
//
//   mvtPredictive(
//     arma::uword _K,
//     arma::uword _B,
//     double _mu_proposal_window,
//     double _cov_proposal_window,
//     double _m_proposal_window,
//     double _S_proposal_window,
//     double _t_df_proposal_window,
//     double _rho,
//     double _theta,
//     double _lambda,
//     arma::uvec _labels,
//     arma::uvec _batch_vec,
//     arma::vec _concentration,
//     arma::mat _X,
//     arma::uvec _fixed
//   ) :
//     sampler(_K, _B, _labels, _batch_vec, _concentration, _X),
//     mvnSampler(_K,
//                _B,
//                _mu_proposal_window,
//                _cov_proposal_window,
//                _m_proposal_window,
//                _S_proposal_window,
//                _rho,
//                _theta,
//                _lambda,
//                _labels,
//                _batch_vec,
//                _concentration,
//                _X
//     ), mvtSampler(
//         _K,
//         _B,
//         _mu_proposal_window,
//         _cov_proposal_window,
//         _m_proposal_window,
//         _S_proposal_window,
//         _t_df_proposal_window,
//         _rho,
//         _theta,
//         _lambda,
//         _labels,
//         _batch_vec,
//         _concentration,
//         _X
//     ), semisupervisedSampler(_K, _B, _labels, _batch_vec, _concentration, _X, _fixed)
//   {
//   };
//
//   virtual ~mvtPredictive() { };
//
//   // virtual void sampleFromPriors() {
//   //
//   //   arma::mat X_k;
//   //
//   //   for(arma::uword k = 0; k < K; k++){
//   //     X_k = X.rows(arma::find(labels == k && fixed == 1));
//   //     cov.slice(k) = arma::diagmat(arma::stddev(X_k).t());
//   //     mu.col(k) = arma::mean(X_k).t();
//   //
//   //     // Draw from a shifted gamma distribution (i.e. gamma with location parameter)
//   //     t_df(k) = t_loc + arma::randg<double>( arma::distr_param(psi, 1.0 / chi));
//   //
//   //   }
//   //   for(arma::uword b = 0; b < B; b++){
//   //     for(arma::uword p = 0; p < P; p++){
//   //
//   //       // Fix the 0th batch at no effect; all other batches have an effect
//   //       // relative to this
//   //       // if(b == 0){
//   //       S(p, b) = 1.0;
//   //       m(p, b) = 0.0;
//   //       // } else {
//   //       // S(p, b) = 1.0 / arma::randg<double>( arma::distr_param(rho, 1.0 / theta ) );
//   //       // m(p, b) = arma::randn<double>() * S(p, b) / lambda + delta(p);
//   //       // }
//   //     }
//   //   }
//   // };
//
// };
//
//
//
// // Factory for creating instances of samplers
// //' @name samplerFactory
// //' @title Factory for different sampler subtypes.
// //' @description The factory allows the type of mixture implemented to change
// //' based upon the user input.
// //' @field new Constructor \itemize{
// //' \item Parameter: samplerType - the density type to be modelled
// //' \item Parameter: K - the number of components to model
// //' \item Parameter: labels - the initial clustering of the data
// //' \item Parameter: concentration - the vector for the prior concentration of
// //' the Dirichlet distribution of the component weights
// //' \item Parameter: X - the data to model
// //' }
// class mixtureFactory
// {
// public:
//   enum mixtureType {
//     MVN = 0
//   };
//
//   static std::unique_ptr<mixture> mixtureFactory(samplerType type,
//                                                 arma::uword K,
//                                                 arma::uvec labels,
//                                                 arma::vec concentration,
//                                                 arma::mat X
//   ) {
//     switch (type) {
//
//     case MVN: return std::make_unique<mvnMixture>(K,
//                                                   labels,
//                                                   concentration,
//                                                   X);
//     default: throw "invalid sampler type.";
//     }
//
//   }
//
// };
//
//
// // Factory for creating instances of samplers
// //' @name semisupervisedSamplerFactory
// //' @title Factory for different sampler subtypes.
// //' @description The factory allows the type of mixture implemented to change
// //' based upon the user input.
// //' @field new Constructor \itemize{
// //' \item Parameter: samplerType - the density type to be modelled
// //' \item Parameter: K - the number of components to model
// //' \item Parameter: labels - the initial clustering of the data
// //' \item Parameter: concentration - the vector for the prior concentration of
// //' the Dirichlet distribution of the component weights
// //' \item Parameter: X - the data to model
// //' }
// class semisupervisedSamplerFactory
// {
// public:
//   enum samplerType {
//     // G = 0,
//     MVN = 1,
//     MVT = 2,
//     MSN = 3
//   };
//
//   static std::unique_ptr<semisupervisedSampler> createSemisupervisedSampler(samplerType type,
//                                                                             arma::uword K,
//                                                                             arma::uword B,
//                                                                             double mu_proposal_window,
//                                                                             double cov_proposal_window,
//                                                                             double m_proposal_window,
//                                                                             double S_proposal_window,
//                                                                             double t_df_proposal_window,
//                                                                             double phi_proposal_window,
//                                                                             double rho,
//                                                                             double theta,
//                                                                             double lambda,
//                                                                             arma::uvec labels,
//                                                                             arma::uvec batch_vec,
//                                                                             arma::vec concentration,
//                                                                             arma::mat X,
//                                                                             arma::uvec fixed
//   ) {
//     switch (type) {
//     // case G: return std::make_unique<gaussianSampler>(K, labels, concentration, X);
//
//     case MVN: return std::make_unique<mvnPredictive>(K,
//                                                      B,
//                                                      mu_proposal_window,
//                                                      cov_proposal_window,
//                                                      m_proposal_window,
//                                                      S_proposal_window,
//                                                      rho,
//                                                      theta,
//                                                      lambda,
//                                                      labels,
//                                                      batch_vec,
//                                                      concentration,
//                                                      X,
//                                                      fixed);
//     case MVT: return std::make_unique<mvtPredictive>(K,
//                                                      B,
//                                                      mu_proposal_window,
//                                                      cov_proposal_window,
//                                                      m_proposal_window,
//                                                      S_proposal_window,
//                                                      t_df_proposal_window,
//                                                      rho,
//                                                      theta,
//                                                      lambda,
//                                                      labels,
//                                                      batch_vec,
//                                                      concentration,
//                                                      X,
//                                                      fixed);
//     case MSN: return std::make_unique<msnPredictive>(K,
//                                                      B,
//                                                      mu_proposal_window,
//                                                      cov_proposal_window,
//                                                      m_proposal_window,
//                                                      S_proposal_window,
//                                                      phi_proposal_window,
//                                                      rho,
//                                                      theta,
//                                                      lambda,
//                                                      labels,
//                                                      batch_vec,
//                                                      concentration,
//                                                      X,
//                                                      fixed);
//     default: throw "invalid sampler type.";
//     }
//
//   }
//
// };
//
//
//
// //' @title Sample batch mixture model
// //' @description Performs MCMC sampling for a mixture model with batch effects.
// //' @param X The data matrix to perform clustering upon (items to cluster in rows).
// //' @param K The number of components to model (upper limit on the number of clusters found).
// //' @param labels Vector item labels to initialise from.
// //' @param dataType Int, 0: independent Gaussians, 1: Multivariate normal, or 2: Categorical distributions.
// //' @param R The number of iterations to run for.
// //' @param thin thinning factor for samples recorded.
// //' @param concentration Vector of concentrations for mixture weights (recommended to be symmetric).
// //' @return Named list of the matrix of MCMC samples generated (each row
// //' corresponds to a different sample) and BIC for each saved iteration.
// // [[Rcpp::export]]
// Rcpp::List sampleMVN (
//     arma::mat X,
//     arma::uword K,
//     arma::uword B,
//     arma::uvec labels,
//     arma::uvec batch_vec,
//     double mu_proposal_window,
//     double cov_proposal_window,
//     double m_proposal_window,
//     double S_proposal_window,
//     double rho,
//     double theta,
//     double lambda,
//     arma::uword R,
//     arma::uword thin,
//     arma::vec concentration,
//     bool verbose = true,
//     bool doCombinations = false,
//     bool printCovariance = false
// ) {
//
//   // The random seed is set at the R level via set.seed() apparently.
//   // std::default_random_engine generator(seed);
//   // arma::arma_rng::set_seed(seed);
//
//
//   mvnSampler my_sampler(K,
//                         B,
//                         mu_proposal_window,
//                         cov_proposal_window,
//                         m_proposal_window,
//                         S_proposal_window,
//                         rho,
//                         theta,
//                         lambda,
//                         labels,
//                         batch_vec,
//                         concentration,
//                         X
//   );
//
//   // // Declare the factory
//   // samplerFactory my_factory;
//   //
//   // // Convert from an int to the samplerType variable for our Factory
//   // samplerFactory::samplerType val = static_cast<samplerFactory::samplerType>(dataType);
//   //
//   // // Make a pointer to the correct type of sampler
//   // std::unique_ptr<sampler> sampler_ptr = my_factory.createSampler(val,
//   //                                                                 K,
//   //                                                                 labels,
//   //                                                                 concentration,
//   //                                                                 X);
//
//   // We use this enough that declaring it is worthwhile
//   arma::uword P = X.n_cols;
//
//   // The output matrix
//   arma::umat class_record(floor(R / thin), X.n_rows);
//   class_record.zeros();
//
//   // We save the BIC at each iteration
//   arma::vec BIC_record = arma::zeros<arma::vec>(floor(R / thin));
//   arma::vec model_likelihood = arma::zeros<arma::vec>(floor(R / thin));
//   arma::uvec acceptance_vec = arma::zeros<arma::uvec>(floor(R / thin));
//   arma::mat weights_saved = arma::zeros<arma::mat>(floor(R / thin), K);
//
//   arma::cube mean_sum_saved(P, K * B, floor(R / thin)), mu_saved(P, K, floor(R / thin)), m_saved(P, B, floor(R / thin)), cov_saved(P, K * P, floor(R / thin)), t_saved(P, B, floor(R / thin)), cov_comb_saved(P, P * K * B, floor(R / thin));
//   // arma::field<arma::cube> cov_saved(my_sampler.P, my_sampler.P, K, floor(R / thin));
//   mu_saved.zeros();
//   cov_saved.zeros();
//   cov_comb_saved.zeros();
//   m_saved.zeros();
//   t_saved.zeros();
//
//   arma::uword save_int = 0;
//
//   // Sampler from priors
//   my_sampler.sampleFromPriors();
//   my_sampler.matrixCombinations();
//   // my_sampler.modelScore();
//   // sampler_ptr->sampleFromPriors();
//
//   // my_sampler.model_score = my_sampler.modelLogLikelihood(
//   //   my_sampler.mu,
//   //   my_sampler.tau,
//   //   my_sampler.m,
//   //   my_sampler.t
//   // ) + my_sampler.priorLogProbability(
//   //     my_sampler.mu,
//   //     my_sampler.tau,
//   //     my_sampler.m,
//   //     my_sampler.t
//   // );
//
//   // sample_prt.model_score->sampler_ptr.modelLo
//
//   // Iterate over MCMC moves
//   for(arma::uword r = 0; r < R; r++){
//
//     my_sampler.updateWeights();
//
//     // Metropolis step for batch parameters
//     my_sampler.metropolisStep();
//
//     my_sampler.updateAllocation();
//
//
//     // sampler_ptr->updateWeights();
//     // sampler_ptr->proposeNewParameters();
//     // sampler_ptr->updateAllocation();
//
//     // Record results
//     if((r + 1) % thin == 0){
//
//       // Update the BIC for the current model fit
//       // sampler_ptr->calcBIC();
//       // BIC_record( save_int ) = sampler_ptr->BIC;
//       //
//       // // Save the current clustering
//       // class_record.row( save_int ) = sampler_ptr->labels.t();
//
//       my_sampler.calcBIC();
//       BIC_record( save_int ) = my_sampler.BIC;
//       model_likelihood( save_int ) = my_sampler.model_likelihood;
//       class_record.row( save_int ) = my_sampler.labels.t();
//       acceptance_vec( save_int ) = my_sampler.accepted;
//       weights_saved.row( save_int ) = my_sampler.w.t();
//       mu_saved.slice( save_int ) = my_sampler.mu;
//       // tau_saved.slice( save_int ) = my_sampler.tau;
//       // cov_saved( save_int ) = my_sampler.cov;
//       m_saved.slice( save_int ) = my_sampler.m;
//       t_saved.slice( save_int ) = my_sampler.S;
//       mean_sum_saved.slice( save_int ) = my_sampler.mean_sum;
//
//
//       cov_saved.slice ( save_int ) = arma::reshape(arma::mat(my_sampler.cov.memptr(), my_sampler.cov.n_elem, 1, false), P, P * K);
//       cov_comb_saved.slice( save_int) = arma::reshape(arma::mat(my_sampler.cov_comb.memptr(), my_sampler.cov_comb.n_elem, 1, false), P, P * K * B);
//
//       if(printCovariance) {
//         std::cout << "\n\nCovariance cube:\n" << my_sampler.cov;
//         std::cout << "\n\nBatch covariance matrix:\n" << my_sampler.S;
//       }
//
//       save_int++;
//     }
//   }
//
//   if(verbose) {
//     std::cout << "\n\nCovariance acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.cov_count) / R;
//     std::cout << "\n\ncluster mean acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.mu_count) / R;
//     std::cout << "\n\nBatch covariance acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.S_count) / R;
//     std::cout << "\n\nBatch mean acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.m_count) / R;
//   }
//
//   return(List::create(Named("samples") = class_record,
//                       Named("means") = mu_saved,
//                       Named("covariance") = cov_saved,
//                       Named("batch_shift") = m_saved,
//                       Named("batch_scale") = t_saved,
//                       Named("mean_sum") = mean_sum_saved,
//                       Named("cov_comb") = cov_comb_saved,
//                       Named("weights") = weights_saved,
//                       Named("cov_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.cov_count) / R,
//                       Named("mu_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.mu_count) / R,
//                       Named("S_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.S_count) / R,
//                       Named("m_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.m_count) / R,
//                       Named("likelihood") = model_likelihood,
//                       Named("BIC") = BIC_record));
//
// };
//
// //' @title Mixture model
// //' @description Performs MCMC sampling for a mixture model.
// //' @param X The data matrix to perform clustering upon (items to cluster in rows).
// //' @param K The number of components to model (upper limit on the number of clusters found).
// //' @param labels Vector item labels to initialise from.
// //' @param fixed Binary vector of the items that are fixed in their initial label.
// //' @param dataType Int, 0: independent Gaussians, 1: Multivariate normal, or 2: Categorical distributions.
// //' @param R The number of iterations to run for.
// //' @param thin thinning factor for samples recorded.
// //' @param concentration Vector of concentrations for mixture weights (recommended to be symmetric).
// //' @return Named list of the matrix of MCMC samples generated (each row
// //' corresponds to a different sample) and BIC for each saved iteration.
// // [[Rcpp::export]]
// Rcpp::List sampleSemisupervisedMVN (
//     arma::mat X,
//     arma::uword K,
//     arma::uword B,
//     arma::uvec labels,
//     arma::uvec batch_vec,
//     arma::uvec fixed,
//     double mu_proposal_window,
//     double cov_proposal_window,
//     double m_proposal_window,
//     double S_proposal_window,
//     double rho,
//     double theta,
//     double lambda,
//     arma::uword R,
//     arma::uword thin,
//     arma::vec concentration,
//     bool verbose = true,
//     bool doCombinations = false,
//     bool printCovariance = false
// ) {
//
//   // // Set the random number
//   // std::default_random_engine generator(seed);
//   //
//   // // Declare the factory
//   // semisupervisedSamplerFactory my_factory;
//   //
//   // // Convert from an int to the samplerType variable for our Factory
//   // semisupervisedSamplerFactory::samplerType val = static_cast<semisupervisedSamplerFactory::samplerType>(dataType);
//   //
//   // // Make a pointer to the correct type of sampler
//   // std::unique_ptr<sampler> sampler_ptr = my_factory.createSemisupervisedSampler(val,
//   //                                                                               K,
//   //                                                                               labels,
//   //                                                                               concentration,
//   //                                                                               X,
//   //                                                                               fixed);
//
//
//   mvnPredictive my_sampler(K,
//                            B,
//                            mu_proposal_window,
//                            cov_proposal_window,
//                            m_proposal_window,
//                            S_proposal_window,
//                            rho,
//                            theta,
//                            lambda,
//                            labels,
//                            batch_vec,
//                            concentration,
//                            X,
//                            fixed
//   );
//
//   // // Declare the factory
//   // samplerFactory my_factory;
//   //
//   // // Convert from an int to the samplerType variable for our Factory
//   // samplerFactory::samplerType val = static_cast<samplerFactory::samplerType>(dataType);
//   //
//   // // Make a pointer to the correct type of sampler
//   // std::unique_ptr<sampler> sampler_ptr = my_factory.createSampler(val,
//   //                                                                 K,
//   //                                                                 labels,
//   //                                                                 concentration,
//   //                                                                 X);
//
//   arma::uword P = X.n_cols, N = X.n_rows;
//
//   // arma::uword restart_count = 0, n_restarts = 3, check_iter = 250;
//   // double min_acceptance = 0.15;
//   //
//   // restart:
//
//   // The output matrix
//   arma::umat class_record(floor(R / thin), X.n_rows);
//   class_record.zeros();
//
//   // We save the BIC at each iteration
//   arma::vec BIC_record = arma::zeros<arma::vec>(floor(R / thin)),
//     model_likelihood = arma::zeros<arma::vec>(floor(R / thin)),
//     model_likelihood_alt = arma::zeros<arma::vec>(floor(R / thin));
//
//   arma::uvec acceptance_vec = arma::zeros<arma::uvec>(floor(R / thin));
//   arma::mat weights_saved = arma::zeros<arma::mat>(floor(R / thin), K);
//
//   arma::cube mean_sum_saved(P, K * B, floor(R / thin)),
//   mu_saved(P, K, floor(R / thin)),
//   m_saved(P, B, floor(R / thin)),
//   cov_saved(P, K * P, floor(R / thin)),
//   S_saved(P, B, floor(R / thin)),
//   cov_comb_saved(P, P * K * B, floor(R / thin)),
//   alloc_prob(N, K, floor(R / thin)),
//   batch_corrected_data(N, P, floor(R / thin));
//
//   // arma::field<arma::cube> cov_saved(my_sampler.P, my_sampler.P, K, floor(R / thin));
//   mu_saved.zeros();
//   cov_saved.zeros();
//   cov_comb_saved.zeros();
//   m_saved.zeros();
//   S_saved.zeros();
//
//   arma::uword save_int = 0;
//
//   // Sampler from priors
//   my_sampler.sampleFromPriors();
//
//   my_sampler.matrixCombinations();
//   // my_sampler.modelScore();
//   // sampler_ptr->sampleFromPriors();
//
//   // my_sampler.model_score = my_sampler.modelLogLikelihood(
//   //   my_sampler.mu,
//   //   my_sampler.tau,
//   //   my_sampler.m,
//   //   my_sampler.t
//   // ) + my_sampler.priorLogProbability(
//   //     my_sampler.mu,
//   //     my_sampler.tau,
//   //     my_sampler.m,
//   //     my_sampler.t
//   // );
//
//   // sample_prt.model_score->sampler_ptr.modelLo
//
//   // Iterate over MCMC moves
//   for(arma::uword r = 0; r < R; r++){
//
//     my_sampler.updateWeights();
//
//     // Metropolis step for batch parameters
//     my_sampler.metropolisStep();
//
//     my_sampler.updateAllocation();
//
//
//     // sampler_ptr->updateWeights();
//     // sampler_ptr->proposeNewParameters();
//     // sampler_ptr->updateAllocation();
//
//     // Record results
//     if((r + 1) % thin == 0){
//
//       // Update the BIC for the current model fit
//       // sampler_ptr->calcBIC();
//       // BIC_record( save_int ) = sampler_ptr->BIC;
//       //
//       // // Save the current clustering
//       // class_record.row( save_int ) = sampler_ptr->labels.t();
//
//       my_sampler.calcBIC();
//       BIC_record( save_int ) = my_sampler.BIC;
//       model_likelihood( save_int ) = my_sampler.model_likelihood;
//       class_record.row( save_int ) = my_sampler.labels.t();
//       acceptance_vec( save_int ) = my_sampler.accepted;
//       weights_saved.row( save_int ) = my_sampler.w.t();
//       mu_saved.slice( save_int ) = my_sampler.mu;
//       // tau_saved.slice( save_int ) = my_sampler.tau;
//       // cov_saved( save_int ) = my_sampler.cov;
//       m_saved.slice( save_int ) = my_sampler.m;
//       S_saved.slice( save_int ) = my_sampler.S;
//       mean_sum_saved.slice( save_int ) = my_sampler.mean_sum;
//
//       alloc_prob.slice( save_int ) = my_sampler.alloc_prob;
//       cov_saved.slice ( save_int ) = arma::reshape(arma::mat(my_sampler.cov.memptr(), my_sampler.cov.n_elem, 1, false), P, P * K);
//       cov_comb_saved.slice( save_int) = arma::reshape(arma::mat(my_sampler.cov_comb.memptr(), my_sampler.cov_comb.n_elem, 1, false), P, P * K * B);
//
//       my_sampler.updateBatchCorrectedData();
//       batch_corrected_data.slice( save_int ) =  my_sampler.Y;
//
//       model_likelihood_alt( save_int ) = my_sampler.model_likelihood_alt;
//
//       if(printCovariance) {
//         std::cout << "\n\nCovariance cube:\n" << my_sampler.cov;
//         std::cout << "\n\nBatch covariance matrix:\n" << my_sampler.S;
//       }
//
//       save_int++;
//     }
//   }
//
//   if(verbose) {
//     std::cout << "\n\nCovariance acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.cov_count) / R;
//     std::cout << "\n\ncluster mean acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.mu_count) / R;
//     std::cout << "\n\nBatch covariance acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.S_count) / R;
//     std::cout << "\n\nBatch mean acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.m_count) / R;
//   }
//
//   // std::cout << "\nReciprocal condition number\n" << my_sampler.rcond_count;
//
//   return(List::create(Named("samples") = class_record,
//                       Named("means") = mu_saved,
//                       Named("covariance") = cov_saved,
//                       Named("batch_shift") = m_saved,
//                       Named("batch_scale") = S_saved,
//                       Named("mean_sum") = mean_sum_saved,
//                       Named("cov_comb") = cov_comb_saved,
//                       Named("weights") = weights_saved,
//                       Named("cov_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.cov_count) / R,
//                       Named("mu_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.mu_count) / R,
//                       Named("S_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.S_count) / R,
//                       Named("m_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.m_count) / R,
//                       Named("alloc_prob") = alloc_prob,
//                       Named("likelihood") = model_likelihood,
//                       Named("likelihood_alt") = model_likelihood_alt,
//                       Named("BIC") = BIC_record,
//                       Named("batch_corrected_data") = batch_corrected_data
//   )
//   );
//
// };

// class semiSupervisedMDIModel: virtual public mdiModel {
//   
// public:
//   
//   using mdiModel::mdiModel;
//   
//   std::vector< std::unique_ptr<semiSupervisedMixture> > mixtures;
//   
//   arma::umat fixed;
//   
//   semiSupervisedMDIModel(
//     arma::field<arma::mat> _X,
//     uvec _types,
//     arma::uvec _K,
//     arma::umat _labels,
//     arma::umat _fixed
//   ) :
//   mdiModel(
//     _X,
//     _types,
//     _K,
//     _labels,
//     _fixed
//   ) {
//     
//     // Indicator matrix for known labels across datasets
//     fixed = _fixed;
//     
//     outliers = 1 - fixed;
//     non_outliers = fixed;
//     
//     
//   };
//   
//   void initialiseMixtures() {
//     
//     // std::cout << "\n\nInitialising mixtures?\n";
//     
//     // delete mixtures*;
//     
//     // std::vector< std::unique_ptr<semiSupervisedMixture> > mixtures;
//     
//     // Initialise the collection of mixtures (will need a vector of types too,, currently all are MVN)
//     mixtures.reserve(L);
//     
//     semiSupervisedMixtureFactory my_factory;
//     
//     for(uword l = 0; l < L; l++) {
//       semiSupervisedMixtureFactory::mixtureType val = static_cast<semiSupervisedMixtureFactory::mixtureType>(types(l));
//       
//       // Push it to the back of the vector
//       mixtures.push_back(my_factory.createMixture(val,
//           K(l),
//           labels.col(l),
//           fixed.col(l),
//           X(l)
//         )
//       );
//       
//       // We have to pass the outliers back to the MDI level
//       non_outliers.col(l) = mixtures[l]->non_outliers;
//     }
//   };
// };

// // [[Rcpp::export]]
// Rcpp::List runMDI(arma::uword R,
//          arma::field<arma::mat> Y,
//          arma::uvec K,
//          arma::uvec types,
//          arma::umat labels,
//          arma::umat fixed) {
// 
//   uword L = size(Y)(0), N;
//   arma::field<arma::mat> X(L);
// 
//   // std::cout << "\n\nL: " << L;
//   
//   // std::cout << "\n\nY:\n" << Y;
//   
//   // throw
//   
//   for(uword l = 0; l < L; l++) {
//     X(l) = Y(l);
//     // std::cout << "\n\nX[l]:\n" << X(l).head_rows( 3 );
//   }
// 
//   // throw std::invalid_argument( "X is happy." );
//   
//   // std::cout << "\n\nMDI initialisation.";
//   mdiModel my_mdi(X, types, K, labels);
//   // std::cout << "\n\nL: " << my_mdi.L;
//   // throw std::invalid_argument( "MDI declated." );
//   
//   // Initialise the dataset level mixtures
//   my_mdi.initialiseMixtures();
// 
//   N = my_mdi.N;
// 
//   ucube class_record(R, N, L);
//   class_record.zeros();
// 
//   // std::cout << "\nMeh.";
//   mat phis_record(R, my_mdi.LC2);
//   cube weight_record(R, my_mdi.K_max, L);
// 
//   // std::cout << "\nSample from priors.";
//   my_mdi.sampleFromPriors();
//   for(arma::uword l = 0; l < L; l++) {
//     // my_mdi.mixtures[l].sampleFromPriors();
//     // my_mdi.mixtures[l].matrixCombinations();
// 
//     // (*my_mdi.mixtures)[l]->sampleFromPriors();
//     // (*my_mdi.mixtures)[l]->matrixCombinations();
//     
//     // std::cout << "\nSample from priors at mixture level.";
//     my_mdi.mixtures[l]->sampleFromPriors();
//     my_mdi.mixtures[l]->matrixCombinations();
//     // 
//     // std::cout << "\n\n\nMixture " << l << " members.";
//     // std::cout << "\nFixed inds:\n" << my_mdi.mixtures[l]->fixed_ind.head(4);
//     // std::cout << "\nUnfixed inds:\n" << my_mdi.mixtures[l]->unfixed_ind.head(4);
//     // 
//     // std::cout << "\nMembers:\n" << size(my_mdi.mixtures[l]->members);
//     // 
//     // std::cout << "\nN_k:\n" << my_mdi.mixtures[l]->N_k;
//     // 
//       
//   }
// 
//   for(uword r = 0; r < R; r++) {
// 
//     // std::cout << "\n\nNormalising constant.";
//     my_mdi.updateNormalisingConst();
// 
//     // std::cout << "\nStrategic latent variable.";
//     my_mdi.sampleStrategicLatentVariable();
// 
//     // std::cout << "\nWeights update.";
//     my_mdi.updateWeights();
// 
//     // std::cout << "\nPhis update.";
//     my_mdi.updatePhis();
// 
//     // std::cout << "\nSample mixture parameters.";
//     for(arma::uword l = 0; l < L; l++) {
//       // my_mdi.mixtures[l].sampleParameters();
//       // my_mdi.mixtures[l].matrixCombinations();
// 
//       // (*my_mdi.mixtures)[l]->sampleParameters();
//       // (*my_mdi.mixtures)[l]->matrixCombinations();
//       
//       my_mdi.mixtures[l]->sampleParameters();
//       my_mdi.mixtures[l]->matrixCombinations();
//       
//     }
// 
//     // std::cout << "\nAllocations update.";
//     my_mdi.updateAllocation();
// 
//     // std::cout << "\nSave objects.";
//     for(uword l = 0; l < L; l++) {
//       // arma::urowvec labels_l = my_mdi.labels.col(l).t();
//       class_record.slice(l).row(r) = my_mdi.labels.col(l).t();
//       weight_record.slice(l).row(r) = my_mdi.w.col(l).t();
//     }
// 
//     phis_record.row(r) = my_mdi.phis.t();
// 
//     // std::cout << "one iteration done.";
//     // throw;
//   }
// 
//   return(List::create(Named("samples") = class_record,
//                       Named("phis") = phis_record,
//                       Named("weights") = weight_record
//                       )
//            );
// 
// }



// [[Rcpp::export]]
Rcpp::List runSemiSupervisedMDI(arma::uword R,
  arma::uword thin,
  arma::field<arma::mat> Y,
  arma::uvec K,
  arma::uvec types,
  arma::umat labels,
  arma::umat fixed
) {
  
  // Indicator if the current iteration should be recorded
  bool save_this_iteration = false;
  
  uword L = size(Y)(0),
    n_saved = floor(R / thin) + 1,
    save_ind = 0,
    N = 0;
  arma::field<arma::mat> X(L);
  
  // std::cout << "\n\nL: " << L;
  
  // std::cout << "\n\nY:\n" << Y;
  
  // throw
  
  for(uword l = 0; l < L; l++) {
    X(l) = Y(l);
    // std::cout << "\n\nX[l]:\n" << X(l).head_rows( 3 );
  }
  
  // throw std::invalid_argument( "X is happy." );
  
  // std::cout << "\n\n" << fixed.head_rows(4);
  
  // std::cout << "\n\nMDI initialisation.";
  mdiModel my_mdi(X, types, K, labels, fixed);
  
  // std::cout << "\n\nL: " << my_mdi.L;
  // throw std::invalid_argument( "MDI declated." );
  
  // Initialise the dataset level mixtures
  // my_mdi.initialiseMixtures();
  my_mdi.initialiseMDI();
  
  // throw std::invalid_argument("Throw reached.");
  
  // std::cout << "\nHere:!\n";
  

  N = my_mdi.N;
  
  ucube class_record(n_saved, N, L),
    outlier_record(n_saved, N, L),
    N_k_record(my_mdi.K_max, L, n_saved);
      
  class_record.zeros();
  outlier_record.zeros();
  
  // std::cout << "\nMeh.";
  mat phis_record(n_saved, my_mdi.LC2),
    likelihood_record(n_saved, L);
  
  cube weight_record(n_saved, my_mdi.K_max, L);
  
  // field<mat> alloc(L);
  field<cube> alloc(n_saved);
  
  for(uword l = 0; l < L; l++) {
    // alloc(l) = zeros<mat>(N, K(l));
    alloc(l) = zeros<cube>(N, K(l), n_saved);
    
  }
  
  

  
  // std::cout << "\nSample from priors.";
  my_mdi.sampleFromPriors();
  
  for(arma::uword l = 0; l < L; l++) {
    // my_mdi.mixtures[l].sampleFromPriors();
    // my_mdi.mixtures[l].matrixCombinations();
    
    // (*my_mdi.mixtures)[l]->sampleFromPriors();
    // (*my_mdi.mixtures)[l]->matrixCombinations();
    
    // std::cout << "\nSample from priors at mixture level.";
    my_mdi.mixtures[l]->sampleFromPriors();
    // my_mdi.mixtures[l]->matrixCombinations();
    
  }
  

  // Save the initial values for each object
  for(uword l = 0; l < L; l++) {
    // arma::urowvec labels_l = my_mdi.labels.col(l).t();
    class_record.slice(l).row(save_ind) = my_mdi.labels.col(l).t();
    weight_record.slice(l).row(save_ind) = my_mdi.w.col(l).t();
    
    // Save the allocation probabilities
    // alloc(l) += my_mdi.mixtures[l]->alloc;
    alloc(l).slice(save_ind) = my_mdi.mixtures[l]->alloc;
    
    // Save the record of which items are considered outliers
    outlier_record.slice(l).row(save_ind) = my_mdi.mixtures[l]->outliers.t();
    
    // Save the complete likelihood
    likelihood_record(save_ind, l) = my_mdi.mixtures[l]->complete_likelihood;
  }
  
  // std::cout << "\n\nSave phis.";
  phis_record.row(save_ind) = my_mdi.phis.t();
  
  N_k_record.slice(save_ind) = my_mdi.N_k;
  
  
  // my_mdi.mixtures[0]->N_k;
  
  // throw std::invalid_argument("Throw reached.");
  
  
  // std::cout << "\n\nMain loop.";
  
  for(uword r = 0; r < R; r++) {
    
    // Should the current MCMC iteration be saved?
    save_this_iteration = ((r + 1) % thin == 0);
    
    // Rcpp::Rcout << "\nr: " << r;
    // Rcpp::Rcout << "\nSave this iteration? " << save_this_iteration;
    
    
    // std::cout << "\n\nNormalising constant.";
    my_mdi.updateNormalisingConst();
    
    // std::cout << "\nStrategic latent variable.";
    my_mdi.sampleStrategicLatentVariable();
    
    // std::cout << "\nWeights update.";
    my_mdi.updateWeights();
    
    // std::cout << "\nPhis update.";
    my_mdi.updatePhis();
    
    
    // std::cout << "\nSample mixture parameters.";
    for(arma::uword l = 0; l < L; l++) {
      
      // std::cout << "\n\nMixture " << l;
      
      // my_mdi.mixtures[l].sampleParameters();
      // my_mdi.mixtures[l].matrixCombinations();
      
      // (*my_mdi.mixtures)[l]->sampleParameters();
      // (*my_mdi.mixtures)[l]->matrixCombinations();
      
      
      my_mdi.mixtures[l]->sampleParameters();
      // my_mdi.mixtures[l]->matrixCombinations();
      
      // throw std::invalid_argument( "sample parameters." );
      
      
      // std::cout << "\n\n" << my_mdi.mixtures[l]->mu;
      
    }
    
    
    // throw std::invalid_argument("Throw reached.");
    
    // std::cout << "\nAllocations update.";
    my_mdi.updateAllocation();
    
    // Try and swap labels within datasets to improve the correlation between 
    // clusterings across datasets
    my_mdi.updateLabels();
    
    // throw std::invalid_argument( "lols." );
    
    
      if( save_this_iteration ) {
      
      // std::cout << "\nSave objects.";
      for(uword l = 0; l < L; l++) {
        save_ind++;
        
        // arma::urowvec labels_l = my_mdi.labels.col(l).t();
        class_record.slice(l).row(save_ind) = my_mdi.labels.col(l).t();
        weight_record.slice(l).row(save_ind) = my_mdi.w.col(l).t();
        
        // Save the allocation probabilities
        // alloc(l) += my_mdi.mixtures[l]->alloc;
        alloc(l).slice(save_ind) = my_mdi.mixtures[l]->alloc;
        
        // Save the record of which items are considered outliers
        outlier_record.slice(l).row(save_ind) = my_mdi.mixtures[l]->outliers.t();
        
        // Save the complete likelihood
        likelihood_record(save_ind, l) = my_mdi.mixtures[l]->complete_likelihood;
      }
      
      // std::cout << "\n\nSave phis.";
      phis_record.row(save_ind) = my_mdi.phis.t();
      
      N_k_record.slice(save_ind) = my_mdi.N_k;
    }

    // std::cout << "one iteration done.";
    // throw;
  }
  
  return(List::create(Named("samples") = class_record,
      Named("phis") = phis_record,
      Named("weights") = weight_record,
      Named("outliers") = outlier_record,
      Named("alloc") = alloc,
      Named("N_k") = N_k_record,
      Named("complete_likelihood") = likelihood_record
    )
  );
  
}