
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
    
    // The outleier component likelihood
    t_ll;
  
  // The dataset covariance
  mat global_cov_inv;
  
  using semiSupervisedMVN::semiSupervisedMVN;
  
  semiSupervisedTAGM(arma::uword _K,
    arma::uvec _labels,
    arma::uvec _fixed,
    arma::mat _X
  );
  
  // semiSupervisedTAGM(arma::uword _K,
  //                   arma::uvec _labels,
  //                   arma::uvec _fixed,
  //                   arma::mat _X) :
  //   semiSupervisedMVN(_K, _labels, _fixed, _X),
  //   semiSupervisedMixture(_K, _labels, _fixed, _X)
  // {
  //   
  //   // Doubles for leading coefficient of outlier distribution likelihood
  //   double lgamma_df_p = 0.0, lgamma_df = 0.0, log_pi_df = 0.0;
  //   vec dist_in_T(N), diff_data_global_mean(P);
  //   
  //   // for use in the outlier distribution
  //   mat global_cov = 0.5 * cov(X);
  //   
  //   // Do we need to add a very little to the diagonal to ensure we can inverse 
  //   // the dataset covariance matrix?
  //   uword count_here = 0;
  //   while( (rcond(global_cov) < 1e-5) && (count_here < 20) ) {
  //     global_cov.diag() += 1 * 0.0001;
  //     count_here++;
  //   }
  //   
  //   // std::cout << "\n\nGlobal covariance:\n" << global_cov;
  //   
  //   global_mean = (mean(X, 0)).t();
  //   
  //   // std::cout << "\n\nGlobal mean:\n" << global_mean;
  //   
  //   global_log_det = log_det(global_cov).real();
  //   global_cov_inv = inv(global_cov);
  // 
  //   // One of the parameters for the outlier weight
  //   b = (double) sum(outliers); // sum(non_outliers);
  //   outlier_weight = rBeta(b + u, N + v - b);
  //   
  //   // Components of the t-distribution likelihood that do not change
  //   lgamma_df_p = std::lgamma((df + (double) P) / 2.0);
  //   lgamma_df = std::lgamma(df / 2.0);
  //   log_pi_df = (double) P / 2.0 * std::log(df * M_PI);
  //   
  //   // Constant in t likelihood
  //   t_likelihood_const = lgamma_df_p - lgamma_df - log_pi_df - 0.5 * global_log_det;
  //   
  //   
  //   t_ll.set_size(N);
  //   for(uword n = 0; n < N; n++){
  //     
  //     diff_data_global_mean = X_t.col(n) - global_mean;
  //     
  //     // dist_in_T(n) = as_scalar(diff_data_global_mean.col(n).t() 
  //     //   * global_cov_inv 
  //     //   * diff_data_global_mean.col(n)
  //     // );
  //     
  //     // The "distance" part of the t-distribution for the current item
  //     dist_in_T(n) = as_scalar(diff_data_global_mean.t() 
  //       * global_cov_inv 
  //       * diff_data_global_mean
  //     );
  //     
  //     // std::cout << "\n\nT-distance form centre: " << dist_in_T(n);
  //     // 
  //     // std::cout << "\n\nExponent of likelihood: " << ((df + (double) P) / 2.0) * std::log(1.0 + (1.0 / df) * dist_in_T(n));
  //     
  //     // The likelihood of each point in the outlier distribution is constant
  //     t_ll(n) = t_likelihood_const 
  //       - ((df + (double) P) / 2.0) * std::log(1.0 + (1.0 / df) * dist_in_T(n));
  //     
  //     // std::cout << "\n\nOutlier likelihood: " << t_ll(n);
  //     
  //   }
  //   
  // };
  
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