
# include "semiSupervisedTAGM.h"

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp ;
using namespace arma ;

semiSupervisedTAGM::semiSupervisedTAGM(arma::uword _K,
                   arma::uvec _labels,
                   arma::uvec _fixed,
                   arma::mat _X) :
  semiSupervisedMVN(_K, _labels, _fixed, _X),
  semiSupervisedMixture(_K, _labels, _fixed, _X)
{
  
  // Doubles for leading coefficient of outlier distribution likelihood
  double lgamma_df_p = 0.0, lgamma_df = 0.0, log_pi_df = 0.0;
  vec dist_in_T(N), diff_data_global_mean(P);
  
  // for use in the outlier distribution
  mat global_cov = 0.5 * arma::cov(X);
  
  // Do we need to add a very little to the diagonal to ensure we can inverse 
  // the dataset covariance matrix?
  uword count_here = 0;
  while( (rcond(global_cov) < 1e-5) && (count_here < 20) ) {
    global_cov.diag() += 1 * 0.00001;
    count_here++;
  }
  
  // Rcpp::Rcout  << "\nNumber of vlaues added to diagonal: " << count_here;
  // 
  // Rcpp::Rcout  << "\n\nGlobal covariance:\n" << global_cov;
  
  global_mean = (mean(X, 0)).t();
  
  // Rcpp::Rcout  << "\n\nGlobal mean:\n" << global_mean;
  // 
  // Rcpp::Rcout  << "\nLog determinant\n" << log_det(global_cov);
  
  global_log_det = log_det(global_cov).real();
  global_cov_inv = inv(global_cov);
  
  // Rcpp::Rcout  << "\n\nOutlier weights.\n";
  
  // 0 indicates not outlier, 1 indicates outlier within assigned cluster.
  // Outliers do not contribute to cluster parameters.
  outliers = 1 - fixed;
  non_outliers = fixed;
  
  // One of the parameters for the outlier weight
  b = (double) sum(non_outliers); // sum(non_outliers);
  
  // Rcpp::Rcout  << "\nb.\n";
  
  outlier_weight = 1 - rBeta(b + v, N + u - b);
  
  // Rcpp::Rcout  << "\nNumber of outliers: " << b;
  // Rcpp::Rcout  << "\nv: " << v;
  // Rcpp::Rcout  << "\nu: " << u;
  // Rcpp::Rcout  << "\nExpected outlier weight (calculated): " << accu(rBeta(1000, b + u, N + v - b))/1000;
  // Rcpp::Rcout  << "\nExpected outlier weight (true): " << (b + u) / (u + N + v);
  
  // Rcpp::Rcout  << "\n\nOutlier weight sampled .\n";
  // 
  // Rcpp::Rcout  << "\n\nOutlier likelihood parts.\n";
  
  // Components of the t-distribution likelihood that do not change
  lgamma_df_p = std::lgamma((df + (double) P) / 2.0);
  
  // Rcpp::Rcout  << "\n\nOutlier likelihood part 1.\n";
  
  lgamma_df = std::lgamma(df / 2.0);
  
  // Rcpp::Rcout  << "\n\nOutlier likelihood part 2.\n";
  
  log_pi_df = ((double) P / 2.0) * std::log(df * M_PI);
  
  // Rcpp::Rcout  << "\n\nOutlier likelihood part 3.\n";
  
  // Constant in t likelihood
  t_likelihood_const = lgamma_df_p - lgamma_df - log_pi_df - 0.5 * global_log_det;
  
  // Rcpp::Rcout  << "\n\nOutlier likelihood part 4.\n";
  
  t_ll.set_size(N);
  
  // Rcpp::Rcout  << "\nLikelihood declared";
  
  for(uword n = 0; n < N; n++){

    diff_data_global_mean = X_t.col(n) - global_mean;

    // dist_in_T(n) = as_scalar(diff_data_global_mean.col(n).t()
    //   * global_cov_inv
    //   * diff_data_global_mean.col(n)
    // );

    // The "distance" part of the t-distribution for the current item
    dist_in_T(n) = as_scalar(diff_data_global_mean.t()
      * global_cov_inv
      * diff_data_global_mean
    );

    // Rcpp::Rcout  << "\n\nT-distance form centre: " << dist_in_T(n);
    //
    // Rcpp::Rcout  << "\n\nExponent of likelihood: " << ((df + (double) P) / 2.0) * std::log(1.0 + (1.0 / df) * dist_in_T(n));

    // The likelihood of each point in the outlier distribution is constant
    t_ll(n) = t_likelihood_const
      - ((df + (double) P) / 2.0) * std::log(1.0 + (1.0 / df) * dist_in_T(n));

    // Rcpp::Rcout  << "\n\nOutlier likelihood: " << t_ll(n);

  }
  
  // Rcpp::Rcout  << "\n\nOK we declare outliers.\n";
  
  
  // Rcpp::Rcout  << "\n\nOK we dance.\n";
};

double semiSupervisedTAGM::calcTdistnLikelihood(arma::uword n) {
  
  double exponent = 0.0, ll = 0.0;
  
  exponent = as_scalar(
    (X_t.col(n) - global_mean).t()
    * global_cov_inv
    * (X_t.col(n) - global_mean)
  );
  
  // Rcpp::Rcout  << "\nConstant: " << t_likelihood_const;
  // Rcpp::Rcout  << "\nGlobal covariance log determinant: " << 0.5 * global_log_det;
  
  // Rcpp::Rcout  << dist_in_T(n);
  
  ll = t_likelihood_const 
    - ((df + (double) P) / 2.0) * std::log(1.0 + (1.0 / df) * exponent);
  
  // Rcpp::Rcout  << "\nLog likelihood: " << ll;
  
  return ll;
};

void semiSupervisedTAGM::updateOutlierWeights(){
  b = (double) accu(non_outliers);
  outlier_weight = 1 - rBeta(b + v, N + u - b);
  
  // Rcpp::Rcout  << "\n\nOutlier weight: " << outlier_weight;
  
};


arma::uword semiSupervisedTAGM::sampleOutlier(arma::uword n) {
  
  uword pred_outlier = 0;
  // arma::uword k = labels(n);
  
  double out_likelihood = 0.0;
  arma::vec outlier_prob(2), point = X_t.col(n);
  outlier_prob.zeros();
  
  // if(std::abs(likelihood(n) - logLikelihood(point, labels(n))) > 1e-12) {
  //   Rcpp::Rcout << "\n\nMy saved likelihood and calculated differ.\n";
  //   throw std::invalid_argument("Thrown.");
  // }
  
  // The likelihood of the point in the current cluster
  outlier_prob(0) = likelihood(n) + log(1 - outlier_weight);
  // outlier_prob(0) = logLikelihood(point, labels(n)) + log(1 - outlier_weight);
  
  // Calculate outlier likelihood
  // out_likelihood = calcTdistnLikelihood(n); //calcTdistnLikelihood(point);
  // 
  // if(t_ll(n) != out_likelihood) {
  //   Rcpp::Rcout  << "\n\nOutlier ind ll: " << out_likelihood;
  //   Rcpp::Rcout  << "\nOutlier big ll: " << t_ll(n);
  //   throw;
  // }
  
  out_likelihood = t_ll(n) + log(outlier_weight);
  outlier_prob(1) = out_likelihood;
  
  // Rcpp::Rcout  << "\n\nOutlier probability:\n" << outlier_prob;
  
  // Normalise and overflow
  outlier_prob = exp(outlier_prob - max(outlier_prob));
  outlier_prob = outlier_prob / sum(outlier_prob);
  
  // Prediction and update
  u = arma::randu<double>( );
  pred_outlier = sum(u > cumsum(outlier_prob));
  
  return pred_outlier;
};



void semiSupervisedTAGM::updateAllocation(arma::vec weights, arma::mat upweigths) {
  
  double u = 0.0;
  arma::uvec uniqueK;
  arma::vec comp_prob(K);
  
  // First update the outlier parameters
  updateOutlierWeights();
  
  // for (auto& n : unfixed_ind) {
  for (uword n = 0; n < N; n++) {
    
    // Rcpp::Rcout  << "\n\nWe reach this.";
    
    // for(arma::uword n = 0; n < N; n++){
    ll = itemLogLikelihood(X_t.col(n));
    
    // Update with weights
    comp_prob = ll + log(weights) + log(upweigths.col(n));
    
    // Record the likelihood - this is used to calculate the observed likelihood
    // likelihood(n) = accu(comp_prob);
    observed_likelihood += accu(comp_prob);

    if(fixed(n) == 0) {
      
      // Normalise and overflow
      comp_prob = exp(comp_prob - max(comp_prob));
      comp_prob = comp_prob / sum(comp_prob);
      
      // Prediction and update
      u = arma::randu<double>( );
      labels(n) = sum(u > cumsum(comp_prob));
      alloc.row(n) = comp_prob.t();
      
      // Update if the point is an outlier or not
      outliers(n) = sampleOutlier(n);
    }
    
    // Update the complete likelihood based on the new labelling
    complete_likelihood += ll(labels(n));
    likelihood(n) = ll(labels(n));
  }
  
  // Update our vector indicating if we're not an outlier
  non_outliers = 1 - outliers;
  
  // Number of occupied components (used in BIC calculation)
  uniqueK = unique(labels);
  K_occ = uniqueK.n_elem;
};


// void semiSupervisedTAGM::calcBIC() {
// };