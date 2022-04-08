// gp.cpp
// =============================================================================
// included dependencies
# include "logLikelihoods.h"
# include "gp.h"
  
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp ;
using namespace arma ;

// =============================================================================
// mvn class

gp::gp(arma::uword _K, arma::uvec _labels, arma::mat _X) : 
  density(_K, _labels, _X) 
{
  
  zero_vec.set_size(P);
  zero_vec.zeros();
  
  // Hyperparameters
  amplitude.set_size(K);
  amplitude.ones();
  
  length.set_size(K);
  length.ones();
    
  noise.set_size(K);
  noise.ones();
 
  // Set the size of the objects to hold the component specific parameters
  mu.set_size(P, K);
  mu.zeros();
  
  kernel_sub_block.set_size(P, P, K);
  kernel_sub_block.zeros();
  
  I_p = eye(P, P);
  
  // Those weird GP objects that can be massive and vary from component to 
  // component, the size being a function of N_k.
  repeated_time_indices.set_size(K);
  repeated_mean_vector.set_size(K);
  covariance_matrix.set_size(K);
  flattened_component_data.set_size(K);
  
  t_inds = arma::linspace< uvec >(0, P - 1, P);
  time_difference_mat.set_size(P, P);
  time_difference_mat.zeros();
  for(uword ii = 0; ii < (P - 1); ii++) {
    for(uword jj = (ii + 1); jj < P; jj++) {
      time_difference_mat(ii, jj) = jj - ii;
      time_difference_mat(jj, ii) = jj - ii;
    }
  }
  time_difference_mat = arma::pow(time_difference_mat, 2.0);
  
  // These will hold vertain matrix operations to avoid computational burden
  // The log determinant of each cluster covariance
  cov_log_det = arma::zeros<arma::vec>(K);
  
  // Inverse of the cluster covariance
  inverse_covariance.set_size(K);
  
  // Mean vector and covariance matrix and a component weight
  // n_param = P * (1 + (P + 1) * 0.5);
  
  // Empirical Bayesian hyperparameters for the mean and covariance
  // empiricalBayesHyperparameters();
  
};

void gp::sampleKthComponentHyperParameterPrior(uword k) {
  amplitude(k) = 1.0; // rHalfCauchy(0, 5);
  length(k) = 1.0; // rHalfCauchy(0, 5);
  noise(k) = 1.0; // std::exp(randn< double >());
  if(noise(k) < 1e-6) {
    noise(k) = 1e-6;
  }
};

void gp::sampleHyperParameterPriors() {
  for(uword k = 0; k < K; k++) {
    sampleKthComponentHyperParameterPrior(k);
    // if(noise(k) < 1e-6) {
    //   noise(k) = 1e-6;
    // }
  }
  // Rcpp::Rcout << "\nAmplitude:\n" << amplitude;
  // Rcpp::Rcout << "\n\nLength:\n" << length;
  // Rcpp::Rcout << "\n\nNoise:\n" << noise;
};

// arma::mat gp::calculateCovarianceKernel(arma::uvec t_inds) {
//   return cov_kernel_ptr->calculateCovarianceMatrix(t_inds);
// };

// std::unique_ptr<kernel> gp::initialiseKernel(uword kernel_type) {
// 
//   kernelFactory my_factory;
//   
//   // Convert the unsigned integer into a mixture type object
//   kernelFactory::kernelType val = static_cast<kernelFactory::kernelType>(kernel_type);
//   
//   // Create a smart pointer to the correct type of model
//   std::unique_ptr<kernel> kernel_ptr = my_factory.createKernel(val);
// 
//   return kernel_ptr;
// }

// === Prior-related functions =================================================
void gp::sampleMuPrior() {
  for(arma::uword k = 0; k < K; k++){
    mu.col(k) = arma::mvnrnd(zero_vec, kernel_sub_block.slice(k));
  }
};

void gp::sampleFromPriors() {
  // Rcpp::Rcout<< "\nSet hypers.";
  sampleHyperParameterPriors();
  calculateKernelSubBlock();
  // Rcpp::Rcout<< "\nSample mean function from prior.";
  sampleMuPrior();
};

// === Covariance function =====================================================

mat gp::calculateKthComponentKernelSubBlock(double amplitude, double length) {
  mat sub_block(P, P);
  sub_block.zeros();
  // for(uword ii = 0; ii < P; ii++) {
  std::for_each(
    std::execution::par,
    P_inds.begin(),
    P_inds.end(),
    [&](uword ii) {
      sub_block(ii, ii) = amplitude;
      for(uword jj = ii + 1; jj < P; jj++) {
        sub_block(ii, jj) = squaredExponentialFunction(
          amplitude, 
          length, 
          ii, 
          jj
        );
        sub_block(jj, ii) = sub_block(ii, jj);
      }
    }
  );
  return sub_block;
};

void gp::calculateKernelSubBlock() {
  kernel_sub_block.ones();
  for(uword k = 0; k < K; k++) {
    kernel_sub_block.slice(k) = calculateKthComponentKernelSubBlock(
      amplitude(k),
      length(k)
    );
    // for(uword ii = 0; ii < P - 1; ii++) {
    //   kernel_sub_block.slice(k)(ii, ii) = amplitude(k);
    //   for(uword jj = ii + 1; jj < P; jj++) {
    //     kernel_sub_block.slice(k)(ii, jj) = squaredExponentialFunction(
    //       amplitude(k), 
    //       length(k), 
    //       ii, 
    //       jj
    //     );
    //       // amplitude(k) * std::exp(- std::pow(jj - ii, 2.0) / length(k));
    //     kernel_sub_block.slice(k)(jj, ii) = kernel_sub_block.slice(k)(ii, jj);
    // // kernel_sub_block.slice(k) = arma::exp(
    // //   2 * std::log(amplitude(k)) - time_difference_mat / (1 / length(k))
    // // );
    //   }
    // }
    
    // Rcpp::Rcout << "\n\nK: " << k;
    // Rcpp::Rcout << "\nKernel sub-block is positive semi-definite: " << kernel_sub_block.slice(k).is_sympd() << endl;
    // Rcpp::Rcout << "\n\nKernel sub-block:\n" << kernel_sub_block.slice(k) << endl;
    
  }
};

// void gp::constructCovarianceMatrix(uword n_k, uword k) {
//   uword inner_base = 0, outer_base = 0;
//   uvec inner_inds(P), outer_inds(P);
//   inner_inds.zeros(), outer_inds.zeros();
//   // mat placehold_check(n_k * P, n_k * P);
//   
//   covariance_matrix(k).reset();
//   covariance_matrix(k).set_size(n_k * P, n_k * P);
//   covariance_matrix(k) = repmat(kernel_sub_block.slice(k), n_k, n_k);
//   
//   // for(uword n = 0; n < n_k; n++) {
//   //   outer_inds = t_inds + outer_base;
//   //   for(uword jj = 0; jj < n_k; jj++) {
//   //     inner_inds = t_inds + inner_base;
//   //     placehold_check.submat(outer_inds, inner_inds) = kernel_sub_block.slice(k);
//   //     // placehold_check.submat(inner_inds, outer_inds) = kernel_sub_block.slice(k);
//   //     inner_base += P;
//   //   }
//   //   outer_base += P;
//   // }
//   // bool same = approx_equal(covariance_matrix(k), placehold_check, "reldiff", 0.1);
//   // if(! same) {
//   //   Rcpp::Rcerr << "\nPlaceholder and more clever way are not equal.";
//   // }
// };

mat gp::constructCovarianceMatrix(uword n_k, uword k, mat kernel_sub_block) {
  mat covariance_matrix(n_k * P, n_k * P);
  covariance_matrix.zeros();
  covariance_matrix = repmat(kernel_sub_block, n_k, n_k);
  // Rcpp::Rcout << "\nCovariance matrix full:\n" << covariance_matrix;
  // Rcpp::Rcerr << "\nWe locking down things here.\n";
  return covariance_matrix;
};

mat gp::invertComponentCovariance(uword n_k, double noise, mat kernel_sub_block) {
  
  uvec rel_inds;
  mat J(n_k, n_k), Q_k(P, P), Z_k(P, P), I_NkP(n_k * P, n_k * P);
  J.ones();
  Q_k.zeros();
  Z_k.zeros();
  
  I_NkP = eye(n_k * P, n_k * P);
  Q_k = I_p + (n_k / noise) * kernel_sub_block;
  Z_k = inv(Q_k);
  return (1.0 / noise) * I_NkP - (1 / (n_k * noise)) * kron(J, I_p - Z_k);
};

double gp::componentCovarianceDeterminant(uword k, uword n_k) {
  return std::pow(noise(k), n_k * P) * det(I_p + (n_k / noise (k)) * kernel_sub_block.slice(k));
};

double gp::componentCovarianceLogDeterminant(uword k, uword n_k) {
  return n_k * P * std::log(noise(k)) +  log_det(I_p + (n_k / noise (k)) * kernel_sub_block.slice(k)).real();
};

void gp::calculateInverseCovariance(umat members, uvec non_outliers) {
  
  uword n_k = 0;
  uvec rel_inds;
  
  for(uword k = 0; k < K; k++) {
    // Find the items relevant to sampling the parameters
    rel_inds = find((members.col(k) == 1) && (non_outliers == 1));
    // Find how many labels have the value
    n_k = rel_inds.n_elem;
    
    inverse_covariance(k) = invertComponentCovariance(
      n_k, 
      noise(k),
      kernel_sub_block.slice(k)
    );
    
    cov_log_det(k) = componentCovarianceLogDeterminant(k, n_k);
  }
};

// void gp::matrixCombinations(umat members, uvec non_outliers) {
//   calculateInverseCovariance(members, non_outliers);
// }


// void gp::constructCovarianceMatrix(uvec N_k) {
//   for(uword k = 0; k < K; k++) {
//     constructCovarianceMatrixK(N_k(k), k);
//   }
// };


// === Mean function posterior==================================================
// vec gp::posteriorMeanParameter(uword k, uword n_k, vec data) {
//   vec mu_tilde(P);
//   for(uword p = 0; p < P; p++) {
//     mu_tilde(p) = as_scalar(
//       covariance_matrix(k).row(p)
//       * inverse_covariance(k)
//       * data
//     );
//   }
//   return mu_tilde;
// };

vec gp::posteriorMeanParameter(
    vec data, 
    mat covariance_matrix, 
    mat inverse_covariance_matrix
) {
  vec mu_tilde(P);
  for(uword p = 0; p < P; p++) {
    mu_tilde(p) = as_scalar(
      covariance_matrix.row(p)
    * inverse_covariance_matrix
    * data
    );
  }
  return mu_tilde;
};


mat gp::posteriorCovarianceParameter(
    mat covariance_matrix,
    mat inverse_covariance_matrix) {
  mat cov_tilde(P, P);
  cov_tilde.zeros();
  // // cov_tilde = ;
  // for(uword ii = 0; ii < P; ii++) {
  //   for(uword jj = 0; jj < P; jj++) {
  //     cov_tilde(ii, jj) = as_scalar(
  //       covariance_matrix(ii, jj)
  //     - covariance_matrix.row(ii) 
  //     * inverse_covariance_matrix 
  //     * covariance_matrix.col(jj)
  //     );
  //   }
  // }
  // 
  // auto I = linspace(0, P - 1);
  
  std::for_each(std::execution::par,
                P_inds.begin(),
                P_inds.end(),
                [&](uword ii) {
  // for(uword ii = 0; ii < P; ii++) {
  // if(ii >= P) {
  //   Rcpp::Rcout << "\ni: " << ii;
  //   // throw;
  // }

    cov_tilde(ii, ii) = as_scalar(
      covariance_matrix(ii, ii)
    - covariance_matrix.row(ii) 
    * inverse_covariance_matrix 
    * covariance_matrix.col(ii)
    );
    
    for(uword jj = ii + 1; jj < P; jj++) {
      cov_tilde(ii, jj) = as_scalar(
         covariance_matrix(ii, jj)
         - covariance_matrix.row(ii) 
           * inverse_covariance_matrix 
           * covariance_matrix.col(jj)
      );
      cov_tilde(jj, ii) = cov_tilde(ii, jj);
    }
  }
  );
  
  return cov_tilde;
};

// mat gp::posteriorCovarianceParameter(uword k, uword n_k) {
//   mat cov_tilde(P, P);
//   cov_tilde.zeros();
//   for(uword ii = 0; ii < P; ii++) {
//     for(uword jj = 0; jj < P; jj++) {
//       cov_tilde(ii, jj) = as_scalar(
//         covariance_matrix(k)(ii, jj)
//         - covariance_matrix(k).row(ii) 
//           * inverse_covariance(k) 
//           * covariance_matrix(k).col(jj)
//       );
//     }
//   }
//   return cov_tilde;
// };

void gp::sampleMeanPosterior(uword k, uword n_k, mat data) {
  vec mu_tilde(P), data_vec = data.as_row().t() ;
  mat cov_tilde(P, P);
  
  
  
  // Rcpp::Rcout << "\n\nConstruct covariance matrix from sub-blocks.";
  
  // calculateKernelSubBlock(k);
  
  // Objects related to the covariance function
  // constructCovarianceMatrix(n_k, k);
  mat covariance_matrix = constructCovarianceMatrix(n_k, k, kernel_sub_block.slice(k));
  
  // Rcpp::Rcout << "\nInvert the covariance matrix.";
  // inverse_covariance(k) = invertComponentCovariance(k, n_k);
  mat inverse_covariance = invertComponentCovariance(n_k, noise(k), kernel_sub_block.slice(k));
  
  // Rcpp::Rcout << "\ncalculate the log-determinant.";
  cov_log_det(k) = componentCovarianceLogDeterminant(k, n_k);
  
  // Posterior parameters
  // Rcpp::Rcout << "\n\nCalculate the mean posterior hyperparameters.";
  mu_tilde = posteriorMeanParameter(data_vec, covariance_matrix, inverse_covariance);
  
  // Rcpp::Rcout << "\nThe covariance matrix used in the sampling.";
  cov_tilde = posteriorCovarianceParameter(covariance_matrix, inverse_covariance);

  // if(! cov_tilde.is_sympd()) {
  //   Rcpp::Rcout << "\nCov is off\n\n" << cov_tilde;
  //   Rcpp::Rcout << "\n\nAmplitude: " << amplitude(k);
  //   Rcpp::Rcout << "\nNoise: " << noise(k);
  //   Rcpp::Rcout << "\nLength: " << length(k);
  // }
  
  mu.col(k) = mvnrnd(mu_tilde, cov_tilde);
  
  // sampleHyperParametersKthComponent(k, mu_tilde, cov_tilde);
  // sampleHyperParametersKthComponent(k, mu_tilde, cov_tilde);
  // sampleHyperParametersKthComponent(
  //   k,
  //   n_k,
  //   mu_tilde,
  //   data_vec,
  //   cov_tilde
  // );
  
  // sampleNoise(k, n_k, data);
  
  // Rcpp::Rcout << "\nSample the mean function.";
  // Rcpp::Rcout << "\n\nMean hyper:\n" << mu_tilde.size();
  // Rcpp::Rcout << "\n\nCov hyper:\n" << cov_tilde.n_cols;
  // return mvnrnd(mu_tilde, cov_tilde);
};

void gp::sampleKthComponentParameters(uword k, umat members, uvec non_outliers) {
  
  // Find the items relevant to sampling the parameters
  uvec rel_inds = find((members.col(k) == 1) && (non_outliers == 1));
  vec component_data_vec;
  mat component_data;
  
  // Find how many labels have the value
  uword n_k = rel_inds.n_elem;
  
  if(n_k > 0){
    
    component_data.reset();
    component_data_vec.reset();
    
    // Component data
    component_data = X.rows( rel_inds ) ;
    
    // sampleHyperParameters();
    // mu.col(k) = 
    sampleMeanPosterior(k, n_k, component_data);
  } else {
    
    // Rcpp::Rcout << "\n\nEmpty component!\n";
    
    // sampleHyperParameters();
    
    // Sample from the prior
    mu.col(k) = arma::mvnrnd(zero_vec, kernel_sub_block.slice(k));
    sampleKthComponentHyperParameterPrior(k);
  }
};

void gp::sampleParameters(arma::umat members, arma::uvec non_outliers) {
  
  arma::uword n_k = 0;
  uvec rel_inds;
  // density_members = members;
  // density_non_outliers = non_outliers;
  
  // Rcpp::Rcout << "\nCalculate sub-blocks.";
  
  calculateKernelSubBlock();
  
  // Rcpp::Rcout << "\nAmplitude:\n" << amplitude;
  // Rcpp::Rcout << "\n\nLength:\n" << length;
  // Rcpp::Rcout << "\n\nNoise:\n" << noise;
  
  // calculateInverseCovariance(members, non_outliers);
  // sampleHyperParameters();
  
  // for(uword k = 0; k < K; k++) {
  std::for_each(
    std::execution::par,
    K_inds.begin(),
    K_inds.end(),
    [&](uword k) {
      sampleKthComponentParameters(k, members, non_outliers);
    }
  );
  
  // for (arma::uword k = 0; k < K; k++) {
  // 
  //   // Find the items relevant to sampling the parameters
  //   rel_inds = find((members.col(k) == 1) && (non_outliers == 1));
  // 
  //   // Find how many labels have the value
  //   n_k = rel_inds.n_elem;
  // 
  //   if(n_k > 0){
  // 
  //     // Component data
  //     vec component_data = X.rows( rel_inds ).as_row().t() ;
  // 
  //     // sampleHyperParameters();
  //     // mu.col(k) = 
  //       sampleMeanPosterior(k, n_k, component_data);
  //   } else {
  // 
  //     // sampleHyperParameters();
  // 
  //     // Rcpp::Rcout << "\nSampling from prior (again).";
  // 
  //     // Sample from the prior
  //     mu.col(k) = arma::mvnrnd(zero_vec, kernel_sub_block.slice(k));
  //   }
  // 
  // }
};

// === Hyper-parameters ========================================================

// Need to sample the hyperparameter and recalculate the mu tilde / cov tilde
double gp::hyperParameterKernel(double hyper, vec mu_k, vec mu_tilde, mat cov_tilde) {
  double score = 0.0;
  vec mean_diff = mu_k - mu_tilde;

  // if(! cov_tilde.is_sympd()) {
  //   Rcpp::Rcout << "\n\nNot positive semi-definite.\n";
  // }
  
  score = log(2 * M_PI)
    + log_det(cov_tilde).real()
    + as_scalar(mean_diff.t() * inv_sympd(cov_tilde) * mean_diff);
  score *= -0.5;
  score += pHalfCauchy(hyper, 0, 5);

  return score;
};


// //' @title Metropolis acceptance step
// //' @description Given a probaility, randomly accepts by sampling from a uniform 
// //' distribution.
// //' @param acceptance_prob Double between 0 and 1.
// //' @return Boolean indicating acceptance.
// bool metropolisAcceptanceStep(double acceptance_prob) {
//   double u = arma::randu();
//   return (u < acceptance_prob);
// };

// void gp::sampleLength(uword k, vec mu_tilde, mat cov_tilde) {
//   bool accept = false;
//   double 
//     length_proposal_window = 0.02, 
//       acceptance_prob = 0.0, 
//       new_score = 0.0, 
//       old_score = 0.0,
//       new_length = 0.0;
//   
//   new_length = std::exp(std::log(length(k) + randn() * length_proposal_window));
//   new_score = hyperParameterKernel(new_length, mu.col(k), mu_tilde, cov_tilde);
//   old_score = hyperParameterKernel(length(k), mu.col(k), mu_tilde, cov_tilde);
// 
//   acceptance_prob =  std::min(1.0, std::exp(new_score - old_score));
//   accept = metropolisAcceptanceStep(acceptance_prob);
//   if(accept) {
//     length(k) = new_length;
//     // length_acceptance_count(k)++;
//   }
// };



void gp::sampleLength(uword k, uword n_k, vec mu_tilde, vec component_data, mat cov_tilde) {
  bool accept = false;
  double 
    length_proposal_window = 75, 
    acceptance_prob = 0.0, 
    new_score = 0.0, 
    old_score = 0.0,
    new_length = 0.0;
  vec new_mu_tilde(P);
  mat 
    new_sub_block(P, P), 
    new_cov_mat(n_k * P, n_k * P),
    new_inv_cov_mat(n_k * P, n_k * P), 
    new_cov_tilde_new(P, P);
  
  new_length = proposeNewNonNegativeValue(length(k), length_proposal_window);
  // new_length = std::exp(std::log(length(k) + randn() * length_proposal_window));
  if(new_length < 1e-6) {
    return;
  }
  new_sub_block = calculateKthComponentKernelSubBlock(amplitude(k), new_length);
  new_cov_mat = constructCovarianceMatrix(n_k, k, new_sub_block);
  new_inv_cov_mat = invertComponentCovariance(n_k, noise(k),new_sub_block);
  new_cov_tilde_new = posteriorCovarianceParameter(new_cov_mat, new_inv_cov_mat);
  new_mu_tilde = posteriorMeanParameter(
    component_data, 
    new_cov_mat, 
    new_inv_cov_mat
  );
  
  new_score = hyperParameterKernel(
    new_length, 
    mu.col(k), 
    new_mu_tilde, 
    new_cov_tilde_new
  );
  
  old_score = hyperParameterKernel(
    length(k), 
    mu.col(k), 
    mu_tilde, 
    cov_tilde
  );
  
  acceptance_prob =  std::min(1.0, std::exp(new_score - old_score));
  accept = metropolisAcceptanceStep(acceptance_prob);
  if(accept) {
    length(k) = new_length;
    // amplitude_acceptance_count(k)++;
  }
};

double gp::proposeNewNonNegativeValue(double x, double window) {
  return randg( distr_param( x * window, 1.0 / window) );
  // return std::exp(std::log(x + randn() * window));
}

void gp::sampleAmplitude(uword k, uword n_k, vec mu_tilde, vec component_data, mat cov_tilde) {
  bool accept = false;
  double 
    amplitude_proposal_window = 75, 
    acceptance_prob = 0.0, 
    new_score = 0.0, 
    old_score = 0.0,
    new_amplitude = 0.0;
  vec new_mu_tilde(P);
  mat 
    new_sub_block(P, P), 
    new_cov_mat(n_k * P, n_k * P),
    new_inv_cov_mat(n_k * P, n_k * P), 
    new_cov_tilde_new(P, P);
  
  new_amplitude = proposeNewNonNegativeValue(amplitude(k), amplitude_proposal_window);
    // std::exp(std::log(amplitude(k) + randn() * amplitude_proposal_window));
  if(new_amplitude < 1e-6) {
    return;
  }
  new_sub_block = calculateKthComponentKernelSubBlock(new_amplitude, length(k));
  new_cov_mat = constructCovarianceMatrix(n_k, k, new_sub_block);
  new_inv_cov_mat = invertComponentCovariance(n_k, noise(k),new_sub_block);
  new_cov_tilde_new = posteriorCovarianceParameter(new_cov_mat, new_inv_cov_mat);
  new_mu_tilde = posteriorMeanParameter(
    component_data, 
    new_cov_mat, 
    new_inv_cov_mat
  );
  
  new_score = hyperParameterKernel(
    new_amplitude, 
    mu.col(k), 
    new_mu_tilde, 
    new_cov_tilde_new
  );
  
  old_score = hyperParameterKernel(
    amplitude(k), 
    mu.col(k), 
    mu_tilde, 
    cov_tilde
  );

  acceptance_prob =  std::min(1.0, std::exp(new_score - old_score));
  accept = metropolisAcceptanceStep(acceptance_prob);
  if(accept) {
    amplitude(k) = new_amplitude;
    // amplitude_acceptance_count(k)++;
  }
};

void gp::sampleHyperParametersKthComponent(
    uword k, 
    uword n_k, 
    vec mu_tilde, 
    vec component_data,
    mat cov_tilde
) {
  sampleAmplitude(
    k,
    n_k,
    mu_tilde,
    component_data,
    cov_tilde
  );

  sampleLength(
    k,
    n_k,
    mu_tilde,
    component_data,
    cov_tilde
  );
  
  // sampleNoise(k, n_k, component_data);
  
  // if(randu() > 0.8) {
  //   Rcpp::Rcout << "\n\nNoise: " << noise(k);
  //   Rcpp::Rcout << "\nLength: " << length(k);
  //   Rcpp::Rcout << "\nAmplitude: " << amplitude(k);
  // }
};

double gp::noiseLogKernel(uword n_k, double noise, vec mean_vec, mat data) {
  double score = 0.0;
  // Rcpp::Rcout << "\nNoise kernel.\n";
  for(uword n = 0; n < n_k; n++) {
    score += pNorm(data.row(n).t(), mean_vec, noise * I_p);
  }
  score += pNorm(log(noise), 0, 1);
  // score += pHalfCauchy(noise, 0, 5);
  // Rcpp::Rcout << "\nNoise kernel done.\n";
  return score;
};

void gp::sampleNoise(uword k, uword n_k, mat component_data) {
  bool accept = false;
  double 
    noise_proposal_window = 75, 
      acceptance_prob = 0.0, 
      new_score = 0.0, 
      old_score = 0.0,
      new_noise = 0.0;

  new_noise = proposeNewNonNegativeValue(noise(k), noise_proposal_window);
  
  if(new_noise < 1e-6) {
    return;
  }
  
  new_score = noiseLogKernel(n_k, new_noise, mu.col(k), component_data);
  old_score = noiseLogKernel(n_k, noise(k), mu.col(k), component_data);
  
  acceptance_prob =  std::min(1.0, std::exp(new_score - old_score));
  accept = metropolisAcceptanceStep(acceptance_prob);
  if(accept) {
    noise(k) = new_noise;
    // amplitude_acceptance_count(k)++;
  }
};


// === Log-likelihoods =========================================================
// The log likelihood of a item belonging to each cluster.
arma::vec gp::itemLogLikelihood(arma::vec item) {
  
  double exponent = 0.0;
  arma::vec ll(K), dist_to_mean(P);
  mat noise_matrix(P, P), inverse_noise_matrix(P, P);
  ll.zeros();
  dist_to_mean.zeros();
  
  for(arma::uword k = 0; k < K; k++){
    // The exponent part of the MVN pdf
    dist_to_mean = item - mu.col(k);
    for(uword p = 0; p < P; p++) {
      // Normal log likelihood
      ll(k) += -0.5 *(
        log(2 * M_PI) 
        + log(noise(k)) 
        + std::pow(dist_to_mean(p), 2.0) / noise(k)
      );
    }
  }
  return(ll);
};

// The log likelihood of a item belonging to a specific cluster.
double gp::logLikelihood(arma::vec item, arma::uword k) {
  
  double ll = 0.0;
  arma::vec dist_to_mean(P);
  dist_to_mean.zeros();
  
  // The exponent part of the MVN pdf
  dist_to_mean = item - mu.col(k);
  for(uword p = 0; p < P; p++) {
    // Normal log likelihood
    ll += -0.5 *(
      log(2 * M_PI) 
    + log(noise(k)) 
    + std::pow(dist_to_mean(p), 2.0) / noise(k)
    );
  }
  return(ll);
};
