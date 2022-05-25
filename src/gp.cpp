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
  
  noise_acceptance_count.set_size(K);
  noise_acceptance_count.zeros();
  
  length_acceptance_count.set_size(K);
  length_acceptance_count.zeros();
  
  amplitude_acceptance_count.set_size(K);
  amplitude_acceptance_count.zeros();
  
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

double gp::sampleAmplitudePriorDistribution(bool logNorm, double threshold) {
  double x = 0.0;
  if(logNorm) {
    x = std::exp(randn< double >());
  } else {
    x = rHalfCauchy(0, 5);
  }
  if(x < threshold) {
    x = sampleAmplitudePriorDistribution(logNorm, threshold);
  }
  return x;
};

double gp::noisePriorLogDensity(double x, bool logNorm) {
  double y = 0.0;
  if(logNorm) {
    y = pNorm(log(x), 0, 1);
  } else {
    y = pHalfCauchy(x, 0, 5, true);
  }
  return y;
}

double gp::ampltiduePriorLogDensity(double x, bool logNorm) {
  double y = 0.0;
  if(logNorm) {
    y = pNorm(log(x), 0, 1);
  } else {
    y = pHalfCauchy(x, 0, 5, true);
  }
  return y;
}

double gp::lengthPriorLogDensity(double x, bool logNorm) {
  double y = 0.0;
  if(logNorm) {
    y = pNorm(log(x), 0, 1);
  } else {
    y = pHalfCauchy(x, 0, 5, true);
  }
  return y;
}

double gp::sampleLengthPriorDistribution(bool logNorm, double threshold) {
  double x = 0.0;
  if(logNorm) {
    x = std::exp(randn< double >());
  } else {
    x = rHalfCauchy(0, 5);
  }
  if(x < threshold) {
    x = sampleLengthPriorDistribution(logNorm, threshold);
  }
  return x;
};

double gp::sampleNoisePriorDistribution(bool logNorm, double threshold) {
  double x = 0.0;
  if(logNorm) {
    x = std::exp(randn< double >());
  } else {
    x = rHalfCauchy(0, 5);
  }
  // x = std::exp(randn< double >());
  if(x < threshold) {
    x = sampleNoisePriorDistribution(threshold);
  }
  return x;
};

void gp::sampleKthComponentHyperParameterPrior(uword k, bool logNorm) {
  amplitude(k) = sampleAmplitudePriorDistribution(logNorm);
  length(k) = sampleLengthPriorDistribution(logNorm);
  noise(k) = sampleNoisePriorDistribution();
};

void gp::sampleHyperParameterPriors() {
  for(uword k = 0; k < K; k++) {
    sampleKthComponentHyperParameterPrior(k, logNormPriorUsed);
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
  for(uword ii = 0; ii < P; ii++) {
  // std::for_each(
  //   std::execution::par,
  //   P_inds.begin(),
  //   P_inds.end(),
  //   [&](uword ii) {
    sub_block(ii, ii) = amplitude;
    for(uword jj = ii + 1; jj < P; jj++) {
      sub_block(ii, jj) = squaredExponentialFunction(
        amplitude, 
        length, 
        ii, 
        jj
      );
      
      // if(sub_block(ii, jj) < kernel_subblock_threshold) {
      //   sub_block(ii, jj) = 0.0;
      //   sub_block(jj, ii) = 0.0;
      //   break;
      // }
      sub_block(jj, ii) = sub_block(ii, jj);
    }
  }
  // );
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
  Z_k = inv_sympd(Q_k);
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
  mu_tilde = covariance_matrix.rows(P_inds)
    * inverse_covariance_matrix
    * data;
    
  // for(uword p = 0; p < P; p++) {
  //   mu_tilde(p) = as_scalar(
  //     covariance_matrix.row(p)
  //   * inverse_covariance_matrix
  //   * data
  //   );
  // }
  return mu_tilde;
};

// vec gp::posteriorMeanParameter(uword n_k, double noise, mat data, mat inv_cov_mat) {
//   vec mu_tilde(P), sample_mean(P);
//   sample_mean = mean(data).as_col();
//   mu_tilde = n_k / noise * inv_cov_mat * sample_mean;
//   return mu_tilde;
// };

uvec gp::relevantIndices(uword ii, uword P) {
  uvec inds_1, inds_2, inds(P);
  
  inds_1 = regspace< uvec >(ii, -1, 0);
  inds = inds_1;
  if(ii < P) {
    inds_2 = regspace< uvec >(1, 1, P - ii - 1);
    inds = join_cols(inds_1, inds_2);
  }
  return inds;
}

// double gp::blockVectorMultiplication(rowvec a, mat B, uword ii, uword jj, uword N, uword P) {
//   double lambda = 0.0, lambda_comp = 0.0;
//   uvec lambda_block_inds(N), unique_inds(P), col_0 = zeros<uvec>(1);
//   vec a_used(P), b_used(P);
//   mat lambda_block(N, N);
//   
//   if(N > 1) {
//     // lambda_block_inds = regspace< uvec >(jj * N, (jj + 1) * N - 1); // seq((jj - 1) * N + 1, jj * N, 1)
//     lambda_block_inds = regspace< uvec >(jj, P, (N * P  - 1) + jj); // seq((jj - 1) * N + 1, jj * N, 1)
//     lambda_block = B.rows(lambda_block_inds);
//     lambda = lambda_block(0, 0) - lambda_block(1, 0);
//     lambda_block.diag() -= lambda;
//     B.rows(lambda_block_inds) = lambda_block;
//     // lambda_comp = a(jj * N) * lambda;
//     lambda_comp = a(jj) * lambda;
//   }
//   
//   // unique_inds = regspace< uvec >(0, N, (N * P) - 1);
//   unique_inds = P_inds;
//   a_used = a.elem(unique_inds);
//   b_used = B.submat(unique_inds, col_0);
//   
//   return N * accu(a_used % b_used) + lambda_comp;
// }
// 
// arma::mat gp::firstProduct(mat A, mat B, uword n_k) {
//   uword jj_bound = 0;
//   double new_entry = 0.0;
//   uvec 
//     N_inds(n_k),
//     P_inds(P), 
//     current_elements(4),
//     first_element(1),
//     second_element(1),
//     third_element(1),
//     fourth_element(1),
//     j_inds,
//     rel_cols_inds(n_k),
//     first_col_inds(n_k),
//     second_col_inds(n_k),
//     third_col_inds(n_k),
//     fourth_col_inds(n_k);
//   
//   mat new_mat(P, n_k * P);
//   new_mat.zeros();
//   
//   current_elements.zeros();
//   
//   N_inds = regspace< uvec >(0, n_k - 1);
//   P_inds = regspace< uvec >(0, P - 1);
//   
//   for(auto & ii : P_inds) {
//     first_element.fill(ii);
//     fourth_element.fill(P - ii - 1);
//     
//     jj_bound = std::min(ii + 1, (uword) (ceil(P / 2)));
//     for(uword jj = 0; jj <  jj_bound; jj++) {
//       second_element.fill(jj);
//       third_element.fill(P - jj - 1);
//       
//       rel_cols_inds = regspace< uvec >(jj, P, n_k * (P - 1) + jj);
//       first_col_inds = rel_cols_inds;
//       second_col_inds = regspace< uvec >(ii, P, n_k * (P - 1) + ii);
//       third_col_inds = regspace< uvec >(P - ii - 1, P, n_k * P - 1);
//       fourth_col_inds = regspace< uvec >(P - jj - 1, P, n_k * P - 1);
//         
//       // rel_cols_inds = regspace< uvec >(n_k * jj, (n_k * (jj + 1)) - 1);
//       // first_col_inds = rel_cols_inds;
//       // second_col_inds = regspace< uvec >(n_k * ii, (n_k * (ii + 1)) - 1);
//       // third_col_inds = regspace< uvec >(n_k * (P - ii - 1), n_k * (P - ii) - 1);
//       // fourth_col_inds = regspace< uvec >(n_k * (P - jj - 1), n_k * (P - jj) - 1);
//       
//       new_entry = blockVectorMultiplication(
//         A.row(ii),
//         B.cols(rel_cols_inds), 
//         ii, 
//         jj, 
//         n_k, 
//         P
//       );
// 
//       new_mat.submat(first_element, first_col_inds).fill(new_entry);
//       new_mat.submat(second_element, second_col_inds).fill(new_entry);
//       new_mat.submat(third_element, third_col_inds).fill(new_entry);
//       new_mat.submat(fourth_element, fourth_col_inds).fill(new_entry);
//     }
//   }
//   return new_mat;
// };

double gp::blockVectorMultiplication(rowvec a, mat B, uword ii, uword jj, uword N, uword P) {
  double lambda = 0.0, lambda_comp = 0.0;
  uvec lambda_block_inds(N), unique_inds(P), col_0 = zeros<uvec>(1);
  vec a_used(P), b_used(P);
  mat lambda_block(N, N);
  
  if(N > 1) {
    lambda_block_inds = regspace< uvec >(jj, P, N * P  - 1);
    lambda_block = B.rows(lambda_block_inds);
    lambda = lambda_block(0, 0) - lambda_block(1, 0);
    lambda_block.diag() -= lambda;
    B.rows(lambda_block_inds) = lambda_block;
    lambda_comp = a(jj) * lambda;
  }
  
  unique_inds = regspace< uvec >(0, 1, P - 1);
  a_used = a.elem(unique_inds);
  b_used = B.submat(unique_inds, col_0);
  
  return N * accu(a_used % b_used) + lambda_comp;
};

double gp::blockVectorMultiplication(rowvec a, 
                                     mat B, 
                                     double lambda,
                                     uword ii, 
                                     uword jj, 
                                     uword N, 
                                     uword P) {
  double lambda_comp = a[jj] * lambda;
  uvec unique_inds(P), col_0 = zeros<uvec>(1);
  vec a_used(P), b_used(P);
  
  unique_inds = regspace< uvec >(0, 1, P - 1);
  a_used = a.elem(unique_inds);
  b_used = B.submat(unique_inds, col_0);
  
  return N * accu(a_used % b_used) + lambda_comp;
};


double gp::findLambda(mat B, uword N, bool testLambdas) {
  bool common_lambda = false;
  double lambda = 0.0;
  vec lambdas(P);
  uvec lambda_block_inds(P), lambda_test(P);
  mat lambda_block(P, P), diagonal_block(P, P);
  
  if(N == 1) {
    return 0.0;
  }
  
  lambda = B(0, 0) - B(P, 0);
    
  if(testLambdas) {
    lambda_block_inds = P_inds;
    lambda_block = B.submat(lambda_block_inds, lambda_block_inds);
    diagonal_block = B.submat(P + lambda_block_inds, lambda_block_inds);
    lambdas = lambda_block.diag() - diagonal_block.diag();
    lambda = lambdas(0);
    
    lambda_test.zeros();
    for(auto& l : lambdas) {
      lambda_test = 1 * doubleApproxEqual(l, lambda);
    }
    common_lambda = all(lambda_test == 1);
    
    if(! common_lambda) {
      Rcpp::Rcout << "\nLambda is NOT constant.";
    }
  }
  return lambda;
};

arma::mat gp::firstCovProduct(mat A, mat B, uword N) {
  uword jj_bound = 0;
  double new_entry = 0.0, lambda = 0.0;
  uvec 
    N_inds(N),
    P_inds(P), 
    current_elements(4),
    first_element(1),
    second_element(1),
    third_element(1),
    fourth_element(1),
    j_inds,
    rel_cols_inds(N),
    first_col_inds(N),
    second_col_inds(N),
    third_col_inds(N),
    fourth_col_inds(N);
  
  mat new_mat(P, N * P);
  new_mat.zeros();
  
  current_elements.zeros();
  
  // lambda = findLambda(B, N, false);
  
  N_inds = regspace< uvec >(0, N - 1);
  P_inds = regspace< uvec >(0, P - 1);
  
  for(uword ii = 0; ii < P; ii++) {
    
    first_element.fill(ii);
    fourth_element.fill(P - ii - 1);
    
    second_col_inds = regspace< uvec >(ii, P, N * P - 1);
    third_col_inds = regspace< uvec >(P - ii - 1, P, N * P - 1);
    
    jj_bound = std::min(ii + 1, P - ii);
    for(uword jj = 0; jj < jj_bound; jj++) {
      
      second_element.fill(jj);
      third_element.fill(P - jj - 1);
      
      rel_cols_inds = regspace< uvec >(jj, P, P * N - 1);
      first_col_inds = rel_cols_inds;
      fourth_col_inds = regspace< uvec >(P - jj - 1, P, N * P - 1);
      
      new_entry = blockVectorMultiplication(
        A.row(ii), 
        B.cols(rel_cols_inds), 
        // lambda,
        ii, 
        jj, 
        N, 
        P
      );
      
      new_mat.submat(first_element, first_col_inds).fill(new_entry);
      if(ii != jj) {
        new_mat.submat(second_element, second_col_inds).fill(new_entry);
      }
      if((ii + jj) != (P - 1)) {
        new_mat.submat(third_element, third_col_inds).fill(new_entry);
        new_mat.submat(fourth_element, fourth_col_inds).fill(new_entry);
      }
    }
  }
  return new_mat;
};


mat gp::posteriorCovarianceParameter(
    mat covariance_matrix,
    mat inverse_covariance_matrix) {
  mat cov_tilde(P, P);
  cov_tilde.zeros();
  
  cov_tilde = covariance_matrix.submat(P_inds, P_inds)
    - covariance_matrix.rows(P_inds)
    * inverse_covariance_matrix
    * covariance_matrix.cols(P_inds);

  // std::for_each(std::execution::par,
  //               P_inds.begin(),
  //               P_inds.end(),
  //               [&](uword ii) {
  // // for(uword ii = 0; ii < P; ii++) {
  // // if(ii >= P) {
  // //   Rcpp::Rcout << "\ni: " << ii;
  // //   // throw;
  // // }
  // 
  //   cov_tilde(ii, ii) = as_scalar(
  //     covariance_matrix(ii, ii)
  //   - covariance_matrix.row(ii)
  //   * inverse_covariance_matrix
  //   * covariance_matrix.col(ii)
  //   );
  // 
  //   for(uword jj = ii + 1; jj < P; jj++) {
  //     cov_tilde(ii, jj) = as_scalar(
  //        covariance_matrix(ii, jj)
  //        - covariance_matrix.row(ii)
  //          * inverse_covariance_matrix
  //          * covariance_matrix.col(jj)
  //     );
  //     cov_tilde(jj, ii) = cov_tilde(ii, jj);
  //   }
  // }
  // );
  
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

  bool not_invertible = false, not_symmetric = false;
  vec mu_tilde(P), data_vec = data.as_row().t(), eigval(P);
  mat
    cov_tilde(P, P), 
    covariance_matrix(n_k * P, n_k * P),
    inverse_covariance(n_k * P, n_k * P),
    inverse_covariance_comp(n_k * P, n_k * P),
    I_nkP(n_k * P, n_k * P),
    chol_cov(P, P),
    stochasticity = mvnrnd(zeros<vec>(P), eye(P, P)),
    first_product(P, n_k * P),
    rel_cov_mat(P, n_k * P),
    final_prod(P, P);
  I_nkP = eye(n_k * P, n_k * P);
  
  // Objects related to the covariance function
  covariance_matrix = constructCovarianceMatrix(n_k, k, kernel_sub_block.slice(k));
  rel_cov_mat = covariance_matrix.rows(P_inds);
  
  inverse_covariance = invertComponentCovariance(n_k, noise(k), kernel_sub_block.slice(k));
  
  // The product of the covariance matrix and the inverse as used in sampling 
  // parameters.
  first_product = firstCovProduct(rel_cov_mat, inverse_covariance, n_k);
  final_prod = n_k * (first_product.cols(P_inds) * rel_cov_mat.cols(P_inds).t());
  
  // Mean and covariance hyperparameter
  mu_tilde = first_product * data_vec;
  cov_tilde = rel_cov_mat.cols(P_inds) - final_prod; // first_product * rel_cov_mat.t();

  mat original_cov_tilde = posteriorCovarianceParameter(covariance_matrix, inverse_covariance);
  
  bool same_cov = approx_equal(cov_tilde, original_cov_tilde, "reldiff", 0.002);
  if(! same_cov) {
    Rcpp::Rcout << "\n\nDIfferent covariances being acquired.\n";
  //   Rcpp::Rcout << "\nCov (original):\n" << cov_tilde.head_rows(3);
  //   Rcpp::Rcout << "\nCov (new):\n" << cov_tilde2.head_rows(3);
  }
  
  // Rcpp::Rcout << "\n\n\nFirst prod:\n" << first_product.cols(P_inds);
  // Rcpp::Rcout << "\n\nFinal prod:\n" << final_prod.cols(P_inds);
  // Rcpp::Rcout << "\n\nFinal cov:\n" << cov_tilde.cols(P_inds);
  
  // If our covariance matrix is poorly behaved (i.e. non-invertible), add a 
  // small constant to the diagonal entries
  
  not_symmetric = ! cov_tilde.is_symmetric();
  if(not_symmetric) {
    mat new_cov(P, P), u_cov = trimatu(cov_tilde, 1);
    new_cov = u_cov + u_cov.t();
    new_cov.diag() = cov_tilde.diag();
    cov_tilde = new_cov;
  }
  
  eigval = eig_sym( cov_tilde );
  not_invertible = min(eigval) < 1e-5;
  
  mat small_identity = I_p;
  if(not_invertible) {
    small_identity *= 1e-5;
    cov_tilde += small_identity;
  }
  
  // cov_tilde = inverse_covariance;
  
  // Rcpp::Rcout << "\nSampled the mean function.";
  // Rcpp::Rcout << "\n\nMean hyper:\n" << mu_tilde;
  // Rcpp::Rcout << "\n\nCov hyper:\n" << cov_tilde;
  
  // Rcpp::Rcout << "\nSampling mean function.\n";
  // mu.col(k) = mvnrnd(mu_tilde, cov_tilde);
  
  chol_cov = chol(cov_tilde);
  // mat stochasticity = mvnrnd(zeros<vec>(P), eye(P, P));
  mu.col(k) = mu_tilde + chol_cov * stochasticity;
  
  
  if((samplingCount % sampleHypersFrequency) == 0) {
    // Rcpp::Rcout << "\nSampling hyperparameters.\n";
    sampleHyperParametersKthComponent(
      k,
      n_k,
      mu_tilde,
      data_vec,
      cov_tilde
    );
    
    sampleNoise(k, n_k, data);
  }
  
  
  // return mvnrnd(mu_tilde, cov_tilde);
};

void gp::sampleKthComponentParameters(uword k, umat members, uvec non_outliers) {
  
  // Find the items relevant to sampling the parameters
  uvec rel_inds = find((members.col(k) == 1) && (non_outliers == 1));
  // vec component_data_vec;
  mat component_data;
  
  // Find how many labels have the value
  uword n_k = rel_inds.n_elem;
  
  if(n_k > 0){
    
    // Rcpp::Rcout << "\nSampling from posterior.\n";
    
    component_data.set_size(n_k);
    
    // Component data
    component_data = X.rows( rel_inds ) ;
    
    // sampleHyperParameters();
    // mu.col(k) = 
    sampleMeanPosterior(k, n_k, component_data);
  } else {
    
    // Rcpp::Rcout << "\n\nEmpty component!\n";
    // Rcpp::Rcout << "\nSampling from prior.\n";
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
  
  // Rcpp::Rcout << "\nAmplitude:\n" << amplitude.t();
  // Rcpp::Rcout << "\n\nLength:\n" << length.t();
  // Rcpp::Rcout << "\n\nNoise:\n" << noise.t();

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
  // if(randu() > 0.9) {
  samplingCount++;
  // if((samplingCount % 5) == 0) {
  //   Rcpp::Rcout << "\n\nSampling count: " << samplingCount;
  //   Rcpp::Rcout << "\nNumber of hyper propsals: " << floor(samplingCount / sampleHypersFrequency);
  //   Rcpp::Rcout << "\n\nNoise acceptance count:\n" << accu(noise_acceptance_count.t()) / 3.0;
  //   Rcpp::Rcout << "\n\nLength acceptance count:\n" << accu(length_acceptance_count.t()) / 3.0;
  //   Rcpp::Rcout << "\n\nAmplitude acceptance count:\n" << accu(amplitude_acceptance_count.t()) / 3.0;
  //   
  //   Rcpp::Rcout << "\n\n\nNoise: " << noise.t();
  //   Rcpp::Rcout << "\n\nLength: " << length.t();
  //   Rcpp::Rcout << "\n\nAmplitude: " << amplitude.t();
  // }

};

// === Hyper-parameters ========================================================

// Need to sample the hyperparameter and recalculate the mu tilde / cov tilde
double gp::hyperParameterLogKernel(
    double hyper, 
    vec mu_k, 
    vec mu_tilde, 
    mat cov_tilde, 
    bool logNorm
  ) {
  double score = 0.0;
  // vec mean_diff = mu_k - mu_tilde;

  // if(! cov_tilde.is_sympd()) {
  //   Rcpp::Rcout << "\n\nNot positive semi-definite.\n";
  // }
  score = pNorm(mu_k, mu_tilde, cov_tilde);
  // score = log(2 * M_PI)
  //   + log_det_sympd(cov_tilde)
  //   + as_scalar(mean_diff.t() * inv_sympd(cov_tilde) * mean_diff);
  // score *= -0.5;
  
  if(logNorm) {
    score += pNorm(log(hyper), 0, 1);
  } else {
    score += pHalfCauchy(hyper, 0, 5);
  }

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
//     length_proposal_window = 150, 
//       acceptance_prob = 0.0, 
//       new_score = 0.0, 
//       old_score = 0.0,
//       new_length = 0.0;
//   
//   new_length = std::exp(std::log(length(k) + randn() * length_proposal_window));
//   new_score = hyperParameterLogKernel(new_length, mu.col(k), mu_tilde, cov_tilde);
//   old_score = hyperParameterLogKernel(length(k), mu.col(k), mu_tilde, cov_tilde);
// 
//   acceptance_prob =  std::min(1.0, std::exp(new_score - old_score));
//   accept = metropolisAcceptanceStep(acceptance_prob);
//   if(accept) {
//     length(k) = new_length;
//     // length_acceptance_count(k)++;
//   }
// };



void gp::sampleLength(
    uword k, 
    uword n_k, 
    vec mu_tilde, 
    vec component_data, 
    mat cov_tilde
) {
  bool accept = false;
  double 
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
  if(rcond(new_cov_mat) < 1e-2) {
    return;
  }
  
  new_inv_cov_mat = invertComponentCovariance(n_k, noise(k),new_sub_block);
  new_cov_tilde_new = posteriorCovarianceParameter(new_cov_mat, new_inv_cov_mat);
  new_mu_tilde = posteriorMeanParameter(
    component_data, 
    new_cov_mat, 
    new_inv_cov_mat
  );
  
  new_score = hyperParameterLogKernel(
    new_length, 
    mu.col(k), 
    new_mu_tilde, 
    new_cov_tilde_new,
    logNormPriorUsed
  );
  
  old_score = hyperParameterLogKernel(
    length(k), 
    mu.col(k), 
    mu_tilde, 
    cov_tilde,
    logNormPriorUsed
  );
  
  acceptance_prob =  std::min(1.0, std::exp(new_score - old_score));
  accept = metropolisAcceptanceStep(acceptance_prob);
  if(accept) {
    length(k) = new_length;
    length_acceptance_count(k)++;
  }
};

double gp::proposeNewNonNegativeValue(double x, double window) {
  return randg( distr_param( x * window, 1.0 / window) );
  // return std::exp(std::log(x) + randn() * window);
}

void gp::sampleAmplitude(uword k, uword n_k, vec mu_tilde, vec component_data, mat cov_tilde) {
  bool accept = false;
  double 
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
  if(rcond(new_cov_mat) < 1e-2) {
    return;
  }
  
  new_inv_cov_mat = invertComponentCovariance(n_k, noise(k),new_sub_block);
  new_cov_tilde_new = posteriorCovarianceParameter(new_cov_mat, new_inv_cov_mat);
  new_mu_tilde = posteriorMeanParameter(
    component_data, 
    new_cov_mat, 
    new_inv_cov_mat
  );
  
  new_score = hyperParameterLogKernel(
    new_amplitude, 
    mu.col(k), 
    new_mu_tilde, 
    new_cov_tilde_new,
    logNormPriorUsed
  );
  
  old_score = hyperParameterLogKernel(
    amplitude(k), 
    mu.col(k), 
    mu_tilde, 
    cov_tilde,
    logNormPriorUsed
  );

  acceptance_prob =  std::min(1.0, std::exp(new_score - old_score));
  accept = metropolisAcceptanceStep(acceptance_prob);
  if(accept) {
    amplitude(k) = new_amplitude;
    amplitude_acceptance_count(k)++;
  }
};

void gp::sampleCovHypers(uword k, uword n_k, vec mu_tilde, vec component_data, mat cov_tilde) {
  bool accept = false;
  double 
    acceptance_prob = 0.0, 
    new_score = 0.0, 
    old_score = 0.0,
    new_amplitude = 0.0,
    new_length = 0.0;
  
  vec new_mu_tilde(P);
  mat 
    new_sub_block(P, P), 
    new_cov_mat(n_k * P, n_k * P),
    new_inv_cov_mat(n_k * P, n_k * P), 
    new_cov_tilde(P, P);
  
  new_amplitude = proposeNewNonNegativeValue(amplitude(k), amplitude_proposal_window);
  new_length = proposeNewNonNegativeValue(length(k), length_proposal_window);
  // std::exp(std::log(amplitude(k) + randn() * amplitude_proposal_window));
  if( (new_amplitude < 1e-6) || (new_length < 1e-6) ) {
    return;
  }
  new_sub_block = calculateKthComponentKernelSubBlock(new_amplitude, new_length);
  new_cov_mat = constructCovarianceMatrix(n_k, k, new_sub_block);
  
  if(! new_cov_mat.is_sympd()) {
    return;
  }
  
  new_inv_cov_mat = invertComponentCovariance(n_k, noise(k), new_sub_block);
  new_cov_tilde = posteriorCovarianceParameter(new_cov_mat, new_inv_cov_mat);
  new_mu_tilde = posteriorMeanParameter(
    component_data, 
    new_cov_mat, 
    new_inv_cov_mat
  );
  
  new_score = log(2 * M_PI)
    + log_det_sympd(new_cov_tilde)
    + as_scalar((mu.col(k) - new_mu_tilde).t() * inv_sympd(new_cov_tilde) * (mu.col(k) - new_mu_tilde));
  new_score *= -0.5;
  new_score += pNorm(log(new_amplitude), 0, 1);
  new_score += pNorm(log(new_length), 0, 1);

  old_score = log(2 * M_PI)
    + log_det_sympd(cov_tilde)
    + as_scalar((mu.col(k) - mu_tilde).t() * inv_sympd(cov_tilde) * (mu.col(k) - mu_tilde));
  old_score *= -0.5;
  old_score += pNorm(log(amplitude(k)), 0, 1);
  old_score += pNorm(log(length(k)), 0, 1);
    
    
  // new_score = hyperParameterLogKernel(
  //   new_amplitude, 
  //   mu.col(k), 
  //   new_mu_tilde, 
  //   new_cov_tilde,
  //   logNormPriorUsed
  // );
  // 
  // old_score = hyperParameterLogKernel(
  //   amplitude(k), 
  //   mu.col(k), 
  //   mu_tilde, 
  //   cov_tilde,
  //   logNormPriorUsed
  // );
  
  acceptance_prob =  std::min(1.0, std::exp(new_score - old_score));
  accept = metropolisAcceptanceStep(acceptance_prob);
  if(accept) {
    amplitude(k) = new_amplitude;
    length(k) = new_length;
    amplitude_acceptance_count(k)++;
    length_acceptance_count(k)++;
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
  
  // sampleCovHypers(
  //   k,
  //   n_k,
  //   mu_tilde,
  //   component_data,
  //   cov_tilde
  // );
  
  // sampleNoise(k, n_k, component_data);
  
  // if(randu() > 0.8) {
  //   Rcpp::Rcout << "\n\nNoise: " << noise(k);
  //   Rcpp::Rcout << "\nLength: " << length(k);
  //   Rcpp::Rcout << "\nAmplitude: " << amplitude(k);
  // }
};

double gp::noiseLogKernel(uword n_k, double noise, vec mean_vec, mat data) {
  double score = 0.0, prior_contribution = 0.0;
  // Rcpp::Rcout << "\nNoise kernel.\n";
  for(uword n = 0; n < n_k; n++) {
    score += pNorm(data.row(n).t(), mean_vec, noise * I_p);
  }
  prior_contribution += noisePriorLogDensity(noise, logNormPriorUsed); 
  
  score += prior_contribution;
  // score += pNorm(log(noise), 0, 1);
  // score += pHalfCauchy(noise, 0, 5);
  return score;
};

void gp::sampleNoise(uword k, uword n_k, mat component_data) {
  bool accept = false;
  double 
      acceptance_prob = 0.0, 
      new_score = 0.0, 
      old_score = 0.0,
      new_noise = 0.0;

  new_noise = proposeNewNonNegativeValue(noise(k), noise_proposal_window);
  
  if(new_noise < 1.0e-5) {
    return;
  }
  
  new_score = noiseLogKernel(n_k, new_noise, mu.col(k), component_data);
  old_score = noiseLogKernel(n_k, noise(k), mu.col(k), component_data);
  
  acceptance_prob =  std::min(1.0, std::exp(new_score - old_score));
  accept = metropolisAcceptanceStep(acceptance_prob);
  if(accept) {
    noise(k) = new_noise;
    noise_acceptance_count(k)++;
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
