// categorical.cpp
// =============================================================================
// included dependencies
# include "logLikelihoods.h"
# include "categorical.h"

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp ;
using namespace arma ;

// =============================================================================
// categorical class

categorical::categorical(arma::uword _K, arma::uvec _labels, arma::mat _X) : 
  density(_K, _labels, _X) 
{
  
  n_cat.set_size(P);
  
  class_probabilities.set_size(P);
  // class_probabilities.zeros();
  
  Y = conv_to<umat>::from(X);
  
  initialiseParameters();
  
};


void categorical::initialiseParameters() {
  uvec Y_p(N), categories;
  mat call_prob_entry;
  
  for(uword p = 0; p < P; p++) {
    
    Y_p = Y.col(p);
    
    // Find the number of categories in the column
    categories = unique(Y_p); 
    n_cat(p) = categories.n_elem;
    
    // Create a matrix of 0's. This is a placeholder for the probability for 
    // each measurement within each cluster.
    call_prob_entry.set_size(n_cat(p), K);
    call_prob_entry.zeros();
    class_probabilities(p) = call_prob_entry;
    
    // Set the prior probability of being in any category empirically
    cat_prior_probability(p).set_size(n_cat(p));
    for(uword ii = 0; ii < n_cat(p); ii++) {
      cat_prior_probability(p)(ii) = accu(Y_p == ii) / N;
    }
  } 
  
  n_param = sum(n_cat) * K;
}

void categorical::sampleFromPriors() {
  for(uword p = 0; p < P; p++) {
    for(uword ii = 0; ii < n_cat(p); ii++) {
      for(uword k = 0; k < K; k++) {
        class_probabilities(p).row(ii) = arma::randg(
          K,
          arma::distr_param(cat_prior_probability(p)(ii), 1.0)
        );
      }
    }
  }
};

void categorical::sampleParameters(arma::umat members, arma::uvec non_outliers) {
  uvec relevant_indices;
  umat component_data;
  uword cat_count = 0;
  double concentration_n = 0.0;
  
  for(uword k = 0; k < K; k++) {
    
    // Find the items relevant to sampling the parameters
    relevant_indices = find((members.col(k) == 1) && (non_outliers == 1));
    
    component_data = Y.rows(relevant_indices);
    for(uword p = 0; p < P; p++) {
      
      for(uword ii = 0; ii < n_cat(p); ii++) {
        cat_count = accu(component_data.col(p) == ii);
        
        concentration_n = cat_prior_probability(p)(ii) + cat_count;
        
        class_probabilities(p).row(ii) = arma::randg(
          K, 
          arma::distr_param(concentration_n, 1.0)
        );
      }
    }
  }
};

double categorical::logLikelihood(arma::vec item, arma::uword k) {
  
  double ll = 0.0;
  uword x_p = 0;
  
  for(uword p = 0; p < P; p++) {
    x_p = item(p);
    ll += std::log(class_probabilities(p)(x_p, k));
  }
  
  return ll;
};

arma::vec categorical::itemLogLikelihood(arma::vec item) {
  vec ll(K);
  
  for(uword k = 0; k < K; k++) {
    ll(k) = logLikelihood(item, k);
  }
  return ll;
};
