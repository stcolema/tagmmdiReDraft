// mdi.cpp
// =============================================================================
// included dependencies
# include <RcppParallel.h>
# include <RcppArmadillo.h>
# include <execution>
# include "mdi.h"

using namespace arma ;

// =============================================================================
// MDI class

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp ;
using namespace arma ;


mdiModelAlt::mdiModelAlt(
  arma::field<arma::mat> _X,
  uvec _mixture_types,
  uvec _outlier_types,
  arma::uvec _K,
  arma::umat _labels,
  arma::umat _fixed
) {
  
  // These are used locally in some constructions
  uword k = 0, col_ind = 0;
  
  // Mixture types
  mixture_types = _mixture_types;
  
  // The type of outlier component used
  outlier_types = _outlier_types;
  
  // First allocate the inputs to their saved, class
  
  // The number of datasets modelled
  L = size(_X)(0);
  
  // The number of pairwise combinations
  if(L > 1) {
    LC2 = L * (L - 1) / 2;
  }
  
  // Used to check all datasets have matching number of rows
  uvec N_check(L);
  
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
  
  // // We want to track which combinations should be unweighed and by what phi.
  // // This object will be used in calculating the normalising constant (Z), the
  // // cluster weights (gammas) and the correlation coefficient (phis)
  // // along with the phi_indicator matrix.
  // comb_inds.set_size(K_to_the_L, L);
  // comb_inds.zeros();
  
  // Concentration mass for each view
  mass.set_size(L);
  mass.zeros();
  
  // A weight vector for each dataset. Note that for ease of manipulations we
  // are using K_max and not K(l); this is to avoid ragged fields of vectors.
  w.set_size(K_max, L);
  w.zeros();
  
  // The ``correlation'' coefficient between each dataset clustering. Use the
  // size that best corresponds to phi_indicator. Possibly a row vector would
  // be easier.
  phis.set_size(LC2);
  phis.zeros();
  
  // Initialise the different objects used to interact with the phis
  initialisePhis();
  
  // The initial labels
  labels = _labels;
  
  // The data
  X = _X;
  
  // Iterate over the number of datasets (currently must be 2)
  for(uword l = 0; l < L; l++) {
    N_check(l) = _X(l).n_rows;
  }
  
  if(any(N_check != N_check(0))) {
    Rcpp::Rcerr << "\n\nDatasets not matching in number of rows.\n";
    throw;
  }
  
  // The number of samples in each dataset
  N = N_check(0);
  
  N_inds = linspace< uvec >(0, N - 1, N);
  L_inds = linspace< uvec >(0, L - 1, L);
  
  // Quantities used in calculating phi shape
  N_ones = regspace(0, N);
  N_log_factorial_vec = log(N_ones);
  N_log_factorial_vec(0) = 0.0;
  N_log_factorial_vec = cumsum(N_log_factorial_vec);

  // The members of each cluster across datasets. Each slice is a binary matrix
  // of the members of the kth class across the datasets.
  members.set_size(N, K_max, L);
  members.zeros();
  
  
  // These are used in t-adjusted mixtures. In all other models they should
  // never be changed.
  non_outliers.set_size(N, L);
  non_outliers.ones();
  
  // The observed labels
  fixed = _fixed;
  
  // The number of fixed components (i.e. non-symbolic labels) and unfixed
  // components. Used when aligning components across views.
  K_fixed.set_size(L);
  K_unfixed.set_size(L);
  
  uvec fixed_l, fixed_labels, labels_l, fixed_components;
  
  for(uword l = 0; l < L; l++){
    
    fixed_l = find(fixed.col(l) == 1);
    labels_l = labels.col(l);
    
    
    // fixed_ind(l) = fixed_l;
    fixed_labels = labels_l(fixed_l);
    fixed_components = unique(fixed_labels);
    K_fixed(l) = fixed_components.n_elem;
    K_unfixed(l) = K(l) - K_fixed(l);
  }
  
  complete_likelihood_vec = zeros< vec >(L);
  
  // Declare the view-specific mixtures
  initialiseMDI();
};


void mdiModelAlt::initialisePhis() {
  
  uword k = 0, col_ind = 0;
  
  // // We want to track which combinations should be unweighed and by what phi.
  // // This object will be used in calculating the normalising constant (Z), the
  // // cluster weights (gammas) and the correlation coefficient (phis)
  // // along with the phi_indicator matrix.
  // comb_inds.set_size(K_to_the_L, L);
  // comb_inds.zeros();
  
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
  one_to_K = linspace<uvec>(0, K_max - 1, K_max);
  one_to_L = linspace<uvec>(0, L - 1, L);
  
  // // The matrix used to construct the rate for sampling the cluster weights
  // KL_powers.set_size(L);
  // for(uword l = 0; l < L; l++) {
  //   KL_powers(l) = std::pow(K_max, l);
  //   for(uword i = 0; i < K_to_the_L; i++){
  //     
  //     // We want the various combinations of the different gammas / cluster weights.
  //     // This format makes it relatively easy to figure out the upscaling too
  //     // (see phi_indicator below).
  //     k = ((double) i / (double) KL_powers(l));
  //     k = k % K_max;
  //     comb_inds(i, l) = k;
  //   }
  // }
  // 
  // // Drop any rows that contain weights for clusters that shouldn't be 
  // // modelled (i.e. if unique K_l are used)
  // for(uword l = 0; l < L; l++) {
  //   rows_to_shed = find(comb_inds.col(l) >= K(l));
  //   comb_inds.shed_rows(rows_to_shed);
  // }
  // 
  // // The final number of combinations
  // n_combinations = comb_inds.n_rows;
  // 
  // // Now construct a matrix to record which phis are upweighing which weight
  // // products, via an indicator matrix. This matrix has a column for each phi
  // // (ncol = LC2) and a row for each combination (nrow = n_combinations).
  // // This is working with the combi_inds object above. That indicated which 
  // // weights to use, this indicates the corresponding up weights (e.g.,
  // // we'd expect the first row to be all ones as all weights are for the first
  // // component in each dataset, similarly for the last row).
  // phi_indicator.set_size(n_combinations, LC2);
  // phi_indicator.zeros();
  
  // Rcpp::Rcout << "\n\nPhi indicator declared.";
  
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
      // phi_indicator.col(col_ind) = (comb_inds.col(l) == comb_inds.col(m));
      
      // Record which column index maps to which phi
      phi_ind_map(m, l) = col_ind;
      phi_ind_map(l, m) = col_ind;
      
      // The column index is awkward, this is the  easiest solution
      col_ind++;
      
    }
  }
  
  // // We use the transpose a surprising amount to ensure correct typing
  // phi_indicator_t = phi_indicator.t();
  // 
  // // And we often multiply by doubles, so convert to a matrix of doubles.
  // phi_indicator_t_mat = conv_to<mat>::from(phi_indicator_t);
  
};

// This wrapper to declare the mixtures at the dataset level is kept as its
// own separate function to make a semi-supervised class easier
void mdiModelAlt::initialiseMixtures() {

  // Initialise the collection of mixtures
  mixtures.reserve(L);
  for(uword l = 0; l < L; l++) {
    mixtures.push_back(
      std::make_unique<mixtureModel>(
        mixture_types(l),
        outlier_types(l),
        K(l),
        labels.col(l),
        fixed.col(l),
        X(l)
      )
    );
    
    // We have to pass this back up
    non_outliers.col(l) = mixtures[l]->non_outliers;
  }
};


// double mdiModelAlt::calcWeightRate(uword lstar, uword kstar) {
//   // The rate for the (k, l)th cluster weight is the sum across all of the clusters
//   // in each dataset (excluding the lth) of the product of the kth cluster
//   // weight in each of the L datasets (excluding the lth) upweigthed by
//   // the pairwise correlation across all LC2 dataset combinations.
//   
//   // The rate we return.
//   double rate = 0.0;
//   
//   // A chunk of these objects are used as our combinations matrix includes the
//   // information for k != kstar and l != lstar, so we can shed some data.
//   // Probably some of these can be dropped and calculated once at the class // [[Rcpp::export]]
//   // level, but to allow K_l != K_m I suspect it will be different for each
//   // dataset (I haven't done the maths though) and thus the objects might be
//   // awkward
//   uword n_used = 0;
//   uvec relevant_inds;
//   vec weight_products, phi_products;
//   umat relevant_combinations;
//   mat relevant_phis, phi_prod_mat;
//   
//   relevant_inds = find(comb_inds.col(lstar) == kstar);
//   relevant_combinations = comb_inds.rows(relevant_inds);
//   relevant_phis =  phi_indicator_t_mat.cols(relevant_inds);
//   n_used = relevant_combinations.n_rows;
//   
//   weight_products.ones(n_used);
//   phi_products.ones(n_used);
//   
//   // The phi products (should be a matrix of 0's and phis)
//   phi_prod_mat = relevant_phis.each_col() % phis;
//   
//   // Add 1 to each entry to have the object ready to be multiplied
//   phi_prod_mat++;
//   
//   // Vector of the products, this should have the \prod (1 + \phi_{lm} ind(k_l = k_m))
//   // ready to multiply by the weight products
//   phi_products = prod(phi_prod_mat, 0).t();
//   
//   vec w_l(K_max);
//   for(uword l = 0; l < L; l++) {
//     if(l != lstar){
//       w_l = w.col(l);
//       weight_products = weight_products % w_l.elem(relevant_combinations.col(l));
//     }
//   }
//   
//   // The rate for the gammas
//   rate = v * accu(weight_products % phi_products);
//   
//   return rate;
// };

double mdiModelAlt::calcWeightRateNaiveSingleIteration(uword k, uword l, uvec current_ks) {
  double rate = 1.0;
  for(uword m = 0; m < L; m++) {
    if(m != l) {
      rate *= w(current_ks(l), l);
    }
  }
  for(uword m1 = 0; m1 < L - 1; m1++) {
    for(uword m2 = m1 + 1; m2 < L; m2++) {
      rate *= (1.0 + phis(phi_ind_map(m1, m2)) * (current_ks(m1) == current_ks(m2)));
    }
  }
  return rate;
};

double mdiModelAlt::calcWeightRateNaive(uword k, uword l) {
  uword K_comb = 0, m = 0;
  double rate = 0.0;
  uvec weight_ind(L), K_cumprod(L - 1), for_loop_inds(L), K_rel(L), K_rel_cum(L - 1);
  
  // We begin at the 0th component for each view and the kth for the view being 
  // updated
  weight_ind.zeros();
  weight_ind(l) = k;
  
  // The number of components in each view bar the vth
  K_rel.zeros();
  K_rel_cum.zeros();
  K_rel = K;
  K_rel.shed_row(l);
  
  // We will use this in the for loop to control accessing and updating of 
  // weights
  for_loop_inds = regspace<uvec>(0,  1,  L - 1);
  for_loop_inds.shed_row(l);
  
  // We have shed an entry (leaving L-1) and we do not need the final entry for 
  // the cumulative check as it should fill once
  if(L > 2) {
    K_rel_cum(span(1, L - 2)) = K_rel(span(0, L - 3));
  }
  
  // Set the 0th entry to 1.0 for the cumulative product
  K_rel_cum(0) = 1;
  
  // The cumulative product of the number of components in each view is used to
  // control updating weight indices
  K_cumprod = cumprod(K_rel_cum);
  
  // This is the number of summations to perform
  K_comb = prod(K_rel);
  
  // Rcpp::Rcout << "\n\nWEIGHT RATE\nView: " << l << "\nComponent: " << k << "\n";
  for(uword ii = 0; ii < K_comb; ii++) {
    // Rcpp::Rcout << "\n\ni: " << ii;
    // Rcpp::Rcout << "\nWeight indices:\n " << weight_ind.t();
    rate += calcWeightRateNaiveSingleIteration(k, l, weight_ind);
    for(uword jj = 0; jj < (L - 1); jj++) {
      // Which view is actually being updated (skipping the vth)
      m = for_loop_inds(jj);
      
      // If in first view we always update the weight index. Otherwise we check 
      // if we are a multiple of the cumulative product of the number of 
      // components in the preceding views and update the index if so.
      if(jj == 0) {
        weight_ind(m)++;
      } else {
        if((((ii + 1) % K_cumprod(jj)) == 0) && (ii != 0)) {
          weight_ind(m)++;
        }
      }
      if(weight_ind(m) == K(l)) {
        weight_ind(m) = 0;
      }
    }
  }
  rate *= v;
  return rate;
};

double mdiModelAlt::calcPhiRateNaiveSingleIteration(uword view_i, uword view_j, uvec current_ks) {
  double rate = 1.0;
  for(uword l = 0; l < L; l++) {
    rate *= w(current_ks(l), l);
  }
  for(uword l = 0; l < L - 1; l++) {
    for(uword m = l + 1; m < L; m++) {
      if(m != view_j) {
        rate *= (1.0 + phis(phi_ind_map(m, l)) * (current_ks(l) == current_ks(m)));
      }
    }
  }
  for(uword l = 0; l < view_j - 1; l++) {
    rate *= (1.0 + phis(phi_ind_map(l, view_j)) * (current_ks(l) == current_ks(view_j)));
  }
  return rate;
};

double mdiModelAlt::calcPhiRateNaive(uword view_i, uword view_j) {
  bool shed_j = K(view_i) < K(view_j), weight_updated = false;
  uword K_comb = prod(K), l = 0;
  double rate = 0.0;
  uvec weight_ind(L), K_cumprod(L - 1), backwards_inds(L), for_loop_inds(L), K_rel(L), K_rel_cum(L - 1);
  weight_ind.zeros();
  
  K_rel.zeros();
  K_rel_cum.zeros();
  K_rel = K;
  
  for_loop_inds = regspace<uvec>(0,  1,  L - 1);
  
  if(shed_j) {
    K_rel.shed_row(view_j);
    for_loop_inds.shed_row(view_j);
  } else {
    K_rel.shed_row(view_i);
    for_loop_inds.shed_row(view_i);
  }
  
  // We have shed an entry and we do not need the final entry for the cumulative check
  // K_rel_cum(span(1, L - 2)) = K_rel(span(0, L - 3));
  if(L > 2) {
    K_rel_cum(span(1, L - 2)) = K_rel(span(0, L - 3));
  }
  K_rel_cum(0) = 1;
  K_cumprod = cumprod(K_rel_cum);
  K_comb = prod(K_rel);
  
  // Rcpp::Rcout << "\n\nPHI RATE\nview i: " << view_i << "\nview j: " << view_j << "\n";
  for(uword ii = 0; ii < K_comb; ii++) {
    // Rcpp::Rcout << "\nWeight indices:\n" << weight_ind.t();
    rate += calcPhiRateNaiveSingleIteration(view_i, view_j, weight_ind);
    
    // We have to hold the index for view_i and view_j the same
    // for(uword l = 0; l < (L - 1); l++) {
    for(uword jj = 0; jj < (L - 1); jj++) {
      weight_updated = false;
      l = for_loop_inds(jj);
      // Rcpp::Rcout << "\nl: " << l;
      if(jj == 0) {
        weight_ind(l)++;
        weight_updated = true;
      } else {
        // if((ii % K_cumprod(jj) == 0) && (ii != 0)) {
        if((((ii + 1) % K_cumprod(jj)) == 0) && (ii != 0)) {
          weight_ind(l)++;
          weight_updated = true;
        }
      }
      if(weight_updated && shed_j && (l == view_i)) {
        weight_ind(view_j)++;
      } 
      if(weight_updated &&  (! shed_j) && (l == view_j)) {
        weight_ind(view_i)++;
      }
      if(weight_ind(l) == K(l)) {
        weight_ind(l) = 0;
        if(shed_j && (l == view_i)) {
          weight_ind(view_j) = 0;
        } 
        if(! shed_j && (l == view_j)) {
          weight_ind(view_i) = 0;
        }
      }
    }
  }
  rate *= v;
  return rate;
};

// // The rate for the phi coefficient between the lth and mth datasets.
// double mdiModelAlt::calcPhiRate(uword lstar, uword mstar) {
//   
//   // The rate we return.
//   double rate = 0.0;
//   vec w_l;
//   
//   // A chunk of these objects are used as our combinations matrix includes the
//   // information for l = lstar and l = mstar, so we can shed some data.
//   uword n_used;
//   uvec relevant_inds;
//   vec weight_products, phi_products, relevant_phis;
//   umat relevant_combinations;
//   mat relevant_phi_inidicators, phi_prod_mat;
//   
//   relevant_inds = find(comb_inds.col(lstar) == comb_inds.col(mstar));
//   relevant_combinations = comb_inds.rows(relevant_inds);
// 
//   // We only need the relevant phi indicators
//   relevant_phi_inidicators = phi_indicator_t_mat.cols(relevant_inds);
//  
//   // Drop phi_{lstar, mstar} from both the indicator matrix and the phis vector
//   relevant_phi_inidicators.shed_row(phi_ind_map(mstar, lstar));
//   relevant_phis = phis;
//   relevant_phis.shed_row(phi_ind_map(mstar, lstar));
//   
//   n_used = relevant_combinations.n_rows;
//   
//   weight_products.ones(n_used);
//   phi_products.ones(n_used);
//   
//   // The phi products (should be a matrix of 0's and phis)
//   phi_prod_mat = relevant_phi_inidicators.each_col() % relevant_phis;
//   
//   // Add 1 to each entry to have the object ready to be multiplied
//   phi_prod_mat++;
//   
//   // Vector of the products, this should have the \prod (1 + \phi_{lm} ind(k_l = k_m))
//   // ready to multiply by the weight products
//   phi_products = prod(phi_prod_mat, 0).t();
// 
//   for(uword l = 0; l < L; l++) {
//     if(l != lstar){
//       w_l = w.col(l);
//       weight_products = weight_products % w_l.elem(relevant_combinations.col(l));
//     }
//   }
//   
//   // The rate for the gammas
//   rate = v * accu(weight_products % phi_products);
//   
//   return rate;
// };

void mdiModelAlt::updateWeightsViewL(uword l) {
  
  double shape = 0.0, rate = 0.0;
  uvec members_lk(N);
  
  for(uword k = 0; k < K(l); k++) {
    
    // Find how many labels have the value of k. We used to consider which
    // were outliers and which were not, but the non-outliers still 
    // contribute to the component weight, but not to the component parameters
    // and we use ot hand this down to the local mixture, mistakenly using 
    // the same value for N_k for the component parameters and the weights.
    members_lk = 1 * (labels.col(l) == k);
    members.slice(l).col(k) = members_lk;
    N_k(k, l) = accu(members_lk);
    
    // The hyperparameters
    shape = 1.0 + N_k(k, l);
    // rate = calcWeightRate(l, k);
    rate = calcWeightRateNaive(k, l);
    
    // Sample a new weight
    w(k, l) = rGamma((mass(l) / (double) K(l)) + shape, w_rate_prior + rate);
    
    // Pass the allocation count down to the mixture
    // (this is used in the parameter sampling)
    mixtures[l]->members.col(k) = members_lk;
    
  }
  
  mixtures[l]->N_k = N_k(span(0, K(l) - 1), l);
}

// Update the cluster weights
void mdiModelAlt::updateWeights() {
  
  for(uword l = 0; l < L; l++) {
    updateWeightsViewL(l);
  }

  // std::for_each(
  //   std::execution::par,
  //   L_inds.begin(),
  //   L_inds.end(),
  //   [&](uword l) {
  //     updateWeightsViewL(l);
  //   }
  // );
  
  // // If we only have one dataset, flip back to normalised weights
  // if(L == 1) {
  //   w = w / accu(w) ;
  // }
  
};

double mdiModelAlt::samplePhiShape(arma::uword l, arma::uword m, double rate) {
  bool rTooSmall = false, priorShapeTooSmall = false;
  
  int N_vw = 0;
  double shape = 0.0,
    u = 0.0,
    prod_to_phi_shape = 0.0, 
    prod_to_r_less_1 = 0.0;
  
  uvec rel_inds_l(N), rel_inds_m(N);
  vec log_weights, weights;
  
  // rel_inds_l = labels.col(l) % non_outliers.col(l);
  // rel_inds_m = labels.col(m) % non_outliers.col(m);
  // 
  // N_vw = accu(rel_inds_l == rel_inds_m);
  
  N_vw = accu(labels.col(l) == labels.col(m));
  weights = zeros<vec>(N_vw + 1);
  log_weights = zeros<vec>(N_vw + 1);
  
  log_weights = calculatePhiShapeMixtureWeights(N_vw, rate);
  
  // Normalise the weights
  weights = exp(log_weights - max(log_weights));
  weights = weights / accu(weights);
  
  // Prediction and update
  u = randu<double>( );
  
  shape = sum(u > cumsum(weights)) ;
  
  return shape; 
}

arma::vec mdiModelAlt::calculatePhiShapeMixtureWeights(
    int N_vw, 
    double rate
) {
  
  double r_factorial = 0.0,
    r_alpha_gamma_function = 0.0,
    N_vw_part = 0.0,
    beta_part = 0.0,
    log_n_choose_r = 0.0;

  vec N_vw_ones(N_vw + 1), 
    N_vw_vec(N_vw + 1),
    log_weights(N_vw + 1);
  log_weights.zeros();
  
  N_vw_ones = regspace(0,  N_vw);
  // r_log_factorial_vec = N_log_factorial_vec.subvec(0, N_vw);

  for(int r = 0; r < (N_vw + 1); r++) {
    // N_vw_part = 0.0;
    // for(int ii = 0; ii < r; ii++) {
    //   N_vw_part += log(N_vw - ii);
    // }
    // 
    // r_factorial = N_log_factorial_vec(r);
    
    // N_vw_part = 0.0;
    // r_factorial = 0.0;
    // for(uword ii = 0; ii < r; ii++) {
    //   N_vw_part += std::log(N_vw - ii);
    //   r_factorial += std::log(r - ii);
    // }
    
    log_n_choose_r = logChoose(N_vw, r);
    r_alpha_gamma_function = lgamma(r + phi_shape_prior);
    beta_part = (r + phi_shape_prior) * std::log(rate + phi_rate_prior);
    // log_weights(r) = N_vw_part - r_factorial + r_alpha_gamma_function + beta_part;
    log_weights(r) = log_n_choose_r + r_alpha_gamma_function - beta_part;
  }
  return log_weights;
};

void mdiModelAlt::updatePhis() {
  if(L == 1) {
    return;
  }
  
  uword r = 0;
  double shape = 0.0, rate = 0.0;
  
  for(uword l = 0; l < (L - 1); l++) {
    for(uword m = l + 1; m < L; m++) {
      
      // Find the parameters based on the likelihood
      // rate = calcPhiRate(l, m);
      rate = calcPhiRateNaive(l, m);
      
      // shape = samplePhiShape(l, m, rate);
      shape = 1 + accu(labels.col(l) == labels.col(m));
      
      phis(phi_ind_map(m, l)) = rGamma(
        phi_shape_prior + shape, 
        phi_rate_prior + rate
      );
    }
  }
};

// // Update the context similarity parameters
// void updatePhis() {
//   
//   // Rcpp::Rcout << "\n\nPhis before update:\n" << phis;
//   uword N_lm = 0;
//   double shape = 0.0, rate = 0.0;
//   uvec rel_inds_l(N), rel_inds_m(N);
//   
//   for(uword l = 0; l < (L - 1); l++) {
//     for(uword m = l + 1; m < L; m++) {
//       
//       // I considered excluding the outliers from the phi calculation, but to 
//       // be consistent we would also have to exclude them from component 
//       // weights.
//       rel_inds_l = labels.col(l); // % non_outliers.col(l);
//       rel_inds_m = labels.col(m); // % non_outliers.col(m);
//       
//       N_lm = accu(rel_inds_l == rel_inds_m);
//       shape = 1 + N_lm;
//       
//       // shape = 1 + accu(labels.col(l) == labels.col(m));
//       rate = calcPhiRate(l, m);
//       
//       // Rcpp::Rcout << "\n\nShape:" << shape;
//       // Rcpp::Rcout << "\nRate:" << rate;
//       
//       phis(phi_ind_map(m, l)) = randg(distr_param(
//         phi_shape_prior + shape,
//         1.0 / (phi_rate_prior + rate)
//       )
//       );
//     }
//   }
//   
// };

// // Updates the normalising constant for the posterior
// void mdiModelAlt::updateNormalisingConst() {
//   
//   vec w_l;
//   
//   // A chunk of these objects are used as our combinations matrix includes the
//   // information for l = lstar and l = mstar, so we can shed some data.
//   vec weight_products, phi_products;
//   mat phi_prod_mat;
//   
//   weight_products.ones(n_combinations);
//   phi_products.ones(n_combinations);
//   
//   // The phi products (should be a matrix of 0's and phis)
//   phi_prod_mat = phi_indicator_t_mat.each_col() % phis;
//   
//   // Add 1 to each entry to have the object ready to be multiplied
//   phi_prod_mat++;
//   
//   // Vector of the products, this should have the \prod (1 + \phi_{lm} ind(k_l = k_m))
//   // ready to multiply by the weight products
//   phi_products = prod(phi_prod_mat, 0).t();
//   
//   
//   for(uword l = 0; l < L; l++) {
//     w_l = w.col(l);
//     
//     weight_products = weight_products % w_l.elem(comb_inds.col(l));
//   }
//   
//   // The rate for the gammas
//   Z = accu(weight_products % phi_products);
// };

double mdiModelAlt::calcNormalisingConstNaiveSingleIteration(uvec current_ks) {
  uword l = 0;
  double iter_value = 1.0;
  
  // Rcpp::Rcout << "\nWeights:\n" << w;
  // for(auto & l : for_loop_inds) {
  for(uword jj = 0; jj < L; jj++) {
    // if(jj != v) {
      iter_value *= w(current_ks(jj), jj);
    // }
  }
  // Rcpp::Rcout << "\nNested views loop.";
  for(uword l = 0; l < L - 1; l++) {
    for(uword m = l + 1; m < L; m++) {
      iter_value *= (1.0 + phis(phi_ind_map(m, l)) * (current_ks(l) == current_ks(m)));
    }
  }
  return iter_value;
};

void mdiModelAlt::updateNormalisingConstantNaive() {
  uword K_comb = 0;
  // double rate = 0.0;
  uvec weight_ind(L), K_cumprod(L), K_rel(L), K_rel_cum(L);
  
  weight_ind.zeros();
  
  K_rel.zeros();
  K_rel_cum.zeros();
  K_rel = K;
  
  // Rcpp::Rcout << "\nRelative cumulative object.\n";
  // Rcpp::Rcout << "\nK: " << K_rel.t();
  // Rcpp::stop("stopped");
  
  // We have shed an entry and we do not need the final entry for the cumulative check
  K_rel_cum(span(1, L - 1)) = K_rel(span(0, L - 2));
  K_rel_cum(0) = 1;
  
  // Rcpp::Rcout << "\nRelative cumulative object.\n";
  // Rcpp::Rcout << "\nK_rel_cum:\n" << K_rel_cum.t();
  
  K_cumprod = cumprod(K_rel_cum);
  K_comb = prod(K_rel);
  
  // Rcpp::Rcout << "\nRelative cumulative object declared and calculated.\n" << K_cumprod.t() << "\n";
  // Rcpp::Rcout << "\nNumber of iterations: " << K_comb;
  // Rcpp::stop("stopped");
  Z = 0.0;
  
  // Rcpp::Rcout << "\n\nNORMALISING CONSTANT\n";
  for(uword ii = 0; ii < K_comb; ii++) {
    // Rcpp::Rcout << "\n\ni: " << ii;
    // Rcpp::Rcout << "\neright indices:\n" << weight_ind.t();
    
    // Rcpp::Rcout << "\nIn long loop. ii: " << ii << "\n";
    // Rcpp::Rcout << "\nWeight indices:\n" << weight_ind.t();
    Z += calcNormalisingConstNaiveSingleIteration(weight_ind);
    
    // Rcpp::Rcout << "\nFirst rate calculated.\n";
    
    // We have to hold the index for view_i and view_j the same
    // for(uword l = 0; l < (L - 1); l++) {
    for(uword l = 0; l < L; l++) {
      
      // for(auto & l : for_loop_inds) {
      if(l == 0) {
        weight_ind(l)++;
      } else {
        // if((ii % K_cumprod(jj) == 0) && (ii != 0)) {
        if((((ii + 1) % K_cumprod(l)) == 0) && (ii != 0)) {
          weight_ind(l)++;
        }
      }
      if(weight_ind(l) == K(l)) {
        // Rcpp::Rcout << "\nReset IF statement.";
        
        weight_ind(l) = 0;
      }
    }
    
  }
   
  // Z = rate;
};

void mdiModelAlt::sampleStrategicLatentVariable() {
  v = rGamma(N, Z);
  // v = randg(distr_param(N, 1.0 / Z));
};

void mdiModelAlt::sampleFromPriors() {
  
  // Sample from the prior distribution for the phis and weights
  sampleFromGlobalPriors();
  
  // Sample from the prior distribution for the view-specific mixtures
  sampleFromLocalPriors();
  
};

void mdiModelAlt::sampleFromLocalPriors() {
  for(uword l = 0; l < L; l++) {
    mixtures[l]->sampleFromPriors();
  }
};


vec mdiModelAlt::samplePhiPrior(uword n_phis) {
  return rGamma(n_phis, phi_shape_prior , phi_rate_prior);
};

double mdiModelAlt::sampleWeightPrior(uword l) {
  // return rGamma(w_shape_prior / K(l) , w_rate_prior);
  return rGamma(mass(l) / (double) K(l) , w_rate_prior);
}

vec mdiModelAlt::sampleMassPrior() {
  return rGamma(L, mass_shape_prior, mass_rate_prior);
}

void mdiModelAlt::sampleFromGlobalPriors() {
  
  mass = sampleMassPrior();
  
  if(L > 1) {
    phis = samplePhiPrior(LC2);
    // phis = randg(LC2, distr_param(2.0 , 1.0 / 2));
  } else {
    phis.ones();
  }
  
  for(uword l = 0; l < L; l++) {
    for(uword k = 0; k < K(l); k++) {
      w(k, l) = sampleWeightPrior(l); // randg(distr_param(1.0 / (double)K(l), 1.0));
    }
  }
};

void mdiModelAlt::updateMassParameters() {
  for(uword l = 0; l < L; l++) {
  // std::for_each(
  //   std::execution::par,
  //   L_inds.begin(),
  //   L_inds.end(),
  //   [&](uword l) {
      updateMassParameterViewL(l);
    }
  // );
}

void mdiModelAlt::updateMassParameterViewL(uword l) {
  bool accepted = false;
  double
    current_mass = 0.0, 
    proposed_mass = 0.0, 
    
    cur_log_likelihood = 0.0,
    cur_log_prior = 0.0,
    
    new_log_likelihood = 0.0,
    new_log_prior = 0.0,
    
    acceptance_ratio = 0.0;
  
  vec current_weights(K(l));
  
  current_weights = w(span(0, K(l) - 1), l);
  current_mass = mass(l);
  cur_log_likelihood = gammaLogLikelihood(current_weights, current_mass / (double) K(l), 1);
  cur_log_prior = gammaLogLikelihood(current_mass, mass_shape_prior, mass_rate_prior);
  
  proposed_mass = proposeNewNonNegativeValue(current_mass,
    mass_proposal_window, 
    use_log_norm_proposal
  ); // current_mass + randn() * mass_proposal_window;
  if(proposed_mass <= 0.0) {
    acceptance_ratio = 0.0;
  } else {
    new_log_likelihood = gammaLogLikelihood(current_weights, proposed_mass / (double) K(l), 1);
    new_log_prior = gammaLogLikelihood(proposed_mass,  mass_shape_prior, mass_rate_prior);
    
    acceptance_ratio = exp(
      new_log_likelihood
      + new_log_prior 
      - cur_log_likelihood 
      - cur_log_prior
    );
  }
  
  accepted = metropolisAcceptanceStep(acceptance_ratio);
  
  if(accepted) {
    mass(l) = proposed_mass;
  }
};

mat mdiModelAlt::calculateUpweights(uword l) {
  uvec matching_labels(N);
  mat upweights(N, K(l));
  upweights.zeros();
  
  for(uword m = 0; m < L; m++) {
    
    if(m != l){
      
      for(uword k = 0; k < K(l); k++) {
        
        matching_labels = 1.0 * (labels.col(m) == k);
        
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

void mdiModelAlt::initialiseMDI() {
  mat upweights;
  
  initialiseMixtures();
  sampleFromPriors();
  for(uword l = 0; l < L; l++) {
    upweights = calculateUpweights(l);
    mixtures[l]->initialiseMixture(w(span(0, K(l) - 1), l), upweights.t());
    
    labels.col(l) = mixtures[l]->labels;
    non_outliers.col(l) = mixtures[l]->non_outliers;
  }
};

void mdiModelAlt::updateAllocationViewL(uword l) {
  
  mat upweights; // (N, K_max);

  upweights = calculateUpweights(l);
  
  // Update the allocation within the mixture using MDI level weights and phis
  mixtures[l]->updateAllocation(w(span(0, K(l) - 1), l), upweights.t());
  
  // Pass the new labels from the mixture level back to the MDI level.
  labels.col(l) = mixtures[l]->labels;
  non_outliers.col(l) = mixtures[l]->non_outliers;
  complete_likelihood_vec(l) = mixtures[l]->complete_likelihood;
}

void mdiModelAlt::updateAllocation() {

  complete_likelihood = 0.0;

  std::for_each(
    std::execution::par,
    L_inds.begin(),
    L_inds.end(),
    [&](uword l) {
      updateAllocationViewL(l);
    }
  );
  
  complete_likelihood = accu(complete_likelihood_vec);
};


// This is used to consider possible label swaps
double mdiModelAlt::sampleLabel(arma::uword k, arma::vec K_inv_cum) {
  
  // Need to account for the fixed labels
  // Need the non-fixed classes (e.g., we now need, K, K_fixed and K_unfixed)
  
  // Select another label randomly
  double u = randu();
  uword k_prime = sum(u > K_inv_cum);
  
  // If it is >= than the current label under consideration, add one
  if(k_prime >= k) {
    k_prime++;
  }
  return k_prime;
}

double mdiModelAlt::calcScore(arma::uword l, arma::umat labels) {
  
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
arma::umat mdiModelAlt::swapLabels(arma::uword l, arma::uword k, arma::uword k_prime) {
  
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
void mdiModelAlt::updateLabels() {
  
  bool multipleUnfixedComponents = true;
  
  // // The other component considered
  // uword k_prime = 0;
  // 
  // // Random uniform number
  // double u = 0.0,
  //   
  //   // The current likelihood
  //   curr_score = 0.0,
  //   
  //   // The competitor
  //   alt_score = 0.0,
  //   
  //   // The accpetance probability
  //   accept = 1.0,
  //   log_accept = 0.0,
  //   
  //   // The weight of the kth component if we do accept the swap
  //   old_weight = 0.0;
  // 
  // // Vector of entries equal to 1/(K - 1) (as we exclude the current label) and
  // // its cumulative sum, used to sample another label to consider swapping.
  // vec K_inv, K_inv_cum;
  // 
  // umat swapped_labels(N, L);
  
  
  std::for_each(
    std::execution::par,
    L_inds.begin(),
    L_inds.end(),
    [&](uword l) {
      updateLabelsViewL(l);
    }
  );
  
  // for(uword l = 0; l < L; l++) {
  //   
  //   multipleUnfixedComponents = (K_unfixed(l) > 1);
  //   if(multipleUnfixedComponents) {
  //     updateLabelsViewL(l);
  //   }
  // }
    //   // K_inv = ones<vec>(K(l) - 1) * 1 / (K(l) - 1);
    //   K_inv = ones<vec>(K_unfixed(l) - 1) * (1.0 / (double)(K_unfixed(l) - 1));
    //   K_inv_cum = cumsum(K_inv);
    //   
    //   // The score associated with the current labelling
    //   curr_score = calcScore(l, labels);
    //   
    //   for(uword k = K_fixed(l); k < K(l); k++) {
    //     
    //     // Select another label randomly
    //     k_prime = sampleLabel(k, K_inv_cum) + K_fixed(l);
    //     
    //     // The label matrix updated with the swapped labels
    //     swapped_labels = swapLabels(l, k, k_prime);
    //     
    //     // The score for the swap
    //     alt_score = calcScore(l, swapped_labels);
    //     
    //     // The log acceptance probability
    //     log_accept = alt_score - curr_score;
    //     
    //     if(log_accept < 0) {
    //       accept = std::exp(log_accept);
    //     }
    //     
    //     // If we accept the label swap, update labels, weights and score
    //     if(randu() < accept) {
    //       
    //       // Update the current score
    //       curr_score = alt_score;
    //       labels = swapped_labels;
    //       
    //       // Update the component weights
    //       old_weight = w(k, l);
    //       w(k, l) = w(k_prime, l);
    //       w(k_prime, l) = old_weight;
    //     }
    //   } 
    // }
};

void mdiModelAlt::updateLabelsViewL(uword l) {
  
  bool multipleUnfixedComponents = (K_unfixed(l) > 1);
  
  if(! multipleUnfixedComponents) {
    return;
  }
  
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
  
  
  uvec members_lk, members_lk_prime;
  
  // Vector of entries equal to 1/(K - 1) (as we exclude the current label) and
  // its cumulative sum, used to sample another label to consider swapping.
  vec K_inv, K_inv_cum;
  
  umat swapped_labels(N, L);
  
  // K_inv = ones<vec>(K(l) - 1) * 1 / (K(l) - 1);
  K_inv = ones<vec>(K_unfixed(l) - 1) * (1.0 / (double)(K_unfixed(l) - 1));
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
      
      // Pass the new labels from the mixture level back to the MDI level.
      mixtures[l]->labels = labels.col(l);
      
      // Update the component weights
      old_weight = w(k, l);
      w(k, l) = w(k_prime, l);
      w(k_prime, l) = old_weight;
      
      members_lk = 1 * (labels.col(l) == k);
      members_lk_prime = 1 * (labels.col(l) == k_prime);
      
      members.slice(l).col(k) = members_lk;
      members.slice(l).col(k_prime) = members_lk_prime;
      
      N_k(k, l) = accu(members_lk);
      N_k(k_prime, l) = accu(members_lk_prime);
      
      // Pass the allocation count down to the mixture
      // (this is used in the parameter sampling)
      mixtures[l]->members.col(k) = members_lk;
      mixtures[l]->members.col(k_prime) = members_lk_prime;
      
      mixtures[l]->N_k = N_k(span(0, K(l) - 1), l);
      
      
    }
  } 
};
