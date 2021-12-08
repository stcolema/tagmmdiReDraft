// noOutliers.cpp
// =============================================================================
// included dependencies
# include <RcppArmadillo.h>
# include "noOutliers.h"

using namespace arma ;

// =============================================================================
// noOutliers class of outlier component


// Parametrised class
noOutliers::noOutliers(arma::mat _X) : outlierComponent(_X) {
  outlier_likelihood.set_size(N);
  outlier_likelihood.zeros();
};

double noOutliers::calculateItemLogLikelihood(arma::vec x) {
  return 0.0;
};