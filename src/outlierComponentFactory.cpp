// outlierComponentFactory.cpp
// =============================================================================
// included dependencies
# include <RcppArmadillo.h>
# include "outlierComponentFactory.h"

using namespace arma ;

// =============================================================================
// virtual outlierComponentFactory class

// empty contructor
outlierComponentFactory::outlierComponentFactory(){ };

std::unique_ptr<outlierComponent> outlierComponentFactory::createOutlierComponent(
    outlierType _type, arma::uvec _fixed, arma::mat _X
) {
  switch (_type) {
  case E: {
    return std::make_unique<noOutliers>(_fixed, _X);
  }
  case MVT: {
    return std::make_unique<mvt>(_fixed, _X);
  }
  default : {
    Rcpp::Rcerr << "invalid outlier type.\n";
    throw;
  }
  }
};

