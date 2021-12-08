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
    outlierType type, arma::mat X
) {
  switch (type) {
  case E: {
    return std::make_unique<noOutliers>(X);
  }
  case MVT: {
    return std::make_unique<mvt>(X);
  }
  default : {
    Rcpp::Rcerr << "invalid outlier type.\n";
    throw;
  }
  }
};
