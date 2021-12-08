// densityFactory.cpp
// =============================================================================
// included dependencies
# include <RcppArmadillo.h>
# include "densityFactory.h"

using namespace arma ;

// =============================================================================
// virtual densityFactory class

// empty contructor
densityFactory::densityFactory(){ };

std::unique_ptr<density> densityFactory::createDensity(
  densityType type,
  arma::uword K,
  arma::uvec labels,
  arma::mat X
) {
  switch (type) {
    // case G: return std::make_unique<gaussianSampler>(K, labels, concentration, X);
    case MVN: return std::make_unique<mvn>(K, labels, X);
    case C: return std::make_unique<categorical>(K, labels, X);
  default : {
      Rcpp::Rcerr << "invalid density type.\n";
      throw;
    }
  }
};
