// noOutliers.h
// =============================================================================
// include guard
#ifndef NOOUTLIERS_H
#define NOOUTLIERS_H

// =============================================================================
// included dependencies
# include <RcppArmadillo.h>
# include "outlierComponent.h"

using namespace arma ;

// =============================================================================
// noOutliers class of outlier component

class noOutliers : virtual public outlierComponent {
  
private:
  
public:
  
  using outlierComponent::outlierComponent;
  
  // Parametrised class
  noOutliers(arma::uvec _fixed, arma::mat _X);
  
  // Destructor
  virtual ~noOutliers() { };
  
  // The likelihood of a given item
  double calculateItemLogLikelihood(arma::vec x);
  
};

#endif /* NOOUTLIERS_H */