// outlierComponentFactory.h
// =============================================================================
// include guard
#ifndef OUTLIERCOMPONENTFACTORY_H
#define OUTLIERCOMPONENTFACTORY_H

// =============================================================================
// included dependencies
# include <RcppArmadillo.h>
# include "outlierComponent.h"
# include "mvt.h"
# include "noOutliers.h"

using namespace arma ;

// =============================================================================
// virtual outlierComponentFactory class

class outlierComponentFactory {
  
public:
  
  // empty contructor
  outlierComponentFactory();
  
  // destructor
  virtual ~outlierComponentFactory(){ };
  
  // copy constructor
  outlierComponentFactory(const outlierComponentFactory &L);
  
  enum outlierType {
    E = 0,
    MVT = 1
  };
  
  static std::unique_ptr<outlierComponent> createOutlierComponent(
      outlierType _type, arma::uvec _fixed, arma::mat _X
  );
};

#endif /* OUTLIERCOMPONENTFACTORY_H */