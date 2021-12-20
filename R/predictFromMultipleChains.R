#' @title Predict from multiple MCMC chains
#' @description Applies a burn in to and finds a point estimate by combining 
#' multiple chains of ``callMDI``. 
#' @param mcmc_outputs Output from ``runMCMCChains``
#' @param burn The number of MCMC samples to drop as part of a burn in.
#' @param point_estimate_method Summary statistic used to define the point 
#' estimate. Must be ``'mean'`` or ``'median'``. ``'median'`` is the default.
#' @returns A named list of quantities related to prediction/clustering:
#'  
#'  * ``allocation_probability``: List with an $(N x K)$ matrix for each 
#'    semi-supervised view. The point estimate of the allocation probabilities for
#'    each data point to each class.
#'  
#'  * ``prob``: List with an $N$ vector for each semi-supervised view. The point
#'    estimate of the probability of being allocated to the class with the
#'    highest probability.
#'  
#'  * ``pred``: List of $N$ vectorsfor each semi-supervised view. The predicted 
#'    class for each sample.
#'    
#'  * ``allocations``: List of sampled allocations for each view. Columns 
#'    correspond to items being clustered, rows to MCMC samples.
#'
#' @export
predictFromMultipleChains <- function(mcmc_outputs, 
                                      burn, 
                                      point_estimate_method = "median") {
  
  reduced_chains <- processMCMCChains(mcmc_outputs, burn, point_estimate_method)
  n_chains <- length(reduced_chains)
  chain_indices <- seq(1, n_chains)
  
  first_chain <- reduced_chains[[1]]
  
  # Dimensions of the dataset
  V <- first_chain$V
  N <- first_chain$N
  K <- first_chain$K
  
  # MCMC call
  R <- first_chain$R
  thin <- first_chain$thin
  
  # The type of mixture model used
  types <- first_chain$types
  
  # Indices for views
  view_inds <- seq(1, V)
  
  # Is the output semisupervised
  is_semisupervised <- first_chain$Semisupervised
  
  # What summary statistic is used to define our point estimates
  use_median <- point_estimate_method == "median"
  use_mean <- point_estimate_method == "mean"
  wrong_method <- ! (use_median | use_mean)
  if(wrong_method)
    stop("Wrong point estimate method given. Must be one of 'mean' or 'median'")
  
  # We burn the floor of burn / thin of these 
  eff_burn <- floor(burn / thin)
  
  # We record only the floor of R / thin samples
  eff_R <- floor(R / thin) - eff_burn
  
  # The indices dropped as part of the burn in
  dropped_indices <- seq(1, eff_burn)
  
  new_outputs <- mcmc_outputs
  merged_chains <- first_chain
  
  merged_outputs <- vector("list", 4)
  merged_outputs$allocations
  merged_outputs$allocation_probabilities <- vector("list", V)
  merged_outputs$prob <- vector("list", V)
  merged_outputs$pred <- vector("list", V)
  
  first_chain <- TRUE
  for(v in view_inds) {
    
    
    if(is_semisupervised[v]) {
    
      merged_outputs$allocation_probabilities[[v]] <- matrix(0, N, K[v])
      for(ii in chain_indices) {
        
        in_first_chain <- ii == 1
        
        if(in_first_chain){
          merged_outputs$allocations <- new_outputs[[ii]]$allocations
          first_chain <- FALSE
        } else {
          .prev <- merged_outputs$allocations
          .current <- new_outputs[[ii]]$allocations
          merged_outputs$allocations <- rbind(.prev, .current)
        }
  
        .prev <- merged_outputs$allocation_probabilities[[v]]
        .alloc <- new_outputs[[ii]]$allocation_probability[[v]]
        
        merged_outputs$allocation_probabilities[[v]] <- .prev + .alloc
          
      }
    
      merged_outputs$allocation_probabilities[[v]] <- .alloc_prob <- 
        merged_outputs$allocation_probabilities[[v]] / n_chains
      
      merged_outputs$prob[[v]] <- .prob <- apply(.alloc_prob, 1, max)
      merged_outputs$pred[[v]] <- apply(.prob, 1, which.max)
    }
  }
  
  merged_outputs
}
