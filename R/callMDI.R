

callMDI <- function(X, K, types, labels, R, thin, n_chains,
                    fixed = NULL,
                    verbose = TRUE) {
  
  # What types are acceptable
  acceptable_types <- c(1, 3)
  
  X_is_not_a_list <- !is.list(X)
  if (X_is_not_a_list) {
    stop("X must be a list of matrices.")
  }
  
  L <- length(X)

  for (l in seq(L)) {
    .x <- X[[l]]
    .x_is_not_a_matrix <- !is.matrix(.x)

    if (.x_is_not_a_matrix) {
      err <- paste0(
        "All entries of X must be matrices. Entry ",
        l,
        " is not a matrix."
      )
      stop(err)
    }
  }

  # Now we are sure that the datasets are matrices, take some information from
  # them.
  N <- nrow(X)
  row_names <- row.names(X)

  for (l in seq(L)) {
    .x_is_not_correct_dimension <- !(nrow(.x) == N)
    if (.x_is_not_correct_dimension) {
      err <- paste0(
        "All datasets must have the same number of rows. Dataset ",
        l,
        " has a different number of rows to the first dataset."
      )
      stop(err)
    }

    .row_names_not_matching <- !all(row.names(.x) == row_names)
    if (.row_names_not_matching) {
      err <- paste0(
        "All datasets must have the row orders. Dataset ",
        l,
        " has a different row names to the first dataset, please check this."
      )
      stop(err)
    }
  }

  # Check if we are running an unsupervised model.
  unsupervised <- is.null(fixed)
  if (unsupervised) {
    fixed <- matrix(0, nrow = N, ncol = L)
  }


  if (length(K) != L)
    stop("K must be a vector of equal length to X.")

  if (length(types) != L)
    stop("types must be a vector of equal length to X.")

  if (ncol(labels) != L)
    stop("The initial labels must have a column for each entry of X.")

  if (ncol(fixed) != L) {
    stop(
      paste0(
        "fixed, the matrix indicating which labels are observed, must",
        "have a column for each entry of X."
      )
    )
  }

  if (ncol(labels) != N) {
    stop("labels must have the same number of rows as each dataset.")
  }

  if (ncol(fixed) != N) {
    stop("fixed must have the same number of rows as each dataset.")
  }

  labels_not_all_integers <- any((labels %% 1) != 0)
  if (labels_not_all_integers) {
    stop("Not all initial labels are integers. Please check this.")
  }

  fixed_not_all_binary <- !all(fixed %in% c(0, 1))
  if (fixed_not_all_binary) {
    stop("All entries of fixed matrix must be 0 or 1.")
  }

  number_iterations_saved <- floor(R / thin)
  if ((number_iterations_saved < 20) & verbose) {
    warning("Number of saved iterations (before applying a burn in) is less than 20.")
  }

  n_initial_components <- apply(2, labels, function(x) length(unique(x)))
  highest_labels <- apply(2, labels, max)

  for (l in seq(L)) {
    too_many_components <- (n_initial_components(l) > K(l))

    if (too_many_components) {
      stop("There are more unique labels in the initial allocations than components modelled.")
    }

    components_are_mislabelled <- (highest_labels(l) > K(l))
    if (components_are_mislabelled) {
      err <- paste0(
        "The initial labels appear to be wrong. Please check that ",
        "the initial labels are a contiguous sequence. Currently there are less ",
        "unique labels than requested to be modelled (appropriately), but the ",
        "largest value used to represent a component is greater than the ",
        "the number of components modelled in dataset ",
        l,
        "."
      )
    }
    stop(err)
  }
  
  wrong_types_given <- ! all(types %in% acceptable_types)
  if( wrong_types_given ) {
    err <- paste0("types hold values in ", 
      wrong_types_given, 
      ". This has the association, 1 = MVT, 3 = TAGM."
    )
    stop(err)
  }
  
  # Create an object to save the output to. This is used to call ``lapply``.
  mdi_output <- vector("list", n_chains)

  # Use lapply to parallelise the chains.
  mdi_output <- lapply(mdi_output, function(x) {

    # Call MDI
    runSemiSupervisedMDI(R, X, K, types, initial_labels - 1, fixed)
  })

  # Add the entry indicating if the model was unsupervised or not
  mdi_output$Unsupervised <- unsupervised
  
  mdi_output
}
