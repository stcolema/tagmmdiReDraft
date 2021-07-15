
# if (!requireNamespace("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
#
# BiocManager::install("MSnbase")
# BiocManager::install("pRolocdata")

library(MSnbase)
library(pRolocdata)

library(tidyr)
library(mdiHelpR)
library(tagmReDraft)
library(magrittr)

# ggplot2 theme
setMyTheme()

# random seed
set.seed(1)

# Datasets
data(tan2009r1)
data(tan2009r2)
data(tan2009r3)

# Find the proteins common to the three datasets
proteins_1 <- row.names(exprs(tan2009r1))
proteins_2 <- row.names(exprs(tan2009r2))
proteins_3 <- row.names(exprs(tan2009r3))
proteins_12 <- proteins_1[!is.na(match(proteins_1, proteins_2))]
proteins_123 <- proteins_12[!is.na(match(proteins_12, proteins_3))]

rows_1 <- match(proteins_123, proteins_1)
rows_2 <- match(proteins_123, proteins_2)
rows_3 <- match(proteins_123, proteins_3)

# Put the expression data in a list
X <- list(
  exprs(tan2009r1)[rows_1, ],
  exprs(tan2009r2)[rows_2, ],
  exprs(tan2009r3)[rows_3, ]
)

L <- length(X)
N <- nrow(X[[1]])

# Find the organelles represetned in the datasets; annoyingly we have to drop
# all of the known proteins for some of these due to missingness across
# replicates
organelles <- c(
  unique(fData(tan2009r1)$markers),
  unique(fData(tan2009r2)$markers),
  unique(fData(tan2009r3)$markers)
) %>%
  unique() %>%
  sort()

# Number of organelles to model
K_l <- length(organelles) - 1

# Initial labelling
initial_labels <- true_labels <- matrix(c(
  as.numeric(factor(fData(tan2009r1)$markers[rows_1], levels = organelles)),
  as.numeric(factor(fData(tan2009r2)$markers[rows_2], levels = organelles)),
  as.numeric(factor(fData(tan2009r3)$markers[rows_3], levels = organelles))
),
byrow = F,
ncol = L
)

# The number of components to model in each dataset (does not have to be symmetric)
K <- rep(K_l, L)

# The representation of organelles among the known labels
fracs <- initial_labels %>%
  apply(2, function(x) {
    table(x[x != 12])
  })

# The unknown labels we want to assign a random initial label
unknown_labs <- vector("list", L)
fixed <- matrix(1, N, L)

for (l in 1:L) {
  unknown_labs[[l]] <- which(initial_labels[, l] == (K[l] + 1))
  fixed[unknown_labs[[l]], l] <- 0
}

n_unknown <- unknown_labs %>%
  lapply(length) %>%
  unlist()

# Not all components are represented across all three organells, so just use a
# flat prior
for (l in 1:L) {
  initial_labels[unknown_labs[[l]], l] <- sample(1:K[l],
    size = n_unknown[l],
    prob = rep(1, K[l]),
    # prob = c(
    # 3L,
    # 21L,
    # 3L,
    # 4L,
    # 22L,
    # 4L,
    # 2L,
    # 14L,
    # 1L,
    # 15L,
    # 27L
    # ),
    replace = T
  )
}

# Types of mixture to model (here all are TAGM)
types <- c(3, 3, 3)

# Number of iterations to run
R <- 25000

# Timing!
t_0 <- Sys.time()

# Modelling
samples <- runSemiSupervisedMDI(R, X, K, types, initial_labels - 1, fixed)

t_1 <- Sys.time()

# Time taken
print(t_1 - t_0)

## Time difference of 2.141972 mins

psm1 <- createSimilarityMat(samples$samples[-c(1:5000), , 1]) %>%
  set_rownames(row.names(X[[1]])) %>%
  set_colnames(row.names(X[[1]]))
psm2 <- createSimilarityMat(samples$samples[-c(1:5000), , 2]) %>%
  set_rownames(row.names(X[[2]])) %>%
  set_colnames(row.names(X[[2]]))
psm_3 <- createSimilarityMat(samples$samples[-c(1:5000), , 3]) %>%
  set_rownames(row.names(X[[3]])) %>%
  set_colnames(row.names(X[[3]]))
annotatedHeatmap(psm1, true_labels[, 1])
