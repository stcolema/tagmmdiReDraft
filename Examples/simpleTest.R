library(tagmReDraft)
library(magrittr)
library(mdiHelpR)
library(ggplot2)

set.seed(1)

N <- 100
X <- matrix(c(rnorm(N, 0, 1), rnorm(N, 3, 1)), ncol = 2, byrow = T) |> 
  scale() |> 
  set_rownames(paste0("Person", seq(1, N)))
Y <- matrix(c(rnorm(N, 0, 1), rnorm(N, 3, 1)), ncol = 2, byrow = T) |> 
  scale() |> 
  set_rownames(paste0("Person", seq(1, N)))
truth <- c(rep(1, N / 2), rep(2, N / 2))
data_modelled <- list(X, X, X)

V <- length(data_modelled)
R <- 2000
thin <- 25
burn <- 500

alpha <- rep(1, V)
K_max <- 10
K <- rep(K_max, V) # c(3, 3, 3)
labels <- matrix(0, nrow = N, ncol = V)
labels[, 1] <- generateInitialUnsupervisedLabels(N, alpha[1], K[1]) # c(rep(1, N / 2), rep(2, N / 2)) #
labels[, 2] <- generateInitialUnsupervisedLabels(N, alpha[2], K[2])
labels[, 3] <- generateInitialUnsupervisedLabels(N, alpha[3], K[3])
# labels[, 4] <- generateInitialUnsupervisedLabels(N, alpha[4], K[4])

fixed <- matrix(0, nrow = N, ncol = V)
fixed[, 1] <- sample(c(0, 1), N, replace = TRUE, prob = c(3, 1))
labels[, 1] <- generateInitialSemiSupervisedLabels(c(rep(1, N / 2), rep(2, N / 2)), fixed = fixed[, 1])
types <- rep("GP", V)

# mcmc_out <- callMDI(data_modelled, R, thin, labels, fixed, types, K, alpha)
mcmc_out <- runMCMCChains(data_modelled, 3, R, thin, labels, fixed, types, K, alpha)

pred_out <- predictFromMultipleChains(mcmc_out, burn, construct_psm = TRUE)
boxplot(pred_out$phis)

pheatmap::pheatmap(pred_out$psm[[1]], color = simColPal())
pheatmap::pheatmap(pred_out$psm[[2]], color = simColPal())
pheatmap::pheatmap(pred_out$psm[[3]], color = simColPal())

colMeans(pred_out$allocations[[1]] == pred_out$allocations[[2]]) |> boxplot()
colMeans(pred_out$allocations[[1]] == pred_out$allocations[[3]]) |> boxplot()
colMeans(pred_out$allocations[[2]] == pred_out$allocations[[3]]) |> boxplot()

new_out <- processMCMCChains(mcmc_out, burn)

new_out[[1]]$phis |> boxplot()
new_out[[2]]$phis |> boxplot()
new_out[[3]]$phis |> boxplot()

boxplot(pred_out$weights[[1]])
boxplot(pred_out$weights[[2]])
boxplot(pred_out$weights[[3]])

psms <- list()
psms[[1]] <- pred_out$psm[[1]] # mdiHelpR::createSimilarityMat(new_out$allocations[ , , 1])
psms[[2]] <- pred_out$psm[[2]] # mdiHelpR::createSimilarityMat(new_out$allocations[ , , 2])
psms[[3]] <- pred_out$psm[[3]] # mdiHelpR::createSimilarityMat(new_out$allocations[ , , 3])

cl1 <- mcclust::maxpear(psms[[1]])$cl
cl2 <- pred_out$pred[[2]] # mcclust::maxpear(psms[[2]])$cl
cl3 <- pred_out$pred[[3]] # mcclust::maxpear(psms[[3]])$cl

psm_df <- tagmReDraft::prepSimilarityMatricesForGGplot(psms)
psm_df |>
  ggplot(aes(x = x, y= y, fill = Entry)) +
  geom_tile() +
  facet_wrap(~Chain) +
  scale_fill_gradient(low = "#FFFFFF", high = "#146EB4")

annotatedHeatmap(X, cl1)
annotatedHeatmap(X, cl2)
annotatedHeatmap(X, cl3)
