
library(tagmReDraft)
library(pRolocdata)
library(magrittr)
library(mdiHelpR)

set.seed(1)
RcppParallel::setThreadOptions()

data("E14TG2aR")

my_data <- prepareMSObject(E14TG2aR)


X <- list(scale(my_data$X))
initial_labels <- my_data$initial_labels
fixed <- my_data$fixed
K <- my_data$K

R <- 15000
thin <- 100
burn <- 5000
n_chains <- 4

mcmc <- runMCMCChains(X, n_chains, R, thin,
  types = "TAGPM",
  K = K,
  initial_labels = initial_labels,
  fixed = fixed,
  proposal_windows = list(c(0.7, 0.5, 0.15))
)


mcmc2 <- runMCMCChains(X, n_chains, R, thin,
  types = "TAGM",
  K = K,
  initial_labels = initial_labels,
  fixed = fixed
)

ensemble_mcmc <- predictFromMultipleChains(mcmc, burn)
ensemble_mcmc2 <- predictFromMultipleChains(mcmc2, burn)

which(colMeans(ensemble_mcmc$outliers[[1]]) > 0.5) |>
  length()


which(colMeans(ensemble_mcmc2$outliers[[1]]) > 0.5) |>
  length()

ensemble_mcmc$hypers[[1]]$amplitude |> 
  log() |>
  boxplot(main = "E14TG2aR: Sampled log amplitude")
ensemble_mcmc$hypers[[1]]$length |> 
  log() |>
  boxplot(main = "E14TG2aR: Sampled log length")
ensemble_mcmc$hypers[[1]]$noise |> 
  log() |>
  boxplot(main = "E14TG2aR: Sampled log noise")

mcmc[[1]]$acceptance_count[[1]]
mcmc[[1]]$hyper_record[[1]] |> boxplot()
par(mfrow = c(2, 2))
mcmc[[1]]$hyper_record[[1]][201:301, seq(1, K)] |>
  log() |>
  boxplot(main = "E14TG2aR: Sampled log amplitude")
mcmc[[1]]$hyper_record[[1]][201:301, seq(K + 1, 2 * K)] |>
  log() |>
  boxplot(main = "E14TG2aR: Sampled log length")
mcmc[[1]]$hyper_record[[1]][201:301, seq(2 * K + 1, 3 * K)] |>
  log() |>
  boxplot(main = "E14TG2aR: Sampled log noise")

# Create the annotation data.frame for the rows
anno_row <- data.frame(
  TAGPM = factor(paste("Cluster", ensemble_mcmc$pred[[1]])),
  TAGM = factor(paste("Cluster", ensemble_mcmc2$pred[[1]]))
) %>%
  magrittr::set_rownames(rownames(X[[1]]))

# Create the annotation colours
ann_colours <- list(TAGPM = viridis::viridis(K), TAGM = viridis::viridis(K))
names(ann_colours$TAGPM) <- paste("Cluster", sort(unique(ensemble_mcmc$pred[[1]])))
names(ann_colours$TAGM) <- paste("Cluster", sort(unique(ensemble_mcmc2$pred[[1]])))

col_pal <- dataColPal()
my_breaks <- defineDataBreaks(scale(X[[1]]), col_pal)

pheatmap::pheatmap(scale(X[[1]]),
  breaks = my_breaks,
  color = col_pal,
  annotation_row = anno_row,
  annotation_colors = ann_colours,
  cluster_cols = FALSE,
  show_rownames = FALSE,
  main = "E14TG2aR data annotated by predicted clusterings"
)

my_inds <- which(ensemble_mcmc$pred[[1]] %in% c(6, 8))
my_inds <- which(ensemble_mcmc$pred[[1]] %in% c(6, 8) & (fixed == 1))

pheatmap::pheatmap(scale(X[[1]])[my_inds, ],
  breaks = my_breaks,
  color = col_pal,
  annotation_row = anno_row[my_inds, ],
  annotation_colors = ann_colours,
  cluster_cols = FALSE,
  show_rownames = FALSE,
  main = "E14TG2aR data annotated by predicted clusterings"
)


ensemble_mcmc$Time
ensemble_mcmc2$Time
