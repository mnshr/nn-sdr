#!/usr/bin/env Rscript

library(MAVE)
library(CVarE)
Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "3") # Suppress `tensorflow` notes/warnings
suppressPackageStartupMessages({
    library(NNSDR)
})

## Parse script parameters
args <- parse.args(defaults = list(
    # Simulation configuration
    reps = 100,             # Number of replications
    dataset = 'M1',         # Name (number) of the data set
    # Neuronal Net. structure/definitions
    hidden_units = 512L,
    activation = 'relu',
    trainable_reduction = TRUE,
    # Neuronal Net. training
    epochs = c(200L, 400L), # Number of training epochs for (`OPG`, Refinement)
    batch_size = 32L,
    initializer = 'fromOPG',
    seed = 1390L
))

## Generate reference data (gets re-sampled for each replication)
# Generates a list with `X`, `Y`, `B` and `name`
ds <- dataset(args$dataset, n = 100)

## Build Dimension Reduction Neuronal Network model (matching the data)
nn <- nnsdr$new(
    input_shapes = list(x = ncol(ds$X)),
    d = ncol(ds$B),
    hidden_units = args$hidden_units,
    activation = args$activation,
    trainable_reduction = args$trainable_reduction
)

## Open simulation log file, write simulation configuration and header
log <- file(format(Sys.time(), "results/sim_%Y%m%d_%H%M.csv"), "w", blocking = FALSE)
cat(paste('#', names(args), args, sep = ' ', collapse = '\n'), '\n',
    'method,replication,dist.subspace,dist.grassmann,mse\n',
    sep = '', file = log, append = TRUE)

## Set seed for sampling simulation data (NOT effecting the `NN` initialization)
set.seed(args$seed)

## Repeated simulation runs
for (rep in seq_len(args$reps)) {
    ## Re-sample seeded data for each simulation replication
    with(dataset(ds$name), {
        ## Sample test dataset
        ds.test <- dataset(ds$name, n = 1000)

        ## First the reference method `MAVE`
        dr <- mave(Y ~ X, method = "meanMAVE")
        d.sub <- dist.subspace(B, coef(dr, ncol(B)), normalize = TRUE)
        d.gra <- dist.grassmann(B, coef(dr, ncol(B)))
        mse <- mean((predict(dr, ds.test$X, dim = ncol(B)) - ds.test$Y)^2)
        cat('"mave",', rep, ',', d.sub, ',', d.gra, ',', mse, '\n', sep = '',
            file = log, append = TRUE)
        ## and the `OPG` method
        dr <- mave(Y ~ X, method = "meanOPG")
        d.sub <- dist.subspace(B, coef(dr, ncol(B)), normalize = TRUE)
        d.gra <- dist.grassmann(B, coef(dr, ncol(B)))
        mse <- mean((predict(dr, ds.test$X, dim = ncol(B)) - ds.test$Y)^2)
        cat('"opg",', rep, ',', d.sub, ',', d.gra, ',', mse, '\n', sep = '',
            file = log, append = TRUE)

        ## Next the `CVE` method
        dr <- cve(Y ~ X, k = ncol(B))
        d.sub <- dist.subspace(B, coef(dr, ncol(B)), normalize = TRUE)
        d.gra <- dist.grassmann(B, coef(dr, ncol(B)))
        mse <- mean((predict(dr, ds.test$X, k = ncol(B)) - ds.test$Y)^2)
        cat('"cve",', rep, ',', d.sub, ',', d.gra, ',', mse, '\n', sep = '',
            file = log, append = TRUE)

        ## Fit `NNSDR` model
        nn$fit(X, Y, epochs = args$epochs,
            batch_size = args$batch_size, initializer = args$initializer)
        # `OPG` estimate
        d.sub <- dist.subspace(B, coef(nn, 'OPG'), normalize = TRUE)
        d.gra <- dist.grassmann(B, coef(nn, 'OPG'))
        cat('"nn.opg",', rep, ',', d.sub, ',', d.gra, ',', NA, '\n', sep = '',
            file = log, append = TRUE)
        # Refinement estimate
        d.sub <- dist.subspace(B, coef(nn), normalize = TRUE)
        d.gra <- dist.grassmann(B, coef(nn))
        mse <- mean((nn$predict(ds.test$X) - ds.test$Y)^2)
        cat('"nn.ref",', rep, ',', d.sub, ',', d.gra, ',', mse, '\n', sep = '',
            file = log, append = TRUE)
    })

    ## Reset model
    nn$reset()
}

## Finished, close simulation log file
close(log)
