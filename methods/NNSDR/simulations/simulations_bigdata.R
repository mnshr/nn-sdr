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
    reps = 10L,             # Number of replications
    dataset = 'M6',          # Name (number) of the data set
    # Sets if reference methods shall be evaluated
    run_mave = TRUE,
    run_cve = TRUE,
    run_nn = TRUE,
    # Neuronal Net. structure/definitions
    hidden_units = 512L,
    activation = 'relu',
    trainable_reduction = TRUE,
    # Neuronal Net. training
    epochs = c(200L, 400L), # Number of training epochs for (`OPG`, Refinement)
    batch_size = 32L,
    initializer = 'fromOPG',
    # Simulation data generation configuration
    seed = 1390L,
    n = 100L,
    p = 20L
))

## Generate reference data (gets re-sampled for each replication)
# Number of observations are irrelevant for the reference to generate a matching
# `NNSDR` estimator
ds <- dataset(args$dataset, n = 100L, p = args$p) # Generates a list with `X`, `Y`, `B` and `name`
# normalize dataset name (before written to the log/results file)
args$dataset <- ds$name

## Build Dimension Reduction Neuronal Network model (matching the data)
nn <- nnsdr$new(
    input_shapes = list(x = ncol(ds$X)),
    d = ncol(ds$B), # depends on the dataset type
    hidden_units = args$hidden_units,
    activation = args$activation,
    trainable_reduction = args$trainable_reduction
)

## Open simulation log file, write simulation configuration and header
log <- file(format(Sys.time(), "results/sim_big_%Y%m%d_%H%M.csv"), "w", blocking = FALSE)
cat(paste('#', names(args), args, sep = ' ', collapse = '\n'), '\n',
    'method,replication,dist.subspace,dist.grassmann,mse,time.user,time.system,time.elapsed\n',
    sep = '', file = log, append = TRUE)

## Repeated simulation runs
for (rep in seq_len(args$reps)) {
    ## Re-sample seeded data for each simulation replication
    with(dataset(ds$name, n = args$n, p = args$p), {
        ## Sample test dataset
        ds.test <- dataset(ds$name, n = 1000L, p = args$p)

        if (args$run_mave) {
            ## First the reference method `MAVE`
            # To be fair for measuring the time, set `max.dim` to true reduction
            # dimension and with `screen = ncol(X)` screening is turned "off".
            time <- system.time(dr <- mave.compute(X, Y, max.dim = ncol(B),
                method = "meanMAVE", screen = ncol(X)))
            d.sub <- dist.subspace(B, coef(dr, ncol(B)), normalize = TRUE)
            d.gra <- dist.grassmann(B, coef(dr, ncol(B)))
            mse <- mean((predict(dr, ds.test$X, dim = ncol(B)) - ds.test$Y)^2)
            cat('"mave",', rep, ',', d.sub, ',', d.gra, ',', mse, ',',
                time['user.self'], ',', time['sys.self'], ',', time['elapsed'],
                '\n', sep = '', file = log, append = TRUE)
            ## and the `OPG` method
            time <- system.time(dr <- mave.compute(X, Y, max.dim = ncol(B),
                method = "meanOPG", screen = ncol(X)))
            d.sub <- dist.subspace(B, coef(dr, ncol(B)), normalize = TRUE)
            d.gra <- dist.grassmann(B, coef(dr, ncol(B)))
            mse <- mean((predict(dr, ds.test$X, dim = ncol(B)) - ds.test$Y)^2)
            cat('"opg",', rep, ',', d.sub, ',', d.gra, ',', mse, ',',
                time['user.self'], ',', time['sys.self'], ',', time['elapsed'],
                '\n', sep = '', file = log, append = TRUE)
        }

        if (args$run_cve) {
            ## Next the CVE method
            time <- system.time(dr <- cve.call(X, Y, k = ncol(B)))
            d.sub <- dist.subspace(B, coef(dr, ncol(B)), normalize = TRUE)
            d.gra <- dist.grassmann(B, coef(dr, ncol(B)))
            mse <- mean((predict(dr, ds.test$X, k = ncol(B)) - ds.test$Y)^2)
            cat('"cve",', rep, ',', d.sub, ',', d.gra, ',', mse, ',',
                time['user.self'], ',', time['sys.self'], ',', time['elapsed'],
                '\n', sep = '', file = log, append = TRUE)
        }

        if (args$run_nn) {
            ## Fit `DR` Neuronal Network model
            time <- system.time(nn$fit(X, Y, epochs = args$epochs,
                batch_size = args$batch_size, initializer = args$initializer))
            # OPG estimate
            d.sub <- dist.subspace(B, coef(nn, 'OPG'), normalize = TRUE)
            d.gra <- dist.grassmann(B, coef(nn, 'OPG'))
            cat('"nn.opg",', rep, ',', d.sub, ',', d.gra, ',NA,NA,NA,NA\n',
                sep = '', file = log, append = TRUE)
            # Refinement estimate
            d.sub <- dist.subspace(B, coef(nn), normalize = TRUE)
            d.gra <- dist.grassmann(B, coef(nn))
            mse <- mean((nn$predict(ds.test$X) - ds.test$Y)^2)
            cat('"nn.ref",', rep, ',', d.sub, ',', d.gra, ',', mse, ',',
                time['user.self'], ',', time['sys.self'], ',', time['elapsed'],
                '\n', sep = '', file = log, append = TRUE)
        }
    })

    ## Invoke the garbage collector
    gc()

    ## Reset model
    nn$reset()
}

## Finished, close simulation log file
close(log)
