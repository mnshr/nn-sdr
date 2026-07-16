#!/usr/bin/env Rscript

Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "3") # Suppress `tensorflow` notes/warnings
suppressPackageStartupMessages({
    library(dr)
    library(NNSDR)
})

## Parse script parameters
args <- parse.args(defaults = list(
    # Simulation configuration
    reps = 100,             # Number of replications
    dataset = 'B1',         # Name ('B' for Binary) of the data set
    # Neuronal Net. structure/definitions
    hidden_units = 512L,
    activation = 'relu',
    trainable_reduction = TRUE,
    # Neuronal Net. training
    epochs = c(3L, 5L), # Number of training epochs for (`OPG`, Refinement)
    batch_size = 32L,
    initializer = 'fromOPG',
    seed = 956294L
))

## Generate reference data (gets re-sampled for each replication)
# Generates a list with `X`, `Y`, `B` and `name`
ds <- dataset(args$dataset, n = 1000)

## Build Dimension Reduction Neuronal Network model (matching the data)
nn <- nnsdr$new(
    input_shapes = list(x = ncol(ds$X)),
    d = ncol(ds$B),
    hidden_units = args$hidden_units,
    activation = args$activation,
    trainable_reduction = args$trainable_reduction,
    output_activation = 'sigmoid',
    loss = 'binary_crossentropy',
    # metrics = list('accuracy')
    metrics = list(
        'accuracy',
        metric.subspace(ds$B, ds$X, ds$Y, type = "OPG", normalize = TRUE),
        metric.subspace(ds$B, type = "Refinement", normalize = TRUE)
    )
)

## Open simulation log file, write simulation configuration and header
log <- file(format(Sys.time(), "results/sim_binary_%Y%m%d_%H%M.csv"), "w", blocking = FALSE)
cat(paste('#', names(args), args, sep = ' ', collapse = '\n'), '\n',
    'method,replication,dist.subspace,dist.grassmann,accuracy\n',
    sep = '', file = log, append = TRUE)

## Set seed for sampling simulation data (NOT effecting the `NN` initialization)
set.seed(args$seed)

## Repeated simulation runs
for (rep in seq_len(args$reps)) {
    ## Re-sample seeded data for each simulation replication
    with(dataset(ds$name), {
        ## Sample test dataset
        ds.test <- dataset(ds$name, n = 1000)

        ## Starting with the reference methods `SIR`, `SAVE` and `PHD`
        for (method in c("sir", "save", "phdy")) {
            fit <- dr(Y ~ X, method = method)
            d.sub <- dist.subspace(B, dr.basis(fit, ncol(B)), normalize = TRUE)
            d.gra <- dist.grassmann(B, dr.basis(fit, ncol(B)))
            accuracy <- NA
            cat('"', method, '",', rep, ',', d.sub, ',', d.gra, ',', accuracy,
                '\n', sep = '', file = log, append = TRUE)
        }

        ## Fit `NNSDR` model
        nn$fit(X, Y, epochs = args$epochs,
            batch_size = args$batch_size, initializer = args$initializer)
        # Model evaluation (with metrics)
        eval <- nn$evaluate(ds.test$X, ds.test$Y)
        # `OPG` estimate
        d.sub <- dist.subspace(B, coef(nn, 'OPG'), normalize = TRUE)
        d.gra <- dist.grassmann(B, coef(nn, 'OPG'))
        accuracy <- eval[["OPG", "accuracy"]]
        cat('"nn.opg",', rep, ',', d.sub, ',', d.gra, ',', accuracy,
            '\n', sep = '', file = log, append = TRUE)
        # Refinement estimate
        d.sub <- dist.subspace(B, coef(nn), normalize = TRUE)
        d.gra <- dist.grassmann(B, coef(nn))
        accuracy <- eval[["Refinement", "accuracy"]]
        cat('"nn.ref",', rep, ',', d.sub, ',', d.gra, ',', accuracy,
            '\n', sep = '', file = log, append = TRUE)
    })

    ## Reset model
    nn$reset()
}

## Finished, close simulation log file
close(log)
