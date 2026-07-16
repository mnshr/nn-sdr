#!/usr/bin/env Rscript

library(MAVE)
library(CVarE)
library(mlbench)
Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "3") # Suppress `tensorflow` notes/warnings
suppressPackageStartupMessages({
    library(NNSDR)
})

## Configuration
d <- 1L

## Load Boston Housing data set
data('BostonHousing')
ds <- BostonHousing[, names(BostonHousing) != 'chas']

## Build Dimension Reduction Neuronal Network model (matching the data)
nn <- nnsdr$new(
    input_shapes = list(x = ncol(ds) - 1L),
    d = d, # Reduction dimension
    hidden_units = 512L,
    activation = 'relu'
)

## Open simulation log file, write simulation configuration and header
log <- file(format(Sys.time(), "results/BostonHousing.csv"), "w", blocking = FALSE)
cat('# d = ', d, '\n', 'method,fold,mse\n', sep = '', file = log, append = TRUE)

## K-Fold Cross Validation
K <- nrow(ds)   # with K == nrow(ds) -> LOO CV
for (i in 1:K) {
    ds.train <- ds[(1:K) != i, ]
    ds.test <- ds[(1:K) == i, , drop = FALSE]
    X.train <- as.matrix(ds.train[, names(ds) != 'medv'])
    Y.train <- as.matrix(ds.train[, 'medv'])
    X.test <- as.matrix(ds.test[, names(ds) != 'medv'])
    Y.test <- as.matrix(ds.test[, 'medv'])

    ## Fit `DR` Neuronal Network model
    nn$fit(X.train, Y.train, epochs = c(200L, 400L), initializer = 'fromOPG')
    Y.pred <- nn$predict(X.test)
    mse <- mean((Y.pred - Y.test)^2)
    cat('"nn.ref",', i, ',', mse, '\n',
        sep = '', file = log, append = TRUE)

    ## MAVE as reference
    dr <- mave.compute(X.train, Y.train, method = 'meanMAVE', max.dim = d)
    Y.pred <- predict(dr, X.test, d)
    mse <- mean((Y.pred - Y.test)^2)
    cat('"mave",', i, ',', mse, '\n',
        sep = '', file = log, append = TRUE)

    ## and CVE
    X.scaled <- scale(X.train)
    dr <- cve.call(X.scaled, Y.train, k = d)
    Y.pred <- predict(dr, scale(X.test,
                                scale = attr(X.scaled, 'scaled:scale'),
                                center = attr(X.scaled, 'scaled:center')),
                      k = d)
    mse <- mean((Y.pred - Y.test)^2)
    cat('"cve",', i, ',', mse, '\n',
        sep = '', file = log, append = TRUE)

    ## Reset model
    nn$reset()
}

## Finished, close simulation log file
close(log)
