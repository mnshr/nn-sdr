#!/usr/bin/env Rscript
## data source: https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data

library(mda)
Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "3") # Suppress `tensorflow` notes/warnings
suppressPackageStartupMessages({
    library(NNSDR)
})

## Configuration
d <- 4L                     # reduction dimension
epochs = c(2L, 3L)        # training epochs (OPG, Refinement)

## Loading one site of the "Beijing Air Quality" data set
files <- list.files('data/Beijing\ Multi\ Site\ Air\ Quality\ Data/',
                    pattern = '*.csv', full.names = TRUE)
ds <- na.omit(Reduce(rbind, lapply(files, read.csv)))

## Create model matrix with dummy variables for factors (One-Hot encoded) for
## regression of PM2.5 (and dropping PM10)
X <- model.matrix(~ year + month + day + hour + SO2 + NO2 + CO + O3 + TEMP +
                    PRES + DEWP + RAIN + wd + WSPM + station + 0, ds)
Y <- as.matrix(ds$PM2.5)

## Build Dimension Reduction Neuronal Network model (matching the data)
nn <- nnsdr$new(
    input_shapes = list(x = ncol(X)),
    d = d, # Reduction dimension
    hidden_units = 512L,
    activation = 'relu'
)

## Open simulation log file, write simulation configuration and header
log <- file(format(Sys.time(), "results/Beijing_Air_Quality.csv"), "w", blocking = FALSE)
cat('# d = ', d, '\n# epochs = ', epochs[1], ',', epochs[2], '\n',
    'method,fold,mse,var(Y.test),time.user,time.system,time.elapsed\n',
    sep = '', file = log, append = TRUE)

## K-Fold Cross Validation
K <- 10
for (i in 1:K) {
    ## Split into train/test sets
    train <- (1:K) != i
    X.train <- scale(X[train, ])
    Y.train <-       Y[train, , drop = FALSE]
    X.test <- scale(X[!train, ], center = attr(X.train, 'scaled:center'),
                                  scale = attr(X.train, 'scaled:scale'))
    Y.test <-       Y[!train, , drop = FALSE]

    ## Training
    time <- system.time(nn$fit(X.train, Y.train, epochs = epochs, initializer = 'fromOPG'))

    mse <- mean((nn$predict(X.test) - Y.test)^2)
    cat('"nn.ref",', i, ',', mse, ',', c(var(Y.test)), ',',
        time['user.self'], ',', time['sys.self'], ',', time['elapsed'], '\n',
        sep = '', file = log, append = TRUE)

    ## Linear Model
    time <- system.time(lm.mod <- lm(y ~ ., data.frame(X.train, y = Y.train)))

    mse <- mean((predict(lm.mod, data.frame(X.test, y = Y.test)) - Y.test)^2)
    cat('"lm",', i, ',', mse, ',', c(var(Y.test)), ',',
        time['user.self'], ',', time['sys.self'], ',', time['elapsed'], '\n',
        sep = '', file = log, append = TRUE)

    ## MARS
    time <- system.time(mars.mod <- mars(X.train, Y.train))

    mse <- mean((predict(mars.mod, X.test) - Y.test)^2)
    cat('"mars",', i, ',', mse, ',', c(var(Y.test)), ',',
        time['user.self'], ',', time['sys.self'], ',', time['elapsed'], '\n',
        sep = '', file = log, append = TRUE)

    ## Reset model
    nn$reset()
}

## Finished, close simulation log file
close(log)
