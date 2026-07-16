#!/usr/bin/env Rscript
## data source: the `MAVE` R-package

library(MAVE)
library(CVarE)
Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "3") # Suppress `tensorflow` notes/warnings
suppressPackageStartupMessages({
    library(NNSDR)
})

## Configuration
d <- 1L                     # reduction dimension
epochs = c(50L, 100L)       # training epochs
dropped <- c('id', 'date', 'zipcode') #, 'sqft_basement')  # columns to be dropped

## Loading the "House Price in King Counte, USA" data set provided by MAVE
data('kc_house_data')
ds <- kc_house_data[, !(names(kc_house_data) %in% dropped)]

## Build Dimension Reduction Neuronal Network model (matching the data)
nn <- nnsdr$new(
    input_shapes = list(x = ncol(ds) - 1L),
    d = d, # Reduction dimension
    hidden_units = 512L,
    activation = 'relu'
)

## Open simulation log file, write simulation configuration and header
log <- file(format(Sys.time(), "results/kc_house_data.csv"), "w", blocking = FALSE)
cat('# d = ', d, '\n# epochs = ', epochs[1], ',', epochs[2], '\n',
    '# dropped = ', paste(dropped, collapse = ', '), '\n',
    'method,fold,mse,var(Y.test),time.user,time.system,time.elapsed\n',
    sep = '', file = log, append = TRUE)

## K-Fold Cross Validation
K <- 10
for (i in 1:K) {
    ds.train <- ds[(1:K) != i, ]
    ds.test <- ds[(1:K) == i, , drop = FALSE]
    X.train <- as.matrix(ds.train[, names(ds) != 'price'])
    Y.train <- as.matrix(ds.train[, 'price'])
    X.test <- as.matrix(ds.test[, names(ds) != 'price'])
    Y.test <- as.matrix(ds.test[, 'price'])

    ## Fit `DR` Neuronal Network model
    time <- system.time(nn$fit(X.train, Y.train, epochs = epochs, initializer = 'fromOPG'))

    mse <- mean((nn$predict(X.test) - Y.test)^2)
    cat('"nn.ref",', i, ',', mse, ',', c(var(Y.test)), ',',
        time['user.self'], ',', time['sys.self'], ',', time['elapsed'], '\n',
        sep = '', file = log, append = TRUE)

    ## `MAVE`
    time <- system.time(dr <- mave.compute(X.train, Y.train, method = 'meanMAVE', max.dim = d))

    # Sometimes the `mda` package fails -> predict with NA/NaN/Inf value error.
    mse <- tryCatch(mean((predict(dr, X.test, d) - Y.test)^2),
                    error = function(err) NA)
    cat('"mave",', i, ',', mse, ',', c(var(Y.test)), ',',
        time['user.self'], ',', time['sys.self'], ',', time['elapsed'], '\n',
        sep = '', file = log, append = TRUE)

    # Current implementation requires too much memory (CVarE v1.1). Run on `VSC`.
    # ## and CVE
    # X.scaled <- scale(X.train)
    # time <- system.time(dr <- cve.call(X.scaled, Y.train, k = d))

    # # Might have the same problem as MAVE since we use `mda` as well.
    # mse <- tryCatch({
    #         Y.pred <- predict(dr, scale(X.test,
    #                                     scale = attr(X.scaled, 'scaled:scale'),
    #                                     center = attr(X.scaled, 'scaled:center')),
    #                           k = d)
    #         mean((Y.pred - Y.test)^2)
    #     },
    #     error = function(err) NA)
    # cat('"cve",', i, ',', mse, ',', c(var(Y.test)), ',',
    #     time['user.self'], ',', time['sys.self'], ',', time['elapsed'], '\n',
    #     sep = '', file = log, append = TRUE)

    ## Reset model
    nn$reset()
}

## Finished, close simulation log file
close(log)
