library(MAVE)
library(CVarE)
library(NNSDR)

set.seed(797)

dataset <- function(n = 100, p = 10, sd = 0.5) {
    X <- matrix(rnorm(n * (p - 1)), n, p - 1)
    X <- cbind(-0.5 * (X[, 1] + X[, 2]) + 0.001 * rnorm(n), X)
    B <- diag(p)[, 4, drop = FALSE]
    Y <- as.matrix(X[, 4]^2 + rnorm(n, 0, sd))
    return(list(X = X, Y = Y, B = B))
}

nn <- nnsdr$new(
    input_shapes = list(x = 10L),
    d = 1L,
    hidden_units = 512L,
    activation = 'relu',
    trainable_reduction = TRUE
)

sim <- data.frame(mave = rep(NA, 10), cve = NA, nn.opg = NA, nn.ref = NA)
for (i in 1:10) {
    cat(i, "/ 10\n")
    with(dataset(), {
        dr <- mave.compute(X, Y, method = 'meanMAVE', max.dim = 1)
        sim[i, 'mave'] <<- dist.subspace(B, coef(dr, 1), normalize = TRUE)

        dr <- cve.call(X, Y, k = 1)
        sim[i, 'cve'] <<- dist.subspace(B, coef(dr, 1), normalize = TRUE)

        nn$fit(X, Y, epochs = c(200L, 400L), batch_size = 32L, initializer = 'fromOPG')
        sim[i, 'nn.opg'] <<- dist.subspace(B, coef(nn, 'OPG'), normalize = TRUE)
        sim[i, 'nn.ref'] <<- dist.subspace(B, coef(nn), normalize = TRUE)
    })
    nn$reset()
}
round(t(data.frame(
    mean   = colMeans(sim),
    median = apply(sim, 2, median),
    sd     = apply(sim, 2, sd)
)), 3)

#        &$\mave$& $\cve$&$\nn_{512}$ \\
# mean   & 0.917 & 0.164 & 0.101 \\
# median & 0.999 & 0.162 & 0.096 \\
# sd     & 0.256 & 0.057 & 0.032 \\
