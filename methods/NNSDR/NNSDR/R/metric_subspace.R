#' @export
metric.subspace <- function(B_true,
    X = NULL, Y = NULL,
    type = c("Refinement", "OPG"),
    name = "metric.subspace",
    normalize = FALSE
) {
    type <- match.arg(type)

    if (!is.matrix(B_true))
        B_true <- as.matrix(B_true)
    P_true <- B_true %*% solve(crossprod(B_true), t(B_true))
    P_true <- tf$constant(P_true, dtype = 'float32')

    if (normalize) {
        rankSum <- 2 * ncol(B_true)
        c <- 1 / sqrt(min(rankSum, 2 * nrow(B_true) - rankSum))
    } else {
        c <- sqrt(2)
    }
    c <- tf$constant(c, dtype = 'float32')

    if (type == "Refinement") {
        structure(function(model) {
                B <- model$get_layer('reduction')$weights
                function(y_true, y_pred) {
                    P <- tf$linalg$matmul(B, B, transpose_b = TRUE)
                    diff <- P_true - P
                    c * tf$sqrt(tf$reduce_sum(tf$math$multiply(diff, diff)))
                }
            },
            class = c("nnsdr.metric", "Refinement"),
            name = name
        )
    } else {
        X <- tf$cast(X, dtype = 'float32')
        begin <- tf$cast(c(0, nrow(B_true) - ncol(B_true) - 1), dtype = 'int32')
        size <- tf$cast(c(nrow(B_true), ncol(B_true)), dtype = 'int32')
        structure(function(model) {
                function(y_true, y_pred) {
                    with(tf$GradientTape() %as% tape, {
                        tape$watch(X)
                        out <- model(X)
                    })
                    G <- tape$gradient(out, X)
                    B <- tf$linalg$eigh(tf$linalg$matmul(G, G, transpose_a = TRUE))
                    B <- tf$linalg$qr(tf$slice(B[[2]], begin, size))$q

                    P <- tf$linalg$matmul(B, B, transpose_b = TRUE)
                    diff <- P_true - P
                    c * tf$sqrt(tf$reduce_sum(tf$math$multiply(diff, diff)))
                }
            },
            class = c("nnsdr.metric", "OPG"),
            name = name
        )
    }
}
