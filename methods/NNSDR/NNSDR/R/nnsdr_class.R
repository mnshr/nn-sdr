Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "3")



#' Build MLP
#'
#' @param input_shapes TODO:
#' @param d TODO:
#' @param name TODO:
#' @param add_reduction TODO:
#' @param hidden_units TODO:
#' @param activation TODO:
#' @param output_activation TODO:
#' @param dropout TODO:
#' @param loss TODO:
#' @param optimizer TODO:
#' @param metrics TODO:
#' @param trainable_reduction TODO:
#'
#' @import reticulate tensorflow
#' @keywords internal
build.MLP <- function(input_shapes, d, name, add_reduction,
    output_shape        = 1L,
    hidden_units        = 512L,
    activation          = 'relu',
    output_activation   = NULL,
    dropout             = 0.4,
    loss                = 'MSE',
    optimizer           = 'RMSProp',
    metrics             = NULL,
    trainable_reduction = TRUE
) {
    K <- tf$keras

    inputs <- Map(K$layers$Input,
        shape = as.integer(input_shapes), # drops names (concatenate key error)
        name = if (is.null(names(input_shapes))) "" else names(input_shapes)
    )

    mlp_inputs <- if (add_reduction) {
        reduction <- K$layers$Dense(
            units = d,
            use_bias = FALSE,
            kernel_constraint = function(w) { # polar projection
                lhs <- tf$linalg$sqrtm(tf$matmul(w, w, transpose_a = TRUE))
                tf$transpose(tf$linalg$solve(lhs, tf$transpose(w)))
            },
            trainable = trainable_reduction,
            name = 'reduction'
        )(inputs[[1]])

        c(reduction, inputs[-1])
    } else {
        inputs
    }

    out <- if (length(inputs) == 1) {
        mlp_inputs[[1]]
    } else {
        K$layers$concatenate(mlp_inputs, axis = 1L, name = 'input_mlp')
    }
    for (i in seq_along(hidden_units)) {
        out <- K$layers$Dense(units = hidden_units[i], activation = activation,
                              name = paste0('hidden', i))(out)
        if (dropout > 0)
            out <- K$layers$Dropout(rate = dropout, name = paste0('dropout', i))(out)
    }
    out <- K$layers$Dense(units = output_shape, activation = output_activation,
                          name = 'output')(out)

    mlp <- K$models$Model(inputs = inputs, outputs = out, name = name)

    if (!is.null(metrics)) {
        metrics <- as.list(metrics)
        for (i in rev(seq_along(metrics))) {
            metric <- metrics[[i]]
            if (all(c("nnsdr.metric", name) %in% class(metric))) {
                metric_fn <- reticulate::py_func(metric(mlp))
                reticulate::py_set_attr(metric_fn, "__name__", attr(metric, "name"))
                metrics[[i]] <- metric_fn
            } else if ("nnsdr.metric" %in% class(metric)) {
                metrics[[i]] <- NULL    # Drop
            }
        }
    }

    mlp$compile(loss = loss, optimizer = optimizer, metrics = metrics)

    mlp
}

#' Base Neuronal Network model class
#'
#' @examples
#' model <- nnsdr$new(
#'     input_shapes = list(x = 7L),
#'     d = 2L, hidden_units = 128L
#' )
#'
#' @import methods tensorflow
#' @export nnsdr
#' @exportClass nnsdr
nnsdr <- setRefClass('nnsdr',
    fields = list(
        config = 'list',
        nn.opg = 'ANY',
        nn.ref = 'ANY',
        history.opg = 'ANY',
        history.ref = 'ANY',
        OPG = 'ANY',
        B.ref = 'ANY',
        history = function() {
            if (is.null(.self$history.opg))
                return(NULL)

            if (is.null(.self$history.ref)) {
                history <- data.frame(
                    .self$history.opg,
                    model = factor('OPG')
                )
            } else {
                hist.opg <- data.frame(
                    .self$history.opg,
                    model = factor('OPG')
                )
                hist.ref <- data.frame(
                    .self$history.ref,
                    model = factor('Refinement')
                )
                # Augment mutualy exclusive columns
                hist.opg[setdiff(names(hist.ref), names(hist.opg))] <- NA
                hist.ref[setdiff(names(hist.opg), names(hist.ref))] <- NA
                # Combine/Bind
                history <- rbind(hist.opg, hist.ref)
            }
            history$epoch <- seq_len(nrow(history))

            history
        }
    ),

    methods = list(
        initialize = function(input_shapes, d, output_shape = 1L, ...) {
            # Create config.
            .config <- c(list(
                input_shapes = input_shapes,
                output_shape = output_shape
            ), list(...))
            # Dimensions added later (alows multiple d's)
            d <- sort(as.integer(d), decreasing = TRUE)

            # Build OPG (Step 1) and Refinement (Step 2) Neuronal Networks
            .self$nn.opg <- do.call(build.MLP, c(.config, list(
                name = 'OPG', add_reduction = FALSE
            )))
            .self$nn.ref <- Map(function(d) {
                do.call(build.MLP, c(.config, list(
                    name = 'Refinement', add_reduction = TRUE, d = d
                )))
            }, d)

            # Set config (including dimension(s)) after `build.MLP`
            .self$config <- c(list(d = d), .config)

            # Set initial history field values. If and only if the `history.*`
            # fields are `NULL`, then the Nets are NOT trained.
            .self$history.opg <- NULL
            .self$history.ref <- NULL

            # Set (not jet available) OPG directions, the OPG estimate for
            # reduction dimension `d` is then `.self$OPG[, 1:d]`.
            .self$OPG <- NULL
        },

        fit = function(inputs, output, epochs = 1L, batch_size = 32L,
            initializer = c('random', 'fromOPG'), ..., verbose = 0L
        ) {
            if (is.list(inputs)) {
                inputs <- Map(tf$cast, as.list(inputs), dtype = 'float32')
            } else {
                inputs <- list(tf$cast(inputs, dtype = 'float32'))
            }
            initializer <- match.arg(initializer)

            # Check for OPG history (Step 1), if available skip it.
            if (is.null(.self$history.opg)) {
                # Fit OPG Net and store training history.
                hist <- .self$nn.opg$fit(inputs, output, ...,
                    epochs = as.integer(head(epochs, 1)),
                    batch_size = as.integer(head(batch_size, 1)),
                    verbose = as.integer(verbose)
                )
                .self$history.opg <- as.data.frame(hist$history)
            } else if (verbose > 0) {
                cat("OPG already trained -> skip OPG training.\n")
            }

            # Compute OPG estimate of the Reduction matrix 'B'.
            # Always compute, different inputs change the estimate.
            with(tf$GradientTape() %as% tape, {
                tape$watch(inputs[[1]])
                out <- .self$nn.opg(inputs)
            })
            G <- as.matrix(tape$gradient(out, inputs[[1]]))
            .self$OPG <- eigen(var(G), symmetric = TRUE)$vectors

            # Check for need to initialize the Refinement Nets.
            if (is.null(.self$history.ref)) {
                # Get OPG estimate for max reduction dimension
                B <- .self$OPG[, seq_len(.self$config$d[1]), drop = FALSE]
                # Set Reduction layer
                .self$nn.ref[[1]]$get_layer('reduction')$set_weights(list(B))

                # Check initialization (for random keep random initialization)
                if (initializer == 'fromOPG') {
                    # Initialize Refinement Net weights from the OPG Net.
                    W <- as.array(.self$nn.opg$get_layer('hidden1')$kernel)
                    W <- rbind(
                        t(B) %*% W[1:nrow(B), , drop = FALSE],
                        W[-(1:nrow(B)), , drop = FALSE]
                    )
                    b <- as.array(.self$nn.opg$get_layer('hidden1')$bias)
                    .self$nn.ref[[1]]$get_layer('hidden1')$set_weights(list(W, b))
                    # Get layer names with weights to be initialized from `nn.opg`
                    # These are the output layer and all hidden layers except the first
                    layer.names <- Filter(function(name) {
                        if (name == 'output') {
                            TRUE
                        } else if (name == 'hidden1') {
                            FALSE
                        } else {
                            startsWith(name, 'hidden')
                        }
                    }, lapply(.self$nn.opg$layers, `[[`, 'name'))
                    # Copy `nn.opg` weights to first `nn.ref`
                    for (name in layer.names) {
                        .self$nn.ref[[1]]$get_layer(name)$set_weights(lapply(
                            .self$nn.opg$get_layer(name)$weights, as.array
                        ))
                    }
                }

                # Now train all but the smallest Refinement Nets and move
                # weight to the next smaller net.
                for (i in seq_len(length(.self$nn.ref) - 1)) {
                    # Train current Net
                    hist <- .self$nn.ref[[i]]$fit(inputs, output, ...,
                        epochs = as.integer(tail(epochs, 1)),
                        batch_size = as.integer(tail(batch_size, 1)),
                        verbose = as.integer(verbose)
                    )
                    .self$history.ref <- rbind(
                        .self$history.ref,
                        cbind(d = .self$config$d[i], as.data.frame(hist$history))
                    )
                    # Compute reduced reduction for the next smaller refinement
                    with(tf$GradientTape() %as% tape, {
                        tape$watch(inputs[[1]])
                        out <- .self$nn.ref[[i]](inputs)
                    })
                    G <- as.matrix(tape$gradient(out, inputs[[1]]))
                    B <- eigen(var(G), symmetric = TRUE)$vectors
                    B <- B[, seq_len(.self$config$d[i + 1]), drop = FALSE]

                    .self$nn.ref[[i + 1]]$get_layer('reduction')$set_weights(list(B))
                    # Transfer weights from current to next smaller net
                    W <- as.array(.self$nn.ref[[i]]$get_layer('hidden1')$kernel)
                    b <- as.array(.self$nn.ref[[i]]$get_layer('hidden1')$bias)
                    B.last <- as.array(.self$nn.ref[[i]]$get_layer('reduction')$kernel)
                    .self$nn.ref[[i + 1]]$get_layer('hidden1')$set_weights(list(
                        t(B) %*% B.last %*% W, b))
                    # These are the output layer and all hidden layers except the first
                    layer.names <- Filter(function(name) {
                        if (name == 'output') {
                            TRUE
                        } else if (name == 'hidden1') {
                            FALSE
                        } else {
                            startsWith(name, 'hidden')
                        }
                    }, lapply(.self$nn.ref[[i]]$layers, `[[`, 'name'))
                    # Copy current weights to first next smaller net
                    for (name in layer.names) {
                        .self$nn.ref[[i + 1]]$get_layer(name)$set_weights(lapply(
                            .self$nn.ref[[i]]$get_layer(name)$weights, as.array
                        ))
                    }
                }
            } else if (verbose > 0) {
                cat("Refinement Nets already trained -> continue training.\n")
            }

            # Fit (or continue fitting) the (last, smallest) Refinement Net.
            hist <- tail(.self$nn.ref, 1)[[1]]$fit(inputs, output, ...,
                epochs = as.integer(tail(epochs, 1)),
                batch_size = as.integer(tail(batch_size, 1)),
                verbose = as.integer(verbose)
            )
            .self$history.ref <- rbind(
                .self$history.ref,
                cbind(d = tail(.self$config$d, 1), as.data.frame(hist$history))
            )

            invisible(NULL)
        },
        predict = function(inputs, type = c('Refinement', 'OPG'),
            d = min(.self$config$d)
        ) {
            type <- match.arg(type)
            # Convert inputs to tensors
            if (is.list(inputs)) {
                inputs <- Map(tf$cast, as.list(inputs), dtype = 'float32')
            } else {
                inputs <- list(tf$cast(inputs, dtype = 'float32'))
            }

            if (type == 'Refinement') {
                # Issue warning if the Refinement model (Step 2) used for
                # prediction is not trained.
                if (is.null(.self$history.ref))
                    warning('Refinement model not trained.')
                # Find correct reduction model
                index <- which(.self$config$d == d)
                if (!length(index)) {
                    warning('There is no Refinement model of dim. ', d)
                    return(NULL)
                }
                # Predict
                output <- .self$nn.ref[[index]](inputs)
            } else {
                # Issue warning if OPG model (Step 1) is not trained
                if (is.null(.self$history.opg))
                    warning('OPG model not trained.')
                # Predict
                output <- .self$nn.opg(inputs)
            }

            if (is.list(output)) {
                if (length(output) == 1L) {
                    as.array(output[[1]])
                } else {
                    Map(as.array, output)
                }
            } else {
                as.array(output)
            }
        },
        evaluate = function(inputs, output) {
            if (is.list(inputs)) {
                inputs <- Map(tf$cast, as.list(inputs), dtype = 'float32')
            } else {
                inputs <- list(tf$cast(inputs, dtype = 'float32'))
            }
            eval.opg <- .self$nn.opg$evaluate(inputs, output,
                                              return_dict = TRUE, verbose = 0L)

            if (is.null(.self$history.ref))
                return(data.frame(eval.opg, row.names = "OPG"))

            eval.ref <- Reduce(rbind, Map(function(model, d) {
                data.frame(d = d,
                    model$evaluate(inputs, output,
                                   return_dict = TRUE, verbose = 0L))
            }, .self$nn.ref, .self$config$d))

            # Convert to data.frame
            eval.opg <- data.frame(eval.opg, row.names = "OPG")
            row.names.ref <- if (nrow(eval.ref) == 1) "Refinement"
                             else paste0("Refinement-", seq_len(nrow(eval.ref)))
            eval.ref <- data.frame(eval.ref, row.names = row.names.ref)
            # Augment mutualy exclusive columns
            eval.opg[setdiff(names(eval.ref), names(eval.opg))] <- NA
            eval.ref[setdiff(names(eval.opg), names(eval.ref))] <- NA
            # Combine/Bind
            rbind(eval.opg, eval.ref)
        },
        coef = function(type = c('Refinement', 'OPG'), d = min(.self$config$d)) {
            type <- match.arg(type)
            if (type == 'Refinement') {
                # Extract refined reduction estimate from refinement model
                # with bottleneck if dimension `d`.
                index <- which(.self$config$d == d)
                if (!length(index)) {
                    warning('There is no Refinement model of dim. ', d)
                    return(NULL)
                }
                .self$nn.ref[[index]]$get_layer('reduction')$get_weights()[[1]]
            } else {
                .self$OPG[, seq_len(d), drop = FALSE]
            }
        },
        reset = function(reset = c('both', 'Refinement')) {
            reset <- match.arg(reset)
            if (reset == 'both') {
                reinitialize_weights(.self$nn.opg)
                reset_optimizer(.self$nn.opg$optimizer)
                .self$history.opg <- NULL
                .self$OPG <- NULL
            }
            for (model in .self$nn.ref) {
                reinitialize_weights(model)
                reset_optimizer(model$optimizer)
            }
            .self$history.ref <- NULL
        },
        summary = function() {
            .self$nn.opg$summary()
            for (model in .self$nn.ref) {
                cat('\n')
                model$summary()
            }
        }
    )
)
nnsdr$lock('config')
