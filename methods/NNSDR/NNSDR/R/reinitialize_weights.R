
#' Re-initialize model weights.
#'
#' An in-place model re-initialization. Intended for simulations to avoid
#' rebuilding the same model architecture for multiple simulation runs.
#'
#' @param model A `keras` model.
#'
#' @seealso https://github.com/keras-team/keras/issues/341
#' @examples
#' # library(tensorflow) # v2
#' K <- tf$keras
#' model <- K$models$Sequential(list(
#'     K$layers$Dense(units = 7L, input_shape = list(3L)),
#'     K$layers$Dense(units = 1L)
#' ))
#' model$compile(loss = 'MSE', optimizer = K$optimizers$RMSprop())
#'
#' model$weights
#' reinitialize_weights(model)
#' model$weights
#'
#' @export
reinitialize_weights <- function(model) {
    for (layer in model$layers) {
        # Unwrap wrapped layers.
        if (any(endsWith(class(layer), 'Wrapper')))
            layer <- layer$layer
        # Re-initialize kernel and bias weight variables.
        for (var in layer$weights) {
            if (any(grep('/recurrent_kernel:', var$name, fixed = TRUE))) {
                var$assign(layer$recurrent_initializer(var$shape, var$dtype))
            } else if (any(grep('/kernel:', var$name, fixed = TRUE))) {
                var$assign(layer$kernel_initializer(var$shape, var$dtype))
            } else if (any(grep('/bias:', var$name, fixed = TRUE))) {
                var$assign(layer$bias_initializer(var$shape, var$dtype))
            } else {
                stop("Unknown initialization for variable ", var$name)
            }
        }
    }
}
