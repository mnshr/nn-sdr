#' Predict using the fittet neuronal networks
#'
#' @param object instance of class `nnsdr`
#' @param ... arguments passed `predict` method of class `nnsdr`
#'
#' @export
predict.nnsdr <- function(object, ...) {
    object$predict(...)
}
