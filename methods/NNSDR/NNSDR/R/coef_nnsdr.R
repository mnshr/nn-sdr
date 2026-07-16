#' Extracts the OPG or refined reduction coefficients from an nnsdr class instance
#' 
#' @param object nnsdr class instance
#' @param ... Additional parameters passed down to `object$coef`.
#'
#' @return Matrix
#'
#' @method coef nnsdr
#' @export
coef.nnsdr <- function(object, ...) {
    object$coef(...)
}
