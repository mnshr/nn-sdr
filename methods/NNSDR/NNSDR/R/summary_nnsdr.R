#' Summary for the OPG and Refinement model of an nnsdr class instance
#' 
#' @param object nnsdr class instance
#' @param ... ignored.
#' 
#' @return No return value, prints human readable summary.
#'
#' @method summary nnsdr
#' @export
summary.nnsdr <- function(object, ...) {
    object$summary()
}
