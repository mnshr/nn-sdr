
#' Gets the `Rscript` file name
#'
#' @note only relevant in scripts (useless in interactive `R` sessions)
#'
#' @return character array of the file name or empty
#'
#' @export
get.script <- function() {
    args <- commandArgs()
    sub('--file=', '', args[startsWith(args, '--file=')])
}
