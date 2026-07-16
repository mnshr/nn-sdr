
#' Parses script arguments.
#'
#' @param defaults list of default parameters. Names of the provided defaults
#'  define the allowed parameters.
#' @param args Arguments to parge, if missing the Rscript params are taken.
#'
#' @return calling script arguments of `Rscript`
#'
#' @export
parse.args <- function(defaults, args) {
    if (missing(args))
        args <- commandArgs(trailingOnly = TRUE)
    if (length((args)) == 0)
        return(defaults)

    args <- strsplit(sub('--', '', args, fixed = TRUE), '=')
    values <- Map(`[`, args, 2)
    names(values) <- unlist(Map(`[`, args, 1))

    if (!all(names(values) %in% names(defaults)))
        stop('Found unknown script parameter')

    for (i in seq_along(defaults)) {
        name <- names(defaults)[i]

        if (name %in% names(values)) {
            value <- unlist(strsplit(values[[name]], ',', fixed = TRUE))
            value <- eval(call(paste0('as.', typeof(defaults[[name]])), value))
            defaults[[name]] <- value
        }
    }

    defaults
}
