#' Grassmann distance
#'
#' @param A,B Basis matrices as representations of elements of the Grassmann
#'  manifold.
#' @param is.ortho Boolean to specify if `A` and `B` are semi-orthogonal (if
#'  false, both arguments are `qr` decomposed)
#' @param tol passed to `qr`, ignored if `is.ortho` is `true`.
#'
#' @seealso
#'  K. Ye and L.-H. Lim (2016) "Schubert varieties and distances between
#'  subspaces of different dimensions" <arXiv:1407.0900>
#'
#' @export
dist.grassmann <- function(A, B, is.ortho = FALSE, tol = 1e-7) {
    if (!is.ortho) {
        A <- qr.Q(qr(A, tol))
        B <- qr.Q(qr(B, tol))
    } else {
        A <- as.matrix(A)
        B <- as.matrix(B)
    }

    sqrt(sum(acos(pmin(La.svd(crossprod(A, B), 0L, 0L)$d, 1))^2))
}
