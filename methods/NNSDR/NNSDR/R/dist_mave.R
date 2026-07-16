#' Subspace distance mentioned in [Xia et al, 2002] (first MAVE paper).
#'
#' @param A,B Basis matrices (assumed full rank) as representations of elements
#'  of the Grassmann manifold.
#' @param is.ortho Boolean to specify if `A` and `B` are semi-orthogonal. If
#'  false, a QR decomposition is used to orthogonalize both `A` and `B`.
#'
#' @seealso
#'  Y. Xia and H. Tong and W.K. Li and L. Zhu (2002) "An adaptive estimation of
#'  dimension reduction space" <DOI:10.1111/1467-9868.03411>
#'
#' @export
dist.mave <- function(A, B, is.ortho = FALSE) {
    if (!is.matrix(A)) A <- as.matrix(A)
    if (!is.matrix(B)) B <- as.matrix(B)

    if (!is.ortho) {
        A <- qr.Q(qr(A))
        B <- qr.Q(qr(B))
    }

    if (ncol(A) < ncol(B)) {
        norm((diag(nrow(A)) - tcrossprod(B, B)) %*% A, 'F')
    } else {
        norm((diag(nrow(A)) - tcrossprod(A, A)) %*% B, 'F')
    }
}
