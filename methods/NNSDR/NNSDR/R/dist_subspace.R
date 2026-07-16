#' Subspace distance
#'
#' @param A,B Basis matrices as representations of elements of the Grassmann
#'  manifold.
#' @param is.ortho Boolean to specify if `A` and `B` are semi-orthogonal. If
#'  false, the projection matrices are computed as
#' \deqn{P_A = A (A' A)^{-1} A'}
#'  otherwise just \eqn{P_A = A A'} since \eqn{A' A} is the identity.
#' @param normalize Boolean to specify if the distance shall be normalized.
#'  Meaning, the maximal distance scaled to be \eqn{1} independent of dimensions.
#' 
#' @seealso
#'  K. Ye and L.-H. Lim (2016) "Schubert varieties and distances between
#'  subspaces of different dimensions" <arXiv:1407.0900>
#' 
#' @export
dist.subspace <- function (A, B, is.ortho = FALSE, normalize = FALSE) {
    if (!is.matrix(A)) A <- as.matrix(A)
    if (!is.matrix(B)) B <- as.matrix(B)

    if (is.ortho) {
        PA <- tcrossprod(A, A)
        PB <- tcrossprod(B, B)
    } else {
        PA <- A %*% solve(t(A) %*% A, t(A))
        PB <- B %*% solve(t(B) %*% B, t(B))
    }

    if (normalize) {
        rankSum <- ncol(A) + ncol(B)
        c <- 1 / sqrt(min(rankSum, 2 * nrow(A) - rankSum))
    } else {
        c <- sqrt(2)
    }

    c * norm(PA - PB, type = "F")
}
