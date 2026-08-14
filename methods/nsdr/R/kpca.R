#' kpca
#'
#' @usage kpca(x,complexity)
#' @param x dataset
#' @param complexity tuning parameter in Gaussian kernel. larger complexity means a wiggly kernel function
#'
#' @return principal component
#' @references Li, B. (2018). Sufficient dimension reduction: Methods and applications with R. CRC Press.
#' @export
#'
#' @examples
#' n = 50; p = 5
#' x = matrix(rnorm(n*p),n,p)
#' pred=kpca(x,1)[,1:3]


kpca=function(x,complexity){
  x=standmat(x);p=dim(x)[2 ];n=dim(x)[1]
  one=rep(1,n);Q=diag(n)-one%*%t(one)/n
  Kx=gramx(x,complexity);Gx=Q%*%Kx%*%Q;v=eigen(Gx)$vectors
  return(matpower(Gx,1/2)%*%v)}
