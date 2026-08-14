#' gsir
#'
#' @usage gsir(x,x.new,y,ytype,ex,ey,complex.x,complex.y,r)
#' @param x input predictor matrix from training set
#' @param x.new input predictor matrix from testing set
#' @param y response variables
#' @param ytype type of response variables
#' @param ex tuning parameter for the Tychonoff regularized inverse for GX
#' @param ey tuning parameter for the Tychonoff regularized inverse for GY
#' @param complex.x tuning parameter for the Gaussian kernel in X
#' @param complex.y tuning parameter for the Gaussian kernel in Y
#' @param r number of dimension
#'
#' @return suff.pred: sufficient predictors from GSIR
#' @return obj.mat: objective matrix of GSIR
#' @return eig.val: the first r eigenvalues from the eigendecomposition of the objective matrix
#' @return eig.vec: the first r eigenvectors from the eigendecomposition of the objective matrix
#' @references Li, B. (2018). Sufficient dimension reduction: Methods and applications with R. CRC Press.
#' @export
#'
#' @examples
#'
#' n = 50; p = 5; sigma = 1;
#' x = matrix(rnorm(n*p),n,p) ; err = rnorm(n)
#' y = sin(0.5+(x[,1]+1)^2) + sigma*err; ex=0.01 ; ey=0.01
#' gsir_res <- gsir(x,x,y,"scalar",ex,ey,1,1,1)

gsir=function(x,x.new,y,ytype,ex,ey,complex.x,complex.y,r){
  n=dim(x)[1];p=dim(x)[2];Q=diag(n)-rep(1,n)%*%t(rep(1,n))/n
  Kx=gram.gauss(x,x,complex.x)
  if(ytype=="scalar") Ky=gram.gauss(y,y,complex.y)
  if(ytype=="categorical") Ky=gram.dis(y)
  Gx=Q%*%Kx%*%Q;Gy=Q%*%Ky%*%Q
  Gxinv=matpower(sym(Gx+ex*onorm(Gx)*diag(n)),-1)
  if(ytype=="categorical") Gyinv=mppower(sym(Gy),-1,1e-9)
  if(ytype=="scalar") Gyinv=matpower(sym(Gy+ey*onorm(Gy)*diag(n)),-1)
  a1=Gxinv%*%Gx;a2=Gy%*%Gyinv;gsir=a1%*%a2%*%t(a1)
  objmat <- sym(gsir)
  lamb <- eigen(objmat)$values[1:r]
  v=eigen(objmat)$vectors[,1:r]
  Kx.new=gram.gauss(x,x.new,complex.x)
  pred.new=t(t(v)%*%Gxinv%*%Q%*%Kx.new)
  return(list(suff.pred=pred.new, obj.mat=objmat , eig.val = lamb, eig.vec=v))
}
