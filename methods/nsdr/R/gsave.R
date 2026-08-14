#' gsave
#' @usage gsave(x,x.new,y,ytype,ex,ey,comx,comy,r)
#' @param x input predictor matrix from training set
#' @param x.new input predictor matrix from testing set
#' @param y response variables
#' @param ytype type of response variables
#' @param ex tuning parameter for the Tychonoff reguralized inverse for GX
#' @param ey tuning parameter for the Tychonoff reguralized inverse for GY
#' @param comx tuning parameter for the Gaussian kernel in X
#' @param comy tuning parameter for the Gaussian kernel in Y
#' @param r number of dimension
#' @return pred: sufficient predictors from GSAVE
#' @return obj.mat: objective matrix of GSAVE
#' @return eig.val: the first r eigenvalues from the eigendecomposition of the objective matrix
#' @return eig.vec: the first r eigenvectors from the eigendecomposition of the objective matrix
#' @references Li, B. (2018). Sufficient dimension reduction: Methods and applications with R. CRC Press.
#' @export
#'
#' @examples
#' n = 50; p = 5; sigma = 1;
#' x = matrix(rnorm(n*p),n,p) ; err = rnorm(n)
#' y = x[,1]/(0.5+(x[,1]+1)^2) + sigma*err; ex=0.01 ; ey=0.01
#' gsave_res <- gsave(x,x,y,"scalar",ex,ey,1,1,1)

gsave=function(x,x.new,y,ytype,ex,ey,comx,comy,r){
  n=dim(x)[1]
  kx0=gram.gauss(x,x,comx);kx=rbind(1,kx0)
  if(ytype=="scalar") {ky0=gram.gauss(y,y,comy);ky=rbind(1,ky0)}
  if(ytype=="categorical") {ky0=gram.dis(y);ky=rbind(1,ky0)}
  Q=diag(n)-rep(1,n)%*%t(rep(1,n))/n
  kkx=kx%*%Q%*%t(kx)
  kky=ky%*%t(ky)
  if(ytype=="scalar") kkyinv=ridgepower(sym(kky),ey,-1)
  if(ytype=="categorical") kkyinv=mppower(sym(kky),-1,1e-9)
  piy=t(ky)%*%kkyinv%*%ky
  sumlam=diag(apply(piy,1,sum))-piy%*%piy
  a1=diag(apply(piy*piy,1,sum))-piy%*%piy/n
  a2=(piy*piy)%*%piy-piy%*%diag(apply(piy,1,mean))%*%piy
  a3=piy%*%diag(diag(piy%*%Q%*%piy))%*%piy
  mid=Q/n-(2/n)*Q%*%sumlam%*%Q+Q%*%(a1-a2-t(a2)+a3)%*%Q
  kx.new.0=gram.gauss(x,x.new,comx)
  kx.new=rbind(1,kx.new.0)
  n1=dim(kx.new)[2];Q1=diag(n1)-rep(1,n1)%*%t(rep(1,n1))/n1
  kk=ridgepower(kx%*%Q%*%t(kx),ex,-1/2)%*%kx%*%Q
  kk.new=ridgepower(kx%*%Q%*%t(kx),ex,-1/2)%*%kx.new%*%Q1
  objmat <- sym(kk%*%mid%*%t(kk))
  lamb <- eigen(objmat)$values[1:r]
  v=eigen(sym(objmat))$vectors[,1:r]
  pred=t(kk.new)%*%eigen(objmat)$vectors[,1:r]
  return(list(suff.pred=pred, obj.mat=objmat , eig.val = lamb, eig.vec=v))
}
