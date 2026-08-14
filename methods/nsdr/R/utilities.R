#' gcv
#'
#' @usage gcv(x,y,eps,which,ytype,complex.x,complex.y)
#' @param x input predictor matrix from training set
#' @param y response variables
#' @param eps candidate
#' @param which choose between ex and ey
#' @param ytype type of response variables
#' @param complex.x tuning parameter for the Gaussian kernel in X
#' @param complex.y tuning parameter for the Gaussian kernel in Y
#'
#' @return gcv criterion
#' @references Li, B. (2018). Sufficient dimension reduction: Methods and applications with R. CRC Press.
#' @export
#'
#' @examples
#' n = 50; p = 5; sigma = 1;
#' x = matrix(rnorm(n*p),n,p) ; err = rnorm(n)
#' y = (x[,1]+1)^2 + sigma*err; ex=0.01 ; ey=0.01; candidate=0.01
#' epsx <- gcv(x,y,candidate,"ex", "categorical",1,1)
#' epsy <- gcv(x,y,candidate,"ey", "categorical",1,1)
gcv <- function(x,y,eps,which,ytype,complex.x,complex.y){
  p=dim(x)[2]; n=dim(x)[1]
  Kx=gram.gauss(x,x,complex.x)
  if(ytype=="scalar") Ky=gram.gauss(y,y,complex.y)
  if(ytype=="categorical") Ky=gram.dis(y)
  if(which=="ey"){G1=Kx;G2=Ky}
  if(which=="ex"){G1=Ky;G2=Kx}
  G2inv=matpower(G2+eps*onorm(G2)*diag(n),-1)
  nu=sum((G1-G2%*%G2inv%*%G1)^2)
  de=(1-tr(G2inv%*%G2)/n)^2
  return(nu/de)
}




#' gram.dis
#'
#' @usage gram.dis(y)
#' @param y discrete vector
#'
#' @return gram matrix for discrete vector
#' @references Li, B. (2018). Sufficient dimension reduction: Methods and applications with R. CRC Press.

#' @export
#'
#' @examples
#' toy <- c(1,2,3,1)
#' result=gram.dis(toy)
gram.dis=function(y){
  n=length(y);yy=matrix(y,n,n);diff=yy-t(yy);vecker=rep(0,n^2)
  vecker[c(diff)==0]=1;vecker[c(diff)!=0]=0
  return(matrix(vecker,n,n))}



#' gram.gauss
#'
#' @usage gram.gauss(x,x.new,complexity)
#' @param x previous observations
#' @param x.new new observations
#' @param complexity tuning parameter in Gaussian kernel
#'
#' @return Gram matrix for the Gaussian kernel. This also can be used to project predictor on the testing set.
#' @references Li, B. (2018). Sufficient dimension reduction: Methods and applications with R. CRC Press.
#' @export
#'
#' @examples
#' old <- matrix(c(1,2,3,4),2,2)
#' new <- matrix(c(5,6,7,8),2,2)
#' result <- gram.gauss(old,new,1)
gram.gauss=function(x,x.new,complexity){
  x=as.matrix(x);x.new=as.matrix(x.new)
  n=dim(x)[1];m=dim(x.new)[1]
  k2=x%*%t(x);k1=t(matrix(diag(k2),n,n));k3=t(k1);k=k1-2*k2+k3
  sigma=sum(sqrt(k))/(2*choose(n,2));gamma=complexity/(2*sigma^2)
  k.new.1=matrix(diag(x%*%t(x)),n,m)
  k.new.2=x%*%t(x.new)
  k.new.3=matrix(diag(x.new%*%t(x.new)),m,n)
  return(exp(-gamma*(k.new.1-2*k.new.2+t(k.new.3))))
}


#' gramx
#'
#' @usage gramx(x,complexity)
#' @param x data
#' @param complexity tuning parameter in Gaussian kernel
#'
#' @return gram matrix Q x KX x Q
#' @references Li, B. (2018). Sufficient dimension reduction: Methods and applications with R. CRC Press.
#' @export
#'
#' @examples
#' vec <- matrix(rnorm(4),2,2)
#' res <- gramx(vec,1)
gramx=function(x,complexity){
  n=dim(x)[1]
  k2=x%*%t(x);k1=t(matrix(diag(k2),n,n));k3=t(k1);k=k1-2*k2+k3
  sigma=(sum(sqrt(k))-tr(sqrt(k)))/(2*choose(n,2));gamma=1/(sigma^2)
  return(exp(-complexity*gamma*(k1-2*k2+k3)))}


#' matpower
#'
#' @usage matpower(a,alpha)
#' @param a matrix
#' @param alpha power
#'
#' @return power of a matrix
#' @references Li, B. (2018). Sufficient dimension reduction: Methods and applications with R. CRC Press.
#' @export
#'
#' @examples
#' mat <- matrix(rnorm(4,1,3),2,2)
#' invmat <- matpower(mat,-1)
matpower = function(a,alpha){
  a = (a + t(a))/2
  tmp = eigen(a)
  return(tmp$vectors%*%diag((tmp$values)^alpha)%*%
           t(tmp$vectors))}


#' mppower
#'
#' @usage mppower(matrix,power,ignore)
#' @param matrix input matrix
#' @param power power
#' @param ignore ignoring criterion
#'
#' @return Moore penrose inverse
#' @references Li, B. (2018). Sufficient dimension reduction: Methods and applications with R. CRC Press.
#' @export
#'
#' @examples
#' a <- matrix(rnorm(4,0,1),2,2)
#' mppower(a,-1,0.2)
mppower=function(matrix,power,ignore){
  eig = eigen(matrix)
  eval = eig$values
  evec = eig$vectors
  m = length(eval[abs(eval)>ignore])
  tmp = evec[,1:m]%*%diag(eval[1:m]^power)%*%t(evec[,1:m])
  return(tmp)
}


#' onorm
#'
#' @usage onorm(a)
#' @param a input matrix
#'
#' @return result of the operator norm
#' @references Li, B. (2018). Sufficient dimension reduction: Methods and applications with R. CRC Press.
#' @export
#'
#' @examples
#' a <- matrix(c(1,2,3,4),2,2)
#' result <- onorm(a)
onorm=function(a){
  return(eigen(round((a+t(a))/2,8))$values[1])
}


#' ridgepower
#'
#' @usage ridgepower(a,e,c)
#' @param a square matrix
#' @param e tuning parameter
#' @param c power
#'
#' @return matrix with the power
#' @references Li, B. (2018). Sufficient dimension reduction: Methods and applications with R. CRC Press.
#' @export
#'
#' @examples
#' x <- matrix(c(1:4),2,2)
#' result <- ridgepower(x, 0.001, -1)
ridgepower=function(a,e,c){
  return(matpower(a+e*onorm(a)*diag(dim(a)[1]),c))
}

#' standmat
#'
#' @usage spearman(x1,x2)
#' @param x1 first argument
#' @param x2 second argument
#' @return standardized matrix
#' @references Li, B. (2018). Sufficient dimension reduction: Methods and applications with R. CRC Press.
#' @export
#' @importFrom stats cor
#' @examples
#' x1 <- rnorm(100)
#' x2 <- rnorm(100)
#' spearman(x1,x2)
spearman = function(x1,x2){
  return(abs(cor(rank(x1),rank(x2))))}


#' standmat
#'
#' @usage standmat(x)
#' @param x matrix
#'
#' @return standardized matrix
#' @references Li, B. (2018). Sufficient dimension reduction: Methods and applications with R. CRC Press.
#' @export
#'
#' @examples
#' mat <- matrix(rnorm(4),2,2)
#' standmat(mat)
standmat=function(x) {
  return(t((t(x)-apply(x,2,mean))/apply(x,2,sd)))
}

utils::globalVariables(c("stats", "sd"))


#' sym
#'
#' @usage sym(a)
#' @param a any matrix
#'
#' @return symmetrize matrix when matrix is theoretically symmetric but not in numerically
#' @references Li, B. (2018). Sufficient dimension reduction: Methods and applications with R. CRC Press.
#' @export
#'
#' @examples
#' ex <- matrix(c(1.1,2.1,1.2,2.2),2,2)
#' result <- sym(ex)
sym=function(a) {
  return(round((a+t(a))/2,9))
}


#' trace
#'
#' @usage tr(a)
#' @param a any matrix
#'
#' @return trace value of the matrix
#' @references Li, B. (2018). Sufficient dimension reduction: Methods and applications with R. CRC Press.
#' @export
#'
#' @examples
#' mat <- matrix(2,2,2)
#' tr(mat)
tr=function(a){ return(sum(diag(a)))}

