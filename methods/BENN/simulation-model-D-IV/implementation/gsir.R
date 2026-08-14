########################################################################################
#                              11/9/2017: GSIR
########################################################################################
########################################################################################
#                   gram matrix for Gauss kernel: kt should be I_n
# It is flexible enough to be evaluated at a set of new observations; 
# that is, K(X lo {1:n}, Z \lo {1:m}). So if you just want the gram matrix 
# then set x=x, x.new=x to be the same set of predictors
########################################################################################
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
########################################################################################
#                   gram matrix for discrete kernel 
########################################################################################
gram.dis=function(y){
n=length(y);yy=matrix(y,n,n);diff=yy-t(yy);vecker=rep(0,n^2)
vecker[c(diff)==0]=1;vecker[c(diff)!=0]=0 
return(matrix(vecker,n,n))}
########################################################################################
#                   gcv for ex and ey (if ytype is categorical then 
#                                       no need to determin ey) 
########################################################################################
gcv=function(x,y,eps,which,ytype,complex.x,complex.y){
p=dim(x)[2];n=dim(x)[1] 
Kx=gram.gauss(x,x,complex.x)
if(ytype=="scalar") Ky=gram.gauss(y,y,complex.y)
if(ytype=="categorical") Ky=gram.dis(y)
if(which=="ey") {G1=Kx;G2=Ky}
if(which=="ex") {G1=Ky;G2=Kx}
G2inv=matpower(G2+eps*onorm(G2)*diag(n),-1)
nu=sum((G1-G2%*%G2inv%*%G1)^2) 
tr=function(a) return(sum(diag(a)))
de=(1-tr(G2inv%*%G2)/n)^2 
return(nu/de)
}
########################################################################################
#                        operator norm 
########################################################################################
onorm=function(a) return(eigen(round((a+t(a))/2,8))$values[1])
########################################################################################
#                       symmetrize a matrix
########################################################################################
sym=function(a) return(round((a+t(a))/2,9))
############################################################################
#       finding maximizer
############################################################################
minimizer = function(x,y) return(x[order(y)[1]])
############################################################################
#       Moore Penrose power 
############################################################################
mppower=function(matrix,power,ignore){
eig = eigen(matrix)
eval = eig$values
evec = eig$vectors
m = length(eval[abs(eval)>ignore])
tmp = evec[,1:m]%*%diag(eval[1:m]^power)%*%t(evec[,1:m])
return(tmp)
}

########################################################################################
#                                               GSIR 
########################################################################################
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
v=eigen(sym(gsir))$vectors[,1:r]
Kx.new=gram.gauss(x,x.new,complex.x)
pred.new=t(t(v)%*%Gxinv%*%Q%*%Kx.new)
return(pred.new)
}
########################################################################################
#                      k-fold CV
########################################################################################
cv.kfold=function(x,y,k,complex.x,ex){
x=as.matrix(x);y=as.matrix(y);n=dim(x)[1]
ind=numeric();for(i in 1:(k-1)) ind=c(ind,floor(n/k))
ind[k]=n-floor(n*(k-1)/k) 
cv.out=0
for(i in 1:k){
if(i<k) groupi=((i-1)*floor(n/k)+1):(i*floor(n/k))
if(i==k) groupi=((k-1)*floor(n/k)+1):n
groupic=(1:n)[-groupi]
x.tra=as.matrix(x[groupic,]);y.tra=as.matrix(y[groupic,])
x.tes=as.matrix(x[groupi,]);y.tes=as.matrix(y[groupi,])
Kx=gram.gauss(x.tra,x.tra,complex.x)
Kx.tes=gram.gauss(x.tra,x.tes,complex.x)
Ky=gram.gauss(y.tra,y.tra,1)
Ky.tes=gram.gauss(y.tra,y.tes,1)
cvi=sum((t(Ky.tes)-t(Kx.tes)%*%matpower(Kx+ex*onorm(Kx)*diag(dim(y.tra)[1]),-1)%*%Ky)^2)
cv.out=cv.out+cvi
}
return(cv.out)
}
