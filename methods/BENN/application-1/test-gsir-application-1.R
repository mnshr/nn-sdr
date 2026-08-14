library(energy)


matpower=function(a,alpha){
  a=(a+t(a))/2;tmp=eigen(a)
  return(tmp$vectors%*%diag((tmp$values)^alpha)%*%t(tmp$vectors))}

source("gsir.R")

gram.gauss=function(x,x.new,complexity){
  x=as.matrix(x);x.new=as.matrix(x.new)
  n=dim(x)[1];m=dim(x.new)[1]
  k2=x%*%t(x);k1=t(matrix(diag(k2),n,n));k3=t(k1);k=k1-2*k2+k3
  k=apply(k,c(1,2),max,0)
  sigma=sum(sqrt(k))/(2*choose(n,2));gamma=complexity/(2*sigma^2)
  k.new.1=matrix(diag(x%*%t(x)),n,m)
  k.new.2=x%*%t(x.new)
  k.new.3=matrix(diag(x.new%*%t(x.new)),m,n)
  return(exp(-gamma*(k.new.1-2*k.new.2+t(k.new.3))))
}

set.seed(313)

x_train=read.csv(paste("./application-data/x_train.csv",sep=""))
x_test=read.csv(paste("./application-data/x_test.csv",sep=""))
y_train=read.csv(paste("./application-data/y_train.csv",sep=""))
y_test=read.csv(paste("./application-data/y_test.csv",sep=""))

n=nrow(x_train)
p=ncol(x_train)-1
x_train=x_train[,-1]
y_train=y_train[,-1]
x_test=x_test[,-1]
y_test=y_test[,-1]

x_all=rbind(x_train,x_test)

time.save=system.time({z_pred=gsir(x_train,x_all,y_train,"scalar",0.1,0.1,1,1,1)})

train.idx=1:nrow(x_train)


res.train=dcor(z_pred[train.idx],y_train)
res.test=dcor(z_pred[-train.idx],y_test)
save(z_pred,res.train,res.test,time.save,
     file=paste("./application-results/results-GSIR.Rdata",sep=""))

plot(z_pred[train.idx],y_train)
plot(z_pred[-train.idx],y_test)

write.csv(z_pred[train.idx],
          file="./application-results/y_suff_train_GSIR.csv")
write.csv(z_pred[-train.idx],
          file="./application-results/y_suff_test_GSIR.csv")
