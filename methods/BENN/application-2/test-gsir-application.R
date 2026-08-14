library(energy)


matpower=function(a,alpha){
  a=(a+t(a))/2;tmp=eigen(a)
  return(tmp$vectors%*%diag((tmp$values)^alpha)%*%t(tmp$vectors))}

source("gsir.R")

set.seed(313)

x_train=read.csv(paste("./application-data-2/x_train.csv",sep=""))
x_test=read.csv(paste("./application-data-2/x_test.csv",sep=""))
y_train=read.csv(paste("./application-data-2/y_train.csv",sep=""))
y_test=read.csv(paste("./application-data-2/y_test.csv",sep=""))

n=nrow(x_train)
p=ncol(x_train)-1
x_train=x_train[,-1]
y_train=y_train[,-1]
x_test=x_test[,-1]
y_test=y_test[,-1]

x_all=rbind(x_train,x_test)

z_pred=gsir(x_train,x_all,y_train,"scalar",0.1,0.1,1,1,1)

train.idx=1:nrow(x_train)


res.train=dcor(z_pred[train.idx],y_train)
res.test=dcor(z_pred[-train.idx],y_test)
save(z_pred,res.train,res.test,
     file=paste("./application-results-2/results-GSIR.Rdata",sep=""))

plot(z_pred[train.idx],y_train)
plot(z_pred[-train.idx],y_test)

write.csv(z_pred[train.idx],
          file="./application-results-2/y_suff_train_GSIR.csv")
write.csv(z_pred[-train.idx],
          file="./application-results-2/y_suff_test_GSIR.csv")
