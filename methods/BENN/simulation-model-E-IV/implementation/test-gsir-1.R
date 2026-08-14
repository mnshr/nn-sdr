#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)


n=as.numeric(args[1])
t=as.numeric(args[2])
model2=as.numeric(args[3])







library(energy)


matpower=function(a,alpha){
  a=(a+t(a))/2;tmp=eigen(a)
  return(tmp$vectors%*%diag((tmp$values)^alpha)%*%t(tmp$vectors))}

source("gsir.R")



dir.create(paste("./results-GSIR/model4-",model2,"-",n,sep=""))

set.seed(313)

x_train=read.csv(paste("./data/model4-",model2,"-",n,"/x_train_",t,".csv",sep=""))
x_test=read.csv(paste("./data/model4-",model2,"-",n,"/x_test_",t,".csv",sep=""))
y_train=read.csv(paste("./data/model4-",model2,"-",n,"/y_train_",t,".csv",sep=""))
z_test=read.csv(paste("./data/model4-",model2,"-",n,"/z_test_",t,".csv",sep=""))

n=nrow(x_train)
p=ncol(x_train)-1
x_train=x_train[,-1]
y_train=y_train[,-1]
x_test=x_test[,-1]
z_test=z_test[,-1]

time.save=system.time({z_pred=gsir(x_train,x_test,y_train,"scalar",0.1,0.1,1,1,1)})

res=dcor(z_pred,z_test)
save(z_pred,res,time.save,
     file=paste("./results-GSIR/model4-",model2,"-",n,"/res_",t,".Rdata",sep=""))
cat(t,res,"\n")




