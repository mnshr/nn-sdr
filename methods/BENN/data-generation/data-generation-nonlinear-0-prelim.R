set.seed(412)
T=2

p=50

library(mvtnorm)

for(model1 in 1:3){
  for(model2 in 1:3){
    for(n.train in c(1000,2000,5000)){
      dir.create(paste("./data-prelim/model",model1,"-",model2,"-",n.train,sep=""))
      for(t in 1:T){
        n=n.train+1000
        X=switch(model2,
                 matrix(rnorm(n*p,0,sqrt(0.5)),n,p),
                 matrix(-1+rpois(n*p,1),n,p),
                 rmvt(n,sigma=0.6*diag(p)+0.4*matrix(1,p,p),df=4))
        Z=switch(model1,
                 X[,1]/(1+(1+X[,2])^2)+(1+X[,2])^2,
                 sin((X[,1]+X[,2])*pi/10)+X[,1]^2,
                 (X[,1]+X[,2])^2*log(X[,1]^2+X[,2]^2+0.001))
        Y=Z+rnorm(n,0,0.5)
        
        test.idx=sample(n,1000)
        X.train=X[-test.idx,]
        Y.train=Y[-test.idx]
        Z.train=Z[-test.idx]
        X.test=X[test.idx,]
        Y.test=Y[test.idx]
        Z.test=Z[test.idx]
        write.csv(X.train,file=paste("./data-prelim/model",model1,"-",model2,"-",n.train,"/x_train_",t,".csv",sep=""))
        write.csv(Y.train,file=paste("./data-prelim/model",model1,"-",model2,"-",n.train,"/y_train_",t,".csv",sep=""))
        write.csv(X.test,file=paste("./data-prelim/model",model1,"-",model2,"-",n.train,"/x_test_",t,".csv",sep=""))
        write.csv(Y.test,file=paste("./data-prelim/model",model1,"-",model2,"-",n.train,"/y_test_",t,".csv",sep=""))
        write.csv(Z.train,file=paste("./data-prelim/model",model1,"-",model2,"-",n.train,"/z_train_",t,".csv",sep=""))
        write.csv(Z.test,file=paste("./data-prelim/model",model1,"-",model2,"-",n.train,"/z_test_",t,".csv",sep=""))
      }
      cat(model1, model2, n, "\n")
    }
    
  }
}

