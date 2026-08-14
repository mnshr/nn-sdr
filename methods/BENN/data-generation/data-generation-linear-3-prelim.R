p=20
T=2

set.seed(412)

model1=5
for(model2 in c(3)){
  for(n.train in 1000*(1:10)){
    dir.create(paste("./data-prelim/model",model1,"-",model2,"-",n.train,sep=""))
    for(t in 1:T){
      n=n.train+1000
      X=matrix(rnorm(n*p,2,1),n,p)
      Z=matrix(NA,n,2)
      Z[,1]=X[,1]+X[,2]
      Z[,2]=X[,3]+3*X[,4]
      Y=2*sin(Z[,1]*pi/20)+3*sin(Z[,2]*pi/30)^2*rnorm(n,mean=0,sd=1)
      
      test.idx=sample(n,1000)
      X.train=X[-test.idx,]
      Y.train=Y[-test.idx]
      Z.train=Z[-test.idx,]
      X.test=X[test.idx,]
      Y.test=Y[test.idx]
      Z.test=Z[test.idx,]
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


plot(Z[,2],Y)
