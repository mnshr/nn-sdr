library(energy)


n=5000
model2=1

results.summary=data.frame()
for(i in 0:11){
  res1=c()
  for(t in 1:100){
    z_test=read.csv(paste("./data/model4-",model2,"-5000/z_test_",t,".csv",sep=""))[,-1]
    z_pred=read.csv(paste("./results-BENN-sensitivity-1/result-4-",model2,"-",i,"-5000/y_suff_",t,".csv",sep=""))[,-1]
    res1=c(res1,dcor(z_pred,z_test))
  }
  results.summary=rbind(results.summary,list(model1=4,model2=model2,n=n,i=i,avg1=mean(res1),std1=sd(res1)))
  print(results.summary)
}



save(results.summary,
     file=paste("./res-sensitivity-4-",model2,".Rdata",sep=""))


n=5000
model2=2

results.summary=data.frame()
for(i in 0:11){
  res1=c()
  for(t in 1:100){
    z_test=read.csv(paste("./data/model4-",model2,"-5000/z_test_",t,".csv",sep=""))[,-1]
    z_pred=read.csv(paste("./results-BENN-sensitivity-1/result-4-",model2,"-",i,"-5000/y_suff_",t,".csv",sep=""))[,-1]
    res1=c(res1,dcor(z_pred,z_test))
  }
  results.summary=rbind(results.summary,list(model1=4,model2=model2,n=n,i=i,avg1=mean(res1),std1=sd(res1)))
  print(results.summary)
}



save(results.summary,
     file=paste("./res-sensitivity-4-",model2,".Rdata",sep=""))














idx=0:11
l1=rep(c(2,1,3,3),times=3)
r1=rep(c(50,100,100,50),times=3)
l2=rep(c(1,1,2),each=4)
r2=rep(c(2000,1000,1000),each=4)
m=rep(c(1000,500,500),each=4)


load("./res-sensitivity-4-1.Rdata")
results.summary.all.1=cbind(idx,l1,r1,l2,r2,m,results.summary)


load("./res-sensitivity-4-2.Rdata")
results.summary.all.2=cbind(idx,l1,r1,l2,r2,m,results.summary)






results.all.string=data.frame()


for(i in 1:nrow(results.summary.all.1)){
  results.all.string=rbind(results.all.string,
                           list(#setting=paste(results.summary.all.1$model1[i],results.summary.all.1$model2[i],sep=""),
                             idx=results.summary.all.1$idx[i],
                             l1=results.summary.all.1$l1[i],
                             r1=results.summary.all.1$r1[i],
                             l2=results.summary.all.1$l2[i],
                             r2=results.summary.all.1$r2[i],
                             m=results.summary.all.1$m[i],
                             model1=paste(format(round(results.summary.all.1$avg1[i],2),nsmall=2),"(",
                                          format(round(results.summary.all.1$std1[i],2),nsmall=2),")",sep=""),
                             model2=paste(format(round(results.summary.all.2$avg1[i],2),nsmall=2),"(",
                                          format(round(results.summary.all.2$std1[i],2),nsmall=2),")",sep="")
                           ))
}


results.all.string

knitr::kable(results.all.string,"latex")
