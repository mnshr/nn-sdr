library(energy)

model1=4
model2=1



## BENN m=1

m=1

results.summary=data.frame()

for(n in 1000*(1:8)){
  res1=c()
  for(t in 1:100){
    z_test=read.csv(paste("./data/model",model1,"-",model2,"-",n,"/z_test_",t,".csv",sep=""))[,-1]
    z_pred=read.csv(paste("./results-BENN-unified-std/result-",model1,"-",model2,"-",m,"-",n,"/y_suff_",t,".csv",sep=""))[,-1]
    res1=c(res1,dcor(z_pred,z_test))
  }
  results.summary=rbind(results.summary,
                        list(model1=4,model2=model2,n=n,m=m,
                             avg1=mean(res1),std1=sd(res1))
  )
}

print(results.summary)

results.benn1.model4=results.summary




## BENN m=2

m=2

results.summary=data.frame()

for(n in 1000*(1:8)){
  res1=c()
  for(t in 1:100){
    z_test=read.csv(paste("./data/model",model1,"-",model2,"-",n,"/z_test_",t,".csv",sep=""))[,-1]
    z_pred=read.csv(paste("./results-BENN-unified-std/result-",model1,"-",model2,"-",m,"-",n,"/y_suff_",t,".csv",sep=""))[,-1]
    res1=c(res1,dcor(z_pred,z_test))
  }
  results.summary=rbind(results.summary,
                        list(model1=4,model2=model2,n=n,m=m,
                             avg1=mean(res1),std1=sd(res1))
  )
}

print(results.summary)

results.benn2.model4=results.summary




## BENN m=1000

m=1000

results.summary=data.frame()

for(n in 1000*(1:8)){
  res1=c()
  for(t in 1:100){
    z_test=read.csv(paste("./data/model",model1,"-",model2,"-",n,"/z_test_",t,".csv",sep=""))[,-1]
    z_pred=read.csv(paste("./results-BENN-unified-std/result-",model1,"-",model2,"-",m,"-",n,"/y_suff_",t,".csv",sep=""))[,-1]
    res1=c(res1,dcor(z_pred,z_test))
  }
  results.summary=rbind(results.summary,
                        list(model1=4,model2=model2,n=n,m=m,
                             avg1=mean(res1),std1=sd(res1))
  )
}

print(results.summary)

results.benn3.model4=results.summary






## GMDDNet


results.summary=data.frame()

for(n in 1000*(1:8)){
  res1=c()
  for(t in 1:100){
    try({
      z_test=read.csv(paste("./data/model",model1,"-",model2,"-",n,"/z_test_",t,".csv",sep=""))[,-1]
      z_pred=read.csv(paste("./results-GMDD-correct/result-",model1,"-",model2,"-",n,"/y_suff_",t,".csv",sep=""))[,-1]
      res1=c(res1,dcor(z_pred,z_test))
    })
  }
  results.summary=rbind(results.summary,
                        list(model1=4,model2=model2,n=n,
                             avg1=mean(res1),std1=sd(res1))
  )
}

print(results.summary)

results.gmdd.model4=results.summary






## StoNet

results.summary=data.frame()

for(n in 1000*(1:8)){
  res1=c()
  for(t in 1:100){
    try({
      z_test=read.csv(paste("./data/model",model1,"-",model2,"-",n,"/z_test_",t,".csv",sep=""))[,-1]
      z_pred=read.csv(paste("./results-StoNet-25/result-",model1,"-",model2,"-",n,"/y_suff_",t,".csv",sep=""))[,-1]
      res1=c(res1,dcor(z_pred,z_test))
    })
  }
  results.summary=rbind(results.summary,
                        list(model1=4,model2=model2,n=n,
                             avg1=mean(res1),std1=sd(res1))
  )
}

print(results.summary)

results.stonet.model4=results.summary



## GSIR

results.summary=data.frame()

for(n in 1000*(1:8)){
  res1=c()
  for(t in 1:100){
    try({
      z_test=read.csv(paste("./data/model",model1,"-",model2,"-",n,"/z_test_",t,".csv",sep=""))[,-1]
      load(paste("./results-GSIR/model",model1,"-",model2,"-",n,"/res_",t,".Rdata",sep=""))
      res1=c(res1,dcor(z_pred,z_test))
    })
  }
  results.summary=rbind(results.summary,
                        list(model1=4,model2=model2,n=n,
                             avg1=mean(res1),std1=sd(res1))
  )
}

print(results.summary)

results.gsir.model4=results.summary



## save results

save(results.benn1.model4,results.benn2.model4,results.benn3.model4,
     results.gmdd.model4,results.stonet.model4,results.gsir.model4,
     file="./results-model4-1.Rdata")






## summarize tables

load("./results-model4-1.Rdata")


results.all.string=data.frame()


for(i in 1:nrow(results.benn1.model4)){
  results.all.string=rbind(results.all.string,
                           list(setting=paste(results.benn1.model4$model1[i],results.benn1.model4$model2[i],sep=""),
                                n=results.benn1.model4$n[i],
                                benn1=paste(format(round(results.benn1.model4$avg1[i],2),nsmall=2),"(",
                                            format(round(results.benn1.model4$std1[i],2),nsmall=2),")",sep=""),
                                benn2=paste(format(round(results.benn2.model4$avg1[i],2),nsmall=2),"(",
                                            format(round(results.benn2.model4$std1[i],2),nsmall=2),")",sep=""),
                                benn3=paste(format(round(results.benn3.model4$avg1[i],2),nsmall=2),"(",
                                            format(round(results.benn3.model4$std1[i],2),nsmall=2),")",sep=""),
                                gmdd=paste(format(round(results.gmdd.model4$avg1[i],2),nsmall=2),"(",
                                           format(round(results.gmdd.model4$std1[i],2),nsmall=2),")",sep=""),
                                stonet=paste(format(round(results.stonet.model4$avg1[i],2),nsmall=2),"(",
                                             format(round(results.stonet.model4$std1[i],2),nsmall=2),")",sep=""),
                                gsir=paste(format(round(results.gsir.model4$avg1[i],2),nsmall=2),"(",
                                           format(round(results.gsir.model4$std1[i],2),nsmall=2),")",sep="")
                           ))
}


results.all.string

knitr::kable(results.all.string,"latex")

