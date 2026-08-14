library(energy)

model1=4
model2=1

results.summary=data.frame()

for(n in 1000*(1:8)){
  for(y_mode in 2:5){
    res1=c()
    for(t in 1:100){
      try({
        z_test=read.csv(paste("./data/model",model1,"-",model2,"-",n,"/z_test_",t,".csv",sep=""))[,-1]
        z_pred=read.csv(paste("./results-GMDD-correct/result-",model1,"-",model2,"-",n,"-",y_mode,"/y_suff_",t,".csv",sep=""))[,-1]
        res1=c(res1,dcor(z_pred,z_test))
      })
    }
    results.summary=rbind(results.summary,
                          list(model1=4,model2=model2,n=n,y_mode=y_mode,avg1=mean(res1),std1=sd(res1)))
    print(results.summary)
  }
}


save(results.summary,
     file=paste("./results-model4-1-gmdd-ymode.Rdata",sep=""))




load("./results-model4-1-gmdd-ymode.Rdata")
results.summary.ymode=results.summary
load("./results-model4-1.Rdata")

results.gmdd2.model4=results.summary.ymode[results.summary.ymode$y_mode==2,]
results.gmdd3.model4=results.summary.ymode[results.summary.ymode$y_mode==3,]
results.gmdd4.model4=results.summary.ymode[results.summary.ymode$y_mode==4,]
results.gmdd5.model4=results.summary.ymode[results.summary.ymode$y_mode==5,]





results.all.string=data.frame()


for(i in 1:nrow(results.benn3.model4)){
  results.all.string=rbind(results.all.string,
                           list(setting=paste(results.benn3.model4$model1[i],results.benn3.model4$model2[i],sep=""),
                                n=results.benn3.model4$n[i],
                                benn3=paste(format(round(results.benn3.model4$avg1[i],2),nsmall=2),"(",
                                            format(round(results.benn3.model4$std1[i],2),nsmall=2),")",sep=""),
                                gmdd=paste(format(round(results.gmdd.model4$avg1[i],2),nsmall=2),"(",
                                           format(round(results.gmdd.model4$std1[i],2),nsmall=2),")",sep=""),
                                gmdd2=paste(format(round(results.gmdd2.model4$avg1[i],2),nsmall=2),"(",
                                            format(round(results.gmdd2.model4$std1[i],2),nsmall=2),")",sep=""),
                                gmdd3=paste(format(round(results.gmdd3.model4$avg1[i],2),nsmall=2),"(",
                                            format(round(results.gmdd3.model4$std1[i],2),nsmall=2),")",sep=""),
                                gmdd4=paste(format(round(results.gmdd4.model4$avg1[i],2),nsmall=2),"(",
                                            format(round(results.gmdd4.model4$std1[i],2),nsmall=2),")",sep=""),
                                gmdd5=paste(format(round(results.gmdd5.model4$avg1[i],2),nsmall=2),"(",
                                            format(round(results.gmdd5.model4$std1[i],2),nsmall=2),")",sep="")
                           ))
}


results.all.string

knitr::kable(results.all.string,"latex")
