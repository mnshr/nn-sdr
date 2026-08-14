## StoNet

results.4cases=data.frame()
for(r in c(10,15,20,25)){
  results.summary=data.frame()
  for(model1 in 1:3){
    for(model2 in 1:3){
      for(n in c(1000,2000,5000)){
        results=read.csv(paste("./results-StoNet-",r,"/result-",model1,"-",model2,"-",n,".csv",sep=""))[,2]
        #cat(model1, model2, n, mean(results), sd(results), "\n")
        results.summary=rbind(results.summary,list(model1=model1,model2=model2,n=n,avg=mean(results),std=sd(results)))
      }
    }
  }
  if(r==10){
    results.4cases=data.frame(avg10=results.summary$avg)
  }else{
    results.4cases=cbind(results.4cases,results.summary$avg)
  }
}
colnames(results.4cases)=c("avg10","avg15","avg20","avg25")



hash=rep(0,4)
for(i in 1:27){
  hash[which.max(results.4cases[i,])]=hash[which.max(results.4cases[i,])]+1
}


hash


r=25

results.summary=data.frame()
for(model1 in 1:3){
  for(model2 in 1:3){
    for(n in c(1000,2000,5000)){
      results=read.csv(paste("./results-StoNet-",r,"/result-",model1,"-",model2,"-",n,".csv",sep=""))[,2]
      #cat(model1, model2, n, mean(results), sd(results), "\n")
      results.summary=rbind(results.summary,list(model1=model1,model2=model2,n=n,avg=mean(results),std=sd(results)))
    }
  }
}

results.avg.stonet=results.summary$avg
results.std.stonet=results.summary$std




## BENN

for(m in 1:3){
  results.summary=data.frame()
  for(model1 in 1:3){
    for(model2 in 1:3){
      for(n in c(1000,2000,5000)){
        results=read.csv(paste("./results-BENN-unified-std/result-",model1,"-",model2,"-",m,"-",n,".csv",sep=""))[,2]
        #cat(model1, model2, n, mean(results), sd(results), "\n")
        results.summary=rbind(results.summary,list(model1=model1,model2=model2,n=n,avg=mean(results),std=sd(results)))
      }
    }
  }
  if(m==1){
    result.summary.list=data.frame(m1.avg=results.summary$avg,
                                   m1.std=results.summary$std)
  }else{
    result.summary.list=cbind(result.summary.list,results.summary$avg,results.summary$std)
  }
}
model1.list=results.summary$model1
model2.list=results.summary$model2
n.list=results.summary$n
result.summary.list=cbind(model1.list,model2.list,n.list,result.summary.list)
colnames(result.summary.list)=c("idx1","idx2","n",
                                "m=1.avg","m=1.std",
                                "m=2.avg","m=2.std",
                                "m=3.avg","m=3.std")




## GMDDNet and GSIR (data from Table 6 of Chen et al. (2024))

results.avg.gmdd=c(0.88,0.89,0.93,
                   0.93,0.95,0.97,
                   0.86,0.93,0.96,
                   0.50,0.66,0.74,
                   0.81,0.87,0.90,
                   0.88,0.92,0.97,
                   0.34,0.72,0.88,
                   0.70,0.81,0.95,
                   0.73,0.79,0.85)

results.std.gmdd=c(0.01,0.01,0.00,
                   0.01,0.01,0.00,
                   0.06,0.03,0.01,
                   0.05,0.03,0.01,
                   0.03,0.02,0.02,
                   0.09,0.08,0.01,
                   0.08,0.05,0.04,
                   0.04,0.03,0.00,
                   0.08,0.06,0.12)


results.avg.gsir=c(0.92,0.92,0.93,
                   0.92,0.92,0.92,
                   0.61,0.59,0.61,
                   0.38,0.42,0.43,
                   0.70,0.72,0.72,
                   0.57,0.55,0.57,
                   0.11,0.14,0.19,
                   0.61,0.61,0.62,
                   0.62,0.64,0.59)

results.std.gsir=c(0.00,0.01,0.00,
                   0.00,0.00,0.00,
                   0.05,0.12,0.02,
                   0.04,0.03,0.02,
                   0.02,0.01,0.02,
                   0.03,0.02,0.03,
                   0.04,0.02,0.02,
                   0.02,0.02,0.01,
                   0.08,0.03,0.09)



## Combine results


results.all=cbind(result.summary.list,
                  gmdd.avg=results.avg.gmdd,
                  gmdd.std=results.std.gmdd,
                  stonet.avg=results.avg.stonet,
                  stonet.std=results.std.stonet,
                  gsir.avg=results.avg.gsir,
                  gsir.std=results.std.gsir)

results.all



## Export to LaTeX file

results.all.string=data.frame()

for(i in 1:nrow(results.all)){
  results.all.string=rbind(results.all.string,
                           list(setting=paste(results.all$idx1[i],results.all$idx2[i],sep=""),
                                n=results.all$n[i],
                                benn1=paste(format(round(results.all$`m=1.avg`[i],2),nsmall=2),"(",
                                            format(round(results.all$`m=1.std`[i],2),nsmall=2),")",sep=""),
                                benn2=paste(format(round(results.all$`m=2.avg`[i],2),nsmall=2),"(",
                                            format(round(results.all$`m=2.std`[i],2),nsmall=2),")",sep=""),
                                benn3=paste(format(round(results.all$`m=3.avg`[i],2),nsmall=2),"(",
                                            format(round(results.all$`m=3.std`[i],2),nsmall=2),")",sep=""),
                                gmdd=paste(format(round(results.all$`gmdd.avg`[i],2),nsmall=2),"(",
                                           format(round(results.all$`gmdd.std`[i],2),nsmall=2),")",sep=""),
                                stonet=paste(format(round(results.all$`stonet.avg`[i],2),nsmall=2),"(",
                                             format(round(results.all$`stonet.std`[i],2),nsmall=2),")",sep=""),
                                gsir=paste(format(round(results.all$`gsir.avg`[i],2),nsmall=2),"(",
                                           format(round(results.all$`gsir.std`[i],2),nsmall=2),")",sep="")
                           ))
}

results.all.string

knitr::kable(results.all.string,"latex")



