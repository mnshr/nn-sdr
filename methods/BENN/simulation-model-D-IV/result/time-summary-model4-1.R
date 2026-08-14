library(gplm)
model1=4
model2=1

results.summary=data.frame()

for(n in 1000*(1:8)){
  time_benn1=c()
  time_benn2=c()
  time_benn3=c()
  time_gmdd=c()
  time_stonet=c()
  time_gsir=c()
  for(t in 1:100){
    time_benn1_current=read.csv(paste("./results-BENN-unified-std/result-",model1,"-",model2,"-",1,"-",n,"/time_",t,".csv",sep=""))[,-1]
    time_benn1=c(time_benn1,time_benn1_current[2])
    time_benn2_current=read.csv(paste("./results-BENN-unified-std/result-",model1,"-",model2,"-",2,"-",n,"/time_",t,".csv",sep=""))[,-1]
    time_benn2=c(time_benn2,time_benn2_current[2])
    time_benn3_current=read.csv(paste("./results-BENN-unified-std/result-",model1,"-",model2,"-",1000,"-",n,"/time_",t,".csv",sep=""))[,-1]
    time_benn3=c(time_benn3,time_benn3_current[2])
    time_gmdd_current=read.csv(paste("./results-GMDD-correct/result-",model1,"-",model2,"-",n,"/time_",t,".csv",sep=""))[,-1]
    time_gmdd=c(time_gmdd,time_gmdd_current[2])
    time_stonet_current=read.csv(paste("./results-StoNet-25/result-",model1,"-",model2,"-",n,"/time_",t,".csv",sep=""))[,-1]
    time_stonet=c(time_stonet,time_stonet_current[2])
    try({
      load(paste("./results-GSIR/model",model1,"-",model2,"-",n,"/res_",t,".Rdata",sep=""))
      time_gsir=c(time_gsir,time.save[1])
    })
  }
  results.summary=rbind(results.summary,
                        list(model=model2,
                             n=n,
                             benn1=mean(time_benn1),
                             benn2=mean(time_benn2),
                             benn3=mean(time_benn3),
                             gmdd=mean(time_gmdd),
                             stonet=mean(time_stonet),
                             gsir=mean(time_gsir)))
  
  print(results.summary)
}




knitr::kable(t(results.summary),"latex",digits=2)








