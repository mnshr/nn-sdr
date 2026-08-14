results.summary=matrix(NA,10,9)
colnames(results.summary)=c("idx1","idx2","n",
                            "m=1.avg","m=1.std",
                            "m=2.avg","m=2.std",
                            "m=3.avg","m=3.std")

B0=matrix(0,nrow=20,ncol=2)
B0[1:2,1]=c(1,1)
B0[3:4,2]=c(1,3)
P.B0=B0%*%solve(t(B0)%*%B0)%*%t(B0)

count=0
model1=5
model2=3

for(n in 1000*(1:10)){
  count=count+1
  results.summary[count,1:3]=c(model1,model2,n)
  for(m in 1:3){
    res1=c()
    for(t in 1:100){
      try({
        z_pred=read.csv(paste("./results-BENN-linear/result-",model1,"-",model2,"-",m,"-",n,"/y_suff_",t,".csv",sep=""))[,-1]
        x_test=read.csv(paste("./data/model",model1,"-",model2,"-",n,"/x_test_",t,".csv",sep=""))[,-1]
        b1=lm(z_pred[,1]~.,data=x_test)$coef[-1]
        b2=lm(z_pred[,2]~.,data=x_test)$coef[-1]
        B=matrix(c(b1,b2),ncol=2)
        P.B=B%*%solve(t(B)%*%B)%*%t(B)
        res1=c(res1,norm(P.B-P.B0,"2"))
      })
    }
    results.summary[count,2*m+2]=mean(res1)
    results.summary[count,2*m+3]=sd(res1)
    print(results.summary)
  }
}

results.summary

save(results.summary,
     file="./results-model5-3.Rdata")



load("./results-model5-3.Rdata")

result.summary.list=data.frame(results.summary)





results.all.string=data.frame()

for(i in 1:nrow(result.summary.list)){
  results.all.string=rbind(results.all.string,
                           list(setting=result.summary.list$idx2[i],
                                n=result.summary.list$n[i],
                                benn1=paste(format(round(result.summary.list$`m.1.avg`[i],2),nsmall=2),"(",
                                            format(round(result.summary.list$`m.1.std`[i],2),nsmall=2),")",sep=""),
                                benn2=paste(format(round(result.summary.list$`m.2.avg`[i],2),nsmall=2),"(",
                                            format(round(result.summary.list$`m.2.std`[i],2),nsmall=2),")",sep=""),
                                benn3=paste(format(round(result.summary.list$`m.3.avg`[i],2),nsmall=2),"(",
                                            format(round(result.summary.list$`m.3.std`[i],2),nsmall=2),")",sep="")
                           ))
}




results.all.string


knitr::kable(results.all.string,"latex")


