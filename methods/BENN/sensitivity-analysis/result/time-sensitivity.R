model1=4
model2=1

time_benn=matrix(NA,12,100)
for(i in 0:11){
    for(t in 1:100){
      time_benn[i+1,t]=read.csv(paste("./results-BENN-sensitivity-1/result-4-",model2,"-",i,"-5000/time_",t,".csv",sep=""))[,-1][2]
    }
}

rowMeans(time_benn)





model1=4
model2=2

time_benn=matrix(NA,12,100)
for(i in 0:11){
  for(t in 1:100){
    time_benn[i+1,t]=read.csv(paste("./results-BENN-sensitivity-1/result-4-",model2,"-",i,"-5000/time_",t,".csv",sep=""))[,-1][2]
  }
}

rowMeans(time_benn)

