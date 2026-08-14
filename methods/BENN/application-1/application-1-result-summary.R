time.BENN1=read.csv("./application-results/time-1-1.csv")[,2]
time.BENN2=read.csv("./application-results/time-1-2.csv")[,2]
time.BENN3=read.csv("./application-results/time-1-10.csv")[,2]
time.BENN4=read.csv("./application-results/time-1-50.csv")[,2]
time.BENN5=read.csv("./application-results/time-1-100.csv")[,2]
time.BENN6=read.csv("./application-results/time-1-200.csv")[,2]
time.GMDD=read.csv("./application-results/time-gmdd-1-correct.csv")[,2]
time.SN=read.csv("./application-results/time-stonet-1.csv")[,2]
load("./application-results/results-GSIR.Rdata")
time.GSIR=time.save[c(3,1)]


output=cbind(rbind(time.BENN1,time.BENN2,time.BENN3,time.BENN4,time.BENN5,time.BENN6,time.GMDD,time.SN,time.GSIR)[,2],
             rbind(time.BENN1,time.BENN2,time.BENN3,time.BENN4,time.BENN5,time.BENN6,time.GMDD,time.SN,time.GSIR)[,1])
output
output

output.round=cbind(round(output[,1],3),
                   round(output[,2],3))

knitr::kable(output.round,"latex")



