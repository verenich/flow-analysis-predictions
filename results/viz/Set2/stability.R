library(hydroGOF)
library(plyr)
library(ggplot2)
library(stargazer)
library(extrafont)
library(gridExtra)
font_install("fontcm")
loadfonts()

setwd("detailed/")

remove_prefix_nr_from_caseid <- function(row) {
  case_id <- row["case_id"]
  parts <- strsplit(case_id, "_")[[1]]
  cut_length <- ifelse(as.numeric(row["nr_events"]) < 2, length(parts), length(parts)-1)
  return(paste(parts[1:cut_length], collapse="_"))
}

result_files1 <- list.files()[grep(paste("^validation_FA2_(?=.*\\.csv)", sep=''), list.files(), perl=TRUE)]
result_files2 <- list.files()[grep(paste("^validation_direct_(?=.*\\.csv)", sep=''), list.files(), perl=TRUE)]

data <- data.frame()
for (filename in result_files1) {
  
  if (!grepl("index", filename)) {
    tmp <- read.table(filename, sep=";", header=T, na.strings=c("None"))
    tmp$case_id <- as.character(tmp$case_id)
    tmp$nr_events <- as.numeric(as.character(tmp$nr_events))
    tmp$case_id <- apply(tmp, 1, remove_prefix_nr_from_caseid)
    datasetname = strsplit(filename,"_zero")[[1]][1]
    tmp$dataset = strsplit(datasetname,"FA2_")[[1]][2]
    tmp$method2 = "FA"
    tmp$method3 = strsplit(filename,"_")[[1]][length(strsplit(filename,"_")[[1]])]
    data <- rbind(data, tmp)
  }
  
}

for (filename in result_files2) {
  
  if (!grepl("index", filename)) {
    tmp <- read.table(filename, sep=";", header=T, na.strings=c("None"))
    tmp$case_id <- as.character(tmp$case_id)
    tmp$nr_events <- as.numeric(as.character(tmp$nr_events))
    tmp$case_id <- apply(tmp, 1, remove_prefix_nr_from_caseid)
    datasetname = strsplit(filename,"_zero")[[1]][1]
    tmp$dataset = strsplit(datasetname,"direct_")[[1]][2]
    tmp$method2 = "regression"
    tmp$method3 = strsplit(filename,"_")[[1]][length(strsplit(filename,"_")[[1]])]
    data <- rbind(data, tmp)
  }
  
}


data$method3[data$method3 == "3000000.csv"] = "mean"
data$method3[data$method3 == "30.csv" & data$method2 == "FA"] = "predictive"
data$method3[data$method3 == "30.csv" & data$method2 == "regression"] = "simple"
data$method3[data$method3 == "5.csv"] = "adaptive"
data$method3[data$method3 == "10.csv"] = "adaptive"
data$method3[data$method3 == "20.csv"] = "adaptive"
data$method3[data$method3 == "sqrt.csv"] = "adaptive"

data$method = paste(data$method3, data$method2, sep = "_")

data$actual = data$actual / (3600*24) # make days
data$predicted = data$predicted / (3600*24) # make days

data$nr_events_num <- data$nr_events
data$nr_events <- as.factor(data$nr_events)
data$method <- as.character(data$method)


dt_aucs <- ddply(data, .(dataset, nr_events, method, cls), summarize, count=length(actual), auc=mae(actual, predicted))
ggplot(dt_aucs, aes(x=as.numeric(nr_events), y=auc, color=method)) + geom_point() + geom_line() + theme_bw() + facet_wrap(~dataset, scales="free")

# stability
dt_stability_cases <- ddply(data, .(dataset, method, cls, case_id), summarize, std=mean(abs(diff(predicted-actual)*nr_events_num)))
dt_stability_cases <- dt_stability_cases[!is.na(dt_stability_cases),]
dt_stability <- ddply(dt_stability_cases, .(dataset, method, cls), summarize, mean_std=mean(std, na.rm=TRUE), std_std=sd(std, na.rm=TRUE))
#dt_stability <- dt_stability[!is.na(dt_stability),]
dt_stability <- dt_stability[1:nrow(dt_stability)-1,]
write.table(dt_stability, "instability_orig.csv", sep=";", row.names=FALSE, col.names=TRUE)

ggplot(dt_stability, aes(x=factor(method), y=mean_std, fill=method, group=method)) + geom_bar(stat="identity") + theme_bw() + facet_wrap(~dataset, scales="free")
#ggplot(dt_stability, aes(x=factor(method), y=std_std, fill=method, group=method)) + geom_bar(stat="identity") + theme_bw() + facet_wrap(~dataset, scales="free")
pdf(file="instability.pdf",family="CM Roman",width=12,height=10)
p = ggplot(dt_stability, aes(x=factor(method), y=mean_std, fill=method, group=method)) + 
  geom_bar(stat="identity") + 
  #geom_errorbar(aes(ymin=mean_std, ymax=mean_std+std_std)) + 
  theme_bw() + facet_wrap(~dataset, scales="free")
print(p)
dev.off()
embed_fonts("instability.pdf",outfile="instability.pdf")

stargazer(dt_stability, summary=FALSE)