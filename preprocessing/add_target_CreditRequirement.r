logName = "CreditRequirement"

dat_preprocessed_ = read.csv(sprintf("../logdata/%s.csv", logName), header = TRUE, sep = ";")
dat_preprocessed_$time = as.POSIXct(dat_preprocessed_$Complete.Timestamp,"%Y-%m-%d %H:%M:%S",tz = 'UTC')
xx = as.POSIXct(strftime(dat_preprocessed_$time, format="%H:%M:%S",tz = 'UTC'),format="%H:%M:%S")
plot(xx)
dat_preprocessed_$Complete.Timestamp = NULL

dat_preprocessed = dat_preprocessed_[with(dat_preprocessed_, order(case_id, time, method = "radix")), ]
sequence_length_dt <- as.data.frame(table(dat_preprocessed$case_id))
colnames(sequence_length_dt) <- c('case_id', 'seq_length')
  
regression_activities = c("Acceptance_of_requests",
                              "Collection_of_documents",
                              "Completeness_check",
                              "Credit_worthiness_check",
                              "Collateral_check",
                              "Credit_committee",
                              "Requirements_review")
#gateways = c("x11","x21","x31")

# extract cycle times for regression and branching labels for classification
target = sequence_length_dt
target$seq_length = NULL

for (regression_activitity in regression_activities) {
  target[,(ncol(target)+1)] = -1
  names(target)[ncol(target)] = regression_activitity
}

for (gateway in gateways) {
  target[,(ncol(target)+1)] = -1
  names(target)[ncol(target)] = gateway
}

for(i in 1:nrow(target)) {
  print(i)
  current_case = target$case_id[i]
  events_current_case=which(dat_preprocessed$case_id == current_case)
  # REGRESSION
  for(regression_activitity in regression_activities) {
    cycle_time=c()
    for(j in events_current_case) {
      if(j!=1) {
        if(dat_preprocessed$Activity[j] == regression_activitity & dat_preprocessed$case_id[j-1] == dat_preprocessed$case_id[j]) {
          cycle_time = c(cycle_time,dat_preprocessed$timesincecasestart[j] - dat_preprocessed$timesincecasestart[j-1])
          target[i,which(names(target)==regression_activitity)] = mean(cycle_time) # if an activity is repeated multiple times, take an average of those
        }
      }
      
    }
  }
  
  # CLASSIFICATION
  # for(j in events_current_case) {
  #   if (j!= nrow(dat_preprocessed)) {
  #     if(dat_preprocessed$activity_name[j] == "A_PARTLYSUBMITTED" & dat_preprocessed$activity_name[j+1] == "A_DECLINED" & dat_preprocessed$case_id[j+1] == dat_preprocessed$case_id[j]) {
  #       target$x11[i] = 0 # X3 is true
  #     }
  #     
  #     if(dat_preprocessed$activity_name[j] == "A_PARTLYSUBMITTED" & dat_preprocessed$activity_name[j+1] == "A_PREACCEPTED" & dat_preprocessed$case_id[j+1] == dat_preprocessed$case_id[j]) {
  #       target$x11[i] = 1 # X3 is true
  #     }
  #     
  #     if(dat_preprocessed$activity_name[j] == "A_PREACCEPTED" & dat_preprocessed$activity_name[j+1] == "A_DECLINED" & dat_preprocessed$case_id[j+1] == dat_preprocessed$case_id[j]) {
  #       target$x21[i] = 0 # X3 is true
  #     }
  #     
  #     if(dat_preprocessed$activity_name[j] == "A_PREACCEPTED" & dat_preprocessed$activity_name[j+1] == "A_CANCELLED" & dat_preprocessed$case_id[j+1] == dat_preprocessed$case_id[j]) {
  #       target$x21[i] = 1 # X3 is true
  #     }        
  #     
  #     if(dat_preprocessed$activity_name[j] == "A_PREACCEPTED" & dat_preprocessed$activity_name[j+1] == "A_ACCEPTED" & dat_preprocessed$case_id[j+1] == dat_preprocessed$case_id[j]) {
  #       target$x21[i] = 2 # X3 is true
  #     }
  #     
  #     if(dat_preprocessed$activity_name[j] == "A_FINALIZED" & dat_preprocessed$activity_name[j+1] == "A_CANCELLED" & dat_preprocessed$case_id[j+1] == dat_preprocessed$case_id[j]) {
  #       target$x31[i] = 0 # X3 is true
  #     }
  #     
  #     if(dat_preprocessed$activity_name[j] == "A_FINALIZED" & dat_preprocessed$activity_name[j+1] == "A_APPROVED" & dat_preprocessed$case_id[j+1] == dat_preprocessed$case_id[j]) {
  #       target$x31[i] = 1 # X3 is true
  #     }
  #     
  #     if(dat_preprocessed$activity_name[j] == "A_FINALIZED" & dat_preprocessed$activity_name[j+1] == "A_REGISTERED" & dat_preprocessed$case_id[j+1] == dat_preprocessed$case_id[j]) {
  #       target$x31[i] = 1 # X3 is true
  #     }
  #     
  #     if(dat_preprocessed$activity_name[j] == "A_FINALIZED" & dat_preprocessed$case_id[j+1] != dat_preprocessed$case_id[j]) {
  #       target$x31[i] = 2 # X3 is true
  #     }
  #     
  #     if(dat_preprocessed$activity_name[j] == "A_FINALIZED" & dat_preprocessed$activity_name[j+1] == "A_ACTIVATED" & dat_preprocessed$case_id[j+1] == dat_preprocessed$case_id[j]) {
  #       target$x31[i] = 3 # X3 is true
  #     }
  #     
  #     if(dat_preprocessed$activity_name[j] == "A_FINALIZED" & dat_preprocessed$activity_name[j+1] == "A_DECLINED" & dat_preprocessed$case_id[j+1] == dat_preprocessed$case_id[j]) {
  #       target$x31[i] = 4 # X3 is true
  #     }
  #     
  #   }
  #   
  # }
}
  

# extract number of branches for each gateway
gateways_exits = c()
for(gateway in gateways){
  foo = which(target[,gateway] != -1)
  gateways_exits = c(gateways_exits,length(unique(target[foo,gateway])))
}
names(gateways_exits) = gateways
gateways_exits = as.data.frame(gateways_exits)

write.csv(target, sprintf("../logdata/target/target_%s.csv", logName), quote = FALSE, row.names = FALSE)
