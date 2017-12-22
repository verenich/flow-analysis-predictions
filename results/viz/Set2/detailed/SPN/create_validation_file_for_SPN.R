result_files <- list.files()[grep(paste("^predictionResults_(?=.*\\.csv)", sep=''), list.files(), perl=TRUE)]

for (filename in result_files) {
  data = read.csv(filename,header = TRUE,sep = ";")
  datasetname = strsplit(filename,"gspn_")[[1]][2]
  dataset = strsplit(datasetname,".csv")[[1]]
  if(dataset == "BPI2012A") max_prefix_length=7
  if(dataset == "BPI2012O") max_prefix_length=4
  if(dataset == "BPI2012W") max_prefix_length=10
  if(dataset == "BPI2012W_no_dup") max_prefix_length=10
  if(dataset == "CreditRequirement") max_prefix_length=7
  if(dataset == "helpdesk") max_prefix_length=4
  if(dataset == "hospital_billing_977") max_prefix_length=6
  if(dataset == "minit_invoice_10") max_prefix_length=20
  if(dataset == "traffic_fines_139") max_prefix_length=8
  max_prefix_length = min(max_prefix_length, length(grep("Iteration", names(data), value = F))) # in case we have less prefixes in the result file than in FA
  
  df <- data.frame(matrix(ncol = 7, nrow = max_prefix_length-1))
  colnames(df) = c("dataset","method","cls","nr_events","metric","score","nr_cases")
  df$dataset = dataset
  df$method = "Rogge-Solti"
  df$cls = "SPN"
  df$metric = "mae"
  
  nr_events = 2:max_prefix_length
  columns_to_use = grep("Error.Pnetconstrained", names(data), value = F)
  
  for (pref_len in nr_events) {
    df$nr_events[pref_len-1] = pref_len
    df$score[pref_len-1] = mean(data[,columns_to_use[pref_len]],na.rm = TRUE)/1000 # make seconds
    df$nr_cases[pref_len-1] = sum(data[,columns_to_use[pref_len]] >= 0,na.rm = TRUE)
  }
  
  write.csv(df,file = sprintf("../../validation_SPN_%s.csv", dataset), row.names = FALSE, quote = FALSE)
}
  
  