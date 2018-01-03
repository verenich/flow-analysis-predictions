library(reshape2)
library(ggplot2)

target = read.csv("target_minit_invoice_10.csv")

regression_activities = c(
  "Status_change_to_Being_Approved",
  "Check_order_numbers",  "Compare_of_sums",
  "Check_cost_center",  "Get_lowest_approval_level",
  "Approving_on_specific_level",  "Manual_identification_CC",
  "Manual_enter_the_order_number",  "Check_whether_the_total_approval",
  "Shift_to_higher_level",  "Status_change_to_Accounted",
  "Process_end",  "Status_change_to_Approved",
  "Invoice_accounting_confirmation",  "Invoice_accounting" )

max_len = 10
foo = as.data.frame(matrix(nrow = length(regression_activities), ncol = max_len))
names(foo)[1] = "activity"
foo$activity = regression_activities

foo_m = foo
foo_M = foo

for (i in 2:max_len) {
  #print(i)
  # names(foo)[1+3*(i-2)] = sprintf("pred_MAE_%s",i)
  # names(foo)[2+3*(i-2)] = sprintf("mean_MAE_%s",i)
  # names(foo)[3+3*(i-2)] = sprintf("mean_CT_%s",i)
  names(foo)[i] = i
  names(foo_m)[i] = i
  names(foo_M)[i] = i
  
  dat = read.csv(sprintf("res_minit_invoice_10_%s_30.csv", i))
  dat_m = read.csv(sprintf("res_minit_invoice_10_%s_3000000.csv", i))

  dat = merge(dat, target, by="Case.ID")
  dat_m = merge(dat_m, target, by="Case.ID")
  
  library(hydroGOF)
  for (act in regression_activities) {
    #print(act)
    gt = which(colnames(dat) == sprintf("%s.y",act))
    pred = which(colnames(dat) == sprintf("%s.x",act))
    foo[which(foo$activity==act),i] = mae(dat[which(dat[,gt]!=-1),gt],dat[which(dat[,gt]!=-1),pred])
    foo_m[which(foo$activity==act),i] = mae(dat[which(dat_m[,gt]!=-1),gt],dat_m[which(dat_m[,gt]!=-1),pred])
    foo_M[which(foo$activity==act),i] = mean(dat[which(dat[,gt]!=-1),gt])
  }
}

foo2 = melt(foo,id.vars = c("activity"), variable.name = "prefix_length")
foo2$var = "pred_MAE"
foo2$prefix_length = as.numeric(as.character(foo2$prefix_length))

foo2_m = melt(foo_m,id.vars = c("activity"), variable.name = "prefix_length")
foo2_m$var = "pred_mean"
foo2_m$prefix_length = as.numeric(as.character(foo2_m$prefix_length))

foo2_M = melt(foo_M,id.vars = c("activity"), variable.name = "prefix_length")
foo2_M$var = "mean_CT"
foo2_M$prefix_length = as.numeric(as.character(foo2_M$prefix_length))

foo3 = rbind(foo2,foo2_m,foo2_M)
ggplot(foo3, aes(x = prefix_length, y = value, color = var)) + geom_point() + geom_line() + facet_wrap(~ activity, ncol = 3, scales = "free")
