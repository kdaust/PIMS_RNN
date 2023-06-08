library(data.table)
library(ncdf4)
library(reticulate)
library(Rlibeemd)

##I'm doing the decomposition in R because the ceedman package in R is faster
nc_data <- nc_open("../cesar_tower_meteo_lb1_t10_v1.2_201501.nc")
wind <- ncvar_get(nc_data, 'F')
wind <- t(wind)
wind_ts <- wind[,2]

time <- ncvar_get(nc_data, 'time')
date <- ncvar_get(nc_data, "date")

date <- date[!is.na(wind_ts)]
time <- time[!is.na(wind_ts)]
wind_ts <- wind_ts[!is.na(wind_ts)]


wind_dat <- fread("Blood Tribe Ag. Project IMCIN.csv")
plot(wind_dat$ws, type = "l")
out <- ceemdan(wind_dat$ws, num_imfs = 6, noise_strength = 0.1)
mnvr <- data.table()
for(i in 1:6){
  mn <- mean(out[,i])
  vr <- var(out[,i])
  out[,i] <- (out[,i]-mn)/vr
  temp <- data.table(Group = i, Mean = mn, Var = vr)
  mnvr <- rbind(mnvr,temp)
}

wind_dat <- fread("Blood Tribe Ag. Project IMCIN_test.csv")
plot(wind_dat$ws, type = "l")
out <- ceemdan(wind_dat$ws, num_imfs = 6, noise_strength = 0.1)

plot(out[,6])


for(i in 1:6){
  out[,i] <- (out[,i]-mnvr[i,Mean])/mnvr[i,Var]
}
wind_data <- out
plot(rowSums(wind_data), type = "l")
colnames(wind_data) <- paste0("Wind_",1:6) ##wind_data now ready for python

df <- wind_dat[,.(month,day,hour)]

df[, Hour := Time %% 24]
df[, SubHour := Time %% 1]
df[, Day := as.numeric(as.factor(Date))]
df[, Date := NULL] ## covariates now ready

###########################################
# plotting results
all_res <- py$results
save(all_res, file = "Final_mod_24.Rdata")
bnum <- 5
#all_res$`5`$preds <- pred
#all_res$`5`$truth <- gt
pred <- rbind(all_res$`0`$preds[bnum,],all_res$`1`$preds[bnum,],all_res$`2`$preds[bnum,],
              all_res$`3`$preds[bnum,],all_res$`4`$preds[bnum,],all_res$`5`$preds[bnum,]) #,all_res$`5`$preds[bnum,]

gt <- rbind(all_res$`0`$truth[bnum,],all_res$`1`$truth[bnum,],all_res$`2`$truth[bnum,],
            all_res$`3`$truth[bnum,],all_res$`4`$truth[bnum,],all_res$`5`$truth[bnum,])

# batch_comb <- all_res$`0`$truth[1,]
# for(i in 2:12){
#   batch_comb <- c(batch_comb,all_res$`0`$truth[i,])
# }
# 
# batch_comb_pred <- all_res$`0`$preds[1,]
# for(i in 2:12){
#   batch_comb_pred <- c(batch_comb_pred,all_res$`0`$preds[i,])
# }

# plot(batch_comb_pred, type = "l")
# plot(batch_comb, type = "l", col = "red")

gt_sum <- colSums(gt)
pred_sum <- colSums(pred)

plot(gt_sum, type = "l")
lines(pred_sum, type = "l", col = "red")

for(i in 1:6){
  plot(pred[i,], type = "l")
  lines(gt[i,], type = "l", col = "red")
}

gt <- py$ground_truth
pred <- py$pred_dat

batch = 4
plot(gt[batch,], type = "l")
lines(pred[batch,], type = "l", col = "red")


res = py$pred_dat
gt = py$ground_truth
plot(res[2,], type = "l")
lines(res[2,], type = "l", col = "red")
