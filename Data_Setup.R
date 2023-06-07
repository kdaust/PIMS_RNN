library(data.table)
library(ncdf4)
library(reticulate)
library(Rlibeemd)

##I'm doing the decomposition in R because the ceedman package in R is faster
nc_data <- nc_open("cesar_tower_meteo_lb1_t10_v1.2_201501.nc")
wind <- ncvar_get(nc_data, 'F')
wind <- t(wind)
wind_ts <- wind[,2]

time <- ncvar_get(nc_data, 'time')
date <- ncvar_get(nc_data, "date")

date <- date[!is.na(wind_ts)]
time <- time[!is.na(wind_ts)]
wind_ts <- wind_ts[!is.na(wind_ts)]
out <- ceemdan(wind_ts, num_imfs = 6, noise_strength = 0.1)

wind_data <- out[,1:5]
plot(rowSums(wind_data), type = "l")
colnames(wind_data) <- paste0("Wind_",1:5) ##wind_data now ready for python

df <- data.table(Date = date, Time = time)
df[, Hour := Time %% 24]
df[, SubHour := Time %% 1]
df[, Day := as.numeric(as.factor(Date))]
df[, Date := NULL] ## covariates now ready