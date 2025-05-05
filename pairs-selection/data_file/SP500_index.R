library(quantmod)
library(dplyr)
library(lubridate)

data <- quantmod::getSymbols("^GSPC", from="2022-01-01", to="2024-12-31", auto.assign = FALSE)
date <- index(data)
data <- as.data.frame(data)
colnames(data) <- c("Open", "High", "Low", "Close", "Volume", "Adjusted")
data$Date <- date
data$Year <- year(date)
data$Month <- month(date)
write.csv(data, file="SP500-index-data.csv", row.names = FALSE)