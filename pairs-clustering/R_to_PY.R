source("pairs_pool.R")

# 取出要做 PT 的 61 支股票
temp_data <- sort(table(selected_stock_vec), decreasing = TRUE)
assets_in_pt <- names(temp_data[temp_data >= 6])

train_data_PT <- data_train[data_train$Ticker %in% assets_in_pt, ]
test_data_PT  <- data_test[data_test$Ticker %in% assets_in_pt, ]

# 輸出 csv 檔案
write.csv(train_data_PT, "train_PT.csv", row.names = FALSE)
write.csv(test_data_PT, "test_PT.csv", row.names = FALSE)
