# Import necessary packages and customized functions
{
    library(dplyr)
    
    source("./utils/utils_data_preprocessing.R")
}

# Load ticker data
component_list <- read.csv("./data_file/SP500_list.csv")

# 1. Load all stock price data / 
# 2. extract stock price data for the specified start and end dates & 
# remove stocks that do not meet the start and end dates / 
# 3. add a time column
data_stock <- get(load("cache_file/all_price.rData")) %>% 
    data_Adjust(data_list = ., date_start = "2015-01-02", date_end = "2024-12-31") %>% 
    feature_Append(data_list = .)

# save(data_stock, file = "data_stock_cache.rData")

#==========================================#
#             Split Data Set               #
#==========================================#
#   in-sample    : 2015-01-02~2021-12-31   #
#   out-of-sample: 2022-01-03~2024-07-01   #
#==========================================#

date_start_train <- as.character.Date("2015-01-02")
date_end_train   <- as.character.Date("2021-12-31")

date_start_test  <- as.character.Date("2022-01-03")
date_end_test    <- as.character.Date("2024-12-31")

data_split <- in_out_sample_Split(
    data_list = data_stock, 
    date_start_train = date_start_train, date_end_train = date_end_train, 
    date_start_test  = date_start_test,  date_end_test  = date_end_test
)

data_train <- data_split$train %>% do.call(rbind, .)
data_test  <- data_split$test %>% do.call(rbind, .)

# component_list[component_list$Symbol %in% remove_list, ] %>% 
#     write.csv(., file = "remove_list.csv", row.names = FALSE)

# write.csv(data_train, file = "train.csv", row.names = FALSE)
# write.csv(data_test,  file = "test.csv",  row.names = FALSE)
