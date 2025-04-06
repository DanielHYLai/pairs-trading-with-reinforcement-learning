# Import necessary packages
{
    library(lubridate)
    library(zoo)
}

# extract stock price data for the specified start and end dates & 
# remove stocks that do not meet the start and end dates
data_Adjust <- function(data_list = data_stock, date_start, date_end) {
    
    ## A vector storing the stocks to be removed
    remove_list <- c()
    
    for (idx in seq_along(data_list)) {
        
        ## Index of the start date
        index_start <- which(zoo::index(data_list[[idx]]) == date_start)
        
        ## Index of the end date
        index_end   <- which(zoo::index(data_list[[idx]]) == date_end)
        
        ## Retrieve stock price data between the specified start and end dates
        if (length(index_start) > 0 & length(index_end) > 0) {
            data_list[[idx]] <- data_list[[idx]][c(index_start:index_end), ]
        }
        
        ## Stocks that do not meet the start and end dates
        else {
            remove_list <- c(remove_list, names(data_list)[idx])
            data_list[[idx]] <- 0
        }
    }
    
    cat("Remove stock:", remove_list, "( total", length(remove_list), ")\n")
    
    ## Assign to the global variable
    remove_list <<- remove_list
    
    ## Remove the data from original data
    data_list <- data_list[lubridate::setdiff(names(data_list), remove_list)]
    
    return(data_list)
}


# Adding new features to original data
feature_Append <- function(data_list = data_stock) {
    
    ticker_names <- names(data_list)  # copy-write
    
    data_list <- lapply(seq_along(data_list), function(idx) {
        
        ## Added new columns: stock code, transaction date
        data <- data.frame(
            data_list[[idx]], 
            Ticker = names(data_list)[idx], 
            Date = as.Date(index(data_list[[idx]]))
        )
        
        ## Unify all column names of the original data
        colnames(data) <- c("Open", "High", "Low", "Close", "Volume", "Adjusted", 
                            "Ticker", "Date")
        
        ## Split year and month from date
        transform(data, Year = lubridate::year(Date), Month = lubridate::month(Date))
    })
    
    ## The new list will overwrite the old list's name, so there needs to re-fill it.
    names(data_list) <- ticker_names  # re-fill
    
    return(data_list)
}

# Split the data into in-sample and out-of-sample
in_out_sample_Split <- function(data_list = data_stock, 
                                date_start_train, date_end_train, 
                                date_start_test, date_end_test) {
    
    ## Store in-sample and out-of-sample lists
    train_data <- list()
    test_data  <- list()
    
    for (idx in seq_along(data_list)) {
        
        temp_data <- data_list[[idx]]  # Retrieve stock prices one by one
        
        ## Search the start and end dates of in-sample and out-of-sample
        index_start_train <- which(row.names(temp_data) == date_start_train)
        index_end_train   <- which(row.names(temp_data) == date_end_train)
        index_start_test  <- which(row.names(temp_data) == date_start_test)
        index_end_test  <- which(row.names(temp_data) == date_end_test)
        
        ## Retrieve stock price data between the specified start and end dates
        train_data[[names(data_list)[idx]]] <- temp_data[c(index_start_train:index_end_train), ]
        test_data[[names(data_list)[idx]]]  <- temp_data[c(index_start_test:index_end_test), ]
    }
    
    ## Output as a multi-layer list
    result <- list(
        train = train_data, 
        test  = test_data
    )
    
    return(result)
}
