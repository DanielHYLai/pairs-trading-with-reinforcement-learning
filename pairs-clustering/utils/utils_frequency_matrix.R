# Import necessary packages
{
    library(parallel)
    library(doParallel)
    library(foreach)
    library(fUnitRoots)
    
    options(warn = -1)
}

# Perform cointegration test on input pairs
all_possible_coTest <- function(data_list = data_train, 
                                ticker = names(data_stock), 
                                load_from_cache = TRUE) {

    if (load_from_cache == FALSE) {
        
        set.seed(123)
        
        ## Get all possible pairing combinations
        pairs_ticker_list <- combn(ticker, m = 2, simplify = FALSE)
        
        ## Set the number of cores & enable clusters
        num_cores <- parallel::detectCores() - 1
        cl <- parallel::makeCluster(num_cores)
        doParallel::registerDoParallel(cl)
        cat("Set up cores:", num_cores, "...\n")
        
        ## time-stamp
        flag_time <- Sys.time()
        cat("Start constructing the cointegration test ...\n")
        
        ## part of cointegration test
        all_test_result_parallel <- foreach::foreach(
            num = seq_along(pairs_ticker_list), 
            .packages = c("fUnitRoots", "stats"), 
            .combine = "c"
        ) %dopar% {
            
            ## Get the stock price data of two stocks
            X <- data_list[data_list$Ticker == pairs_ticker_list[[num]][1], ]
            Y <- data_list[data_list$Ticker == pairs_ticker_list[[num]][2], ]
            test_result <- c()  # A vector storing the test results
            
            ## Get the closing prices of two stocks for each month
            for (year in unique(X$Year)) {
                for (month in unique(X[X$Year == year, "Month"])) {
                    X_mask <- X[(X$Year == year) & (X$Month == month), "Close"]
                    Y_mask <- Y[(Y$Year == year) & (Y$Month == month), "Close"]
                    model_res <- stats::residuals(stats::lm(Y_mask ~ X_mask))
                    model_adf <- fUnitRoots::adfTest(x = model_res, lags = 4, type = "nc")
                    test_result <- c(test_result, model_adf@test$`p.value`)
                }
            }
            
            list(setNames(list(test_result), 
                          paste(pairs_ticker_list[[num]][1], 
                                pairs_ticker_list[[num]][2], sep = "-")))
        }
        
        ## time-stamp
        print(Sys.time() - flag_time)
        
        ## Disable Parallel Processing Unit
        parallel::stopCluster(cl)
        cat("Already shutdown cores.\n")
        
        ## Rename the list index
        all_test_result <- Map(function(x) as.vector(unlist(x)), all_test_result_parallel)
        names(all_test_result) <- sapply(all_test_result_parallel, names)
        cat("Rename index ...\n")
        
        ## Storing cache data
        cat("Save cache ...")
        save(all_test_result, file = "all_possible_coTest_cache.rData")
        cat("Done.")
        
        return(all_test_result)
    }
    
    else {
        all_test_result <- get(load(file = "./cache_file/all_possible_coTest_cache.rData"))
        print("Load cache successfully.")
        
        return(all_test_result)
    }
}


# Discretize the test results
test_result_Discretize <- function(obj = all_test_result, alpha = 0.01) {
    obj <- lapply(obj, function(x) {
        ifelse(x > alpha, 0, 1)
    })
    
    return(obj)
}


# Generate frequency matrix
FM_generator <- function(result_test, result_mat, discrete = FALSE, threshold_q = NULL) {
    
    for (idx in seq_along(result_test)) {
        
        ## Split pairs name
        split_idx <- unlist(strsplit(names(result_test)[idx], split = "-"))
        
        idx_1 <- which(row.names(result_mat) == split_idx[1])
        idx_2 <- which(row.names(result_mat) == split_idx[2])
        
        ## Generate discretize frequency matrix
        if (discrete == TRUE & is.null(threshold_q) == TRUE) {
            
            ## Sum up the significant results
            result_mat[idx_1, idx_2] <- sum(result_test[[idx]])
            result_mat[idx_2, idx_1] <- sum(result_test[[idx]])
        }
        
        ## Generate threshold frequency matrix
        else if (discrete == FALSE & is.null(threshold_q) == FALSE) {
            
            ## the p-value of the cointegration test under tau
            threshold_loc <- floor(length(result_test[[idx]]) * threshold_q)
            value <- result_test[[idx]][threshold_loc]
            
            result_mat[idx_1, idx_2] <- value
            result_mat[idx_2, idx_1] <- value
        }
    }
    
    return(result_mat)
}