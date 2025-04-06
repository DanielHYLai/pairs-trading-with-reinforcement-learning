# Import necessary packages
{
    library(doParallel)
    library(foreach)
    library(parallel)
    library(quantmod)
}

# Load the component list and obtain the tickers
component_list <- read.csv("data_file/SP500_list.csv")
tickers <- component_list$Symbol

# Obtain all stock price of the input ticker
num_cores <- parallel::detectCores() - 5
clusters  <- parallel::makeCluster(num_cores)
doParallel::registerDoParallel(clusters)
groups <- split(tickers, ceiling(seq_along(tickers) / 5))
data_stock <- foreach::foreach(group = groups, .combine = "c", .packages = "quantmod") %dopar% {
    lapply(group, function(ticker) {
        tryCatch({
            quantmod::getSymbols(ticker, src = "yahoo", auto.assign = FALSE)
        }, error = function(e) {
            message(paste("Error fetching data for ticker:", ticker))
            return(NULL)
        })
    })
}

parallel::stopCluster(clusters)

# Rename the index of result list
names(data_stock) <- tickers

# Export the result
save(data_stock, file = "all_price.rData")
