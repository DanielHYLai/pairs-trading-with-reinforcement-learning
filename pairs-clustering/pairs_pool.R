# Import necessary packages and customized functions
{
    library(dplyr)
    library(readr)
    
    source("./utils/utils_frequency_matrix.R")
    source("./utils/utils_clustering_method.R")
    source("./utils/utils_pairs_pool.R")
    
    set.seed(123)
}

# Load data
{
    data_stock <- get(load("./cache_file/data_stock_cache.rData"))
    data_train <- readr::read_csv("./data_file/train.csv", show_col_types = FALSE)
    data_test  <- read.csv("./data_file/test.csv")
}

# Perform a test on all paired results
all_test_result <- all_possible_coTest(
    data_list = data_train, 
    ticker = names(data_stock), 
    load_from_cache = TRUE
)

# Store TFM and clustering results for tau = 8 to 16 in list
threshold_FM_all <- list()

for (tau in seq(0.1, 0.2, length.out = 10)[-8]) {
    cat("tau =", floor(length(all_test_result[[1]]) * tau), "...\n")
    
    ## Generate the TFM and remove BRK-B and BF-B
    threshold_FM <- matrix(
        0, nrow = length(data_stock), ncol = length(data_stock), 
        dimnames = list(names(data_stock), names(data_stock))
    ) %>% 
        FM_generator(result_test = all_test_result, 
                     result_mat  = ., 
                     discrete    = FALSE, 
                     threshold_q = tau) %>% 
        .[!row.names(.) %in% c("BRK-B", "BF-B"), 
          !colnames(.)  %in% c("BRK-B", "BF-B")]
    
    ## Fill the TFM, clustering result under K=2 and 3, and the criteria values
    threshold_FM_all[[
        as.character(floor(length(all_test_result[[1]]) * tau))
        ]] <- list(
        FM    = threshold_FM, 
        `K=2` = fundamental_Cluster(df = threshold_FM, K = 2, 
                                    assigned_method = "HC", seed = 123), 
        `K=3` = fundamental_Cluster(df = threshold_FM, K = 3, 
                                    assigned_method = "HC", seed = 123), 
        criteria_value = clustering_method_Optimal(
            df = threshold_FM, K = 3, assigned_method = "HC", 
            seed = 123, visualized = FALSE
        )
    )
}

# Search for the cluster with the fewest elements 
# (the cluster with a more significant cointegration property)
threshold_FM_all <- min_cluster_Find(data_list = threshold_FM_all)

TFM_CvalueTB <- lapply(names(threshold_FM_all), function(name) {
    df <- threshold_FM_all[[name]][["criteria_value"]]
    df$source <- paste0("TFM_", name)
    
    return(df)
}) %>% 
    do.call(rbind, .)

CvalueTB <- TFM_CvalueTB
row.names(CvalueTB) <- paste(CvalueTB$source, c("K=2", "K=3"), sep = "_")
CvalueTB <- CvalueTB[, -5]

# Filter the required grouping results from table for criteria value (greater than median)
selected_stock_vec <- c()
for (name in names(CvalueTB)) {
    selected_stock_vec <- c(
        selected_stock_vec, 
        CvalueTB[CvalueTB[[name]] > median(CvalueTB[[name]]), ] %>% 
            row.names() %>% strsplit("_") %>% 
            selected_stock_vec_Gen(input_data = .)
    )
}

# Draw a bar chart of the number of times a stock is selected
stock_pool_Barchart(input_data = sort(table(selected_stock_vec), decreasing = TRUE),
                    criteria_name = "Selected from Four Criteria Values")
