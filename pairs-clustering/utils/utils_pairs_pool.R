# Search for the cluster with the fewest elements 
# (the cluster with a more significant cointegration property)
min_cluster_Find <- function(data_list) {
    
    for (idx in names(data_list)) {
        K2 <- data_list[[idx]]$`K=2`
        minimal_cluster <- names(K2[K2 == which.min(table(K2))])
        data_list[[idx]]$`K=2_minC` <- minimal_cluster
        
        K3 <- data_list[[idx]]$`K=3`
        minimal_cluster <- names(K3[K3 == which.min(table(K3))])
        data_list[[idx]]$`K=3_minC` <- minimal_cluster
    }
    
    return(data_list)
}


# Draw a bar chart of the number of times a stock is selected
stock_pool_Barchart <- function(input_data, criteria_name = "") {
    fig <- barplot(
        input_data,
        border = TRUE, 
        col = adjustcolor("#46A3FF", alpha.f = 1), 
        ylim = c(0, max(input_data) + 3), 
        xlab = "stock", 
        ylab = "count", 
        main = criteria_name
    )
    counts <- rev(table(input_data))
    loc <- c()
    start_idx <- 1
    for (end_idx in cumsum(counts)) {
        loc <- c(loc, median(fig[start_idx:end_idx]))
        start_idx <- end_idx + 1
    }
    text(loc, as.integer(names(counts)) + 0.5, labels = counts)
}


# Filter the required grouping results from table for criteria value
selected_stock_vec_Gen <- function(input_data) {
    
    selected_stock_vec <- c()
    
    for (idx in c(1:length(input_data))) {
        selected_stock_vec <- c(
            selected_stock_vec, 
            threshold_FM_all[[input_data[[idx]][2]]][[
                paste0(input_data[[idx]][3], "_minC")
            ]]
        )
    }
    
    return(selected_stock_vec)
}