# 載入套件 & 自訂義函式
{
    library(dplyr)
    library(readr)
    
    source("./utils/utils_frequency_matrix.R")
    source("./utils/utils_clustering_method.R")
    
    set.seed(123)
}

# 匯入資料
{
    data_stock <- get(load("./cache_file/data_stock_cache.rData"))
    data_train <- readr::read_csv("./data_file/train.csv", show_col_types = FALSE)
    data_test  <- read.csv("./data_file/test.csv")
}

# 執行所有配對的檢定結果
all_test_result <- all_possible_coTest(
    data_list = data_train, 
    ticker = names(data_stock), 
    load_from_cache = TRUE
)

# 將 alpha = 0.01 至 0.1 的 DFM 以及分群結果儲存至 list 
discrete_FM_all <- list()

for (alpha in c(0.01, 0.05)) {
    cat("alpha =", alpha, "...\n")
    discrete_FM <- matrix(
        0, nrow = length(data_stock), ncol = length(data_stock), 
        dimnames = list(names(data_stock), names(data_stock))
    ) %>% 
        FM_generator(result_test = test_result_Discretize(
            obj = all_test_result, alpha = alpha
            ), 
                     result_mat  = ., 
                     discrete    = TRUE) %>% 
        .[!row.names(.) %in% c("BRK-B", "BF-B"), 
          !colnames(.)  %in% c("BRK-B", "BF-B")]
    discrete_FM_all[[as.character(alpha)]] <- list(
        FM    = discrete_FM, 
        `K=2` = fundamental_Cluster(df = discrete_FM, K = 2, 
                                    assigned_method = "HC", seed = 123), 
        `K=3` = fundamental_Cluster(df = discrete_FM, K = 3, 
                                    assigned_method = "HC", seed = 123), 
        criteria_value = clustering_method_Optimal(
            df = discrete_FM, K = 3, assigned_method = "HC", 
            seed = 123, visualized = FALSE
        )
    )
}

# 將 tau = 8 至 16 的 TFM 以及分群結果儲存至 list 
threshold_FM_all <- list()

for (tau in seq(0.1, 0.2, length.out = 10)[-8]) {
    cat("tau =", floor(length(all_test_result[[1]]) * tau), "...\n")
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

discrete_FM_all  <- min_cluster_Find(data_list = discrete_FM_all)
threshold_FM_all <- min_cluster_Find(data_list = threshold_FM_all)

DFM_CvalueTB <- lapply(names(discrete_FM_all), function(name) {
    df <- discrete_FM_all[[name]][["criteria_value"]]
    df$source <- paste0("DFM_", name)
    
    return(df)
}) %>% 
    do.call(rbind, .)

TFM_CvalueTB <- lapply(names(threshold_FM_all), function(name) {
    df <- threshold_FM_all[[name]][["criteria_value"]]
    df$source <- paste0("TFM_", name)
    
    return(df)
}) %>% 
    do.call(rbind, .)

CvalueTB <- TFM_CvalueTB
row.names(CvalueTB) <- paste(CvalueTB$source, c("K=2", "K=3"), sep = "_")
CvalueTB <- CvalueTB[, -5]

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

# # 根據最佳 criteria values 挑選的方法
# selected_stock_vec <- c(
#     discrete_FM_all[["0.01"]][["K=2_minC"]], 
#     discrete_FM_all[["0.05"]][["K=2_minC"]], 
#     discrete_FM_all[["0.01"]][["K=3_minC"]], 
#     
#     threshold_FM_all[["8"]][["K=2_minC"]], 
#     threshold_FM_all[["14"]][["K=2_minC"]], 
#     threshold_FM_all[["13"]][["K=3_minC"]], 
#     threshold_FM_all[["14"]][["K=3_minC"]], 
#     threshold_FM_all[["15"]][["K=3_minC"]]
# )
# 
# stock_pool_Barchart(input_data = sort(table(selected_stock_vec), decreasing = TRUE), 
#                     criteria_name = "Selected from the Four Highest Criteria Values")

selected_stock_vec_Gen <- function(input_data) {
    
    selected_stock_vec <- c()
    
    for (idx in c(1:length(input_data))) {
        if (input_data[[idx]][1] == "DFM") {
            selected_stock_vec <- c(
                selected_stock_vec, 
                discrete_FM_all[[input_data[[idx]][2]]][[
                    paste0(input_data[[idx]][3], "_minC")
                ]]
            )
        }
        else if (input_data[[idx]][1] == "TFM") {
            selected_stock_vec <- c(
                selected_stock_vec, 
                threshold_FM_all[[input_data[[idx]][2]]][[
                    paste0(input_data[[idx]][3], "_minC")
                ]]
            )
        }
    }
    
    return(selected_stock_vec)
}

# (11)
CvalueTB[CvalueTB$SSI > median(CvalueTB$SSI), ] %>% row.names() %>% strsplit("_") %>% 
    selected_stock_vec_Gen(input_data = .) -> selected_stock_vec_1
stock_pool_Barchart(input_data = sort(table(selected_stock_vec_1), decreasing = TRUE), 
                    criteria_name = "Selected from SSI")

# (11)
CvalueTB[CvalueTB$CHI > median(CvalueTB$CHI), ] %>% row.names() %>% strsplit("_") %>% 
    selected_stock_vec_Gen(input_data = .) -> selected_stock_vec_2
stock_pool_Barchart(input_data = sort(table(selected_stock_vec_2), decreasing = TRUE), 
                    criteria_name = "Selected from CHI")

# (11)
CvalueTB[CvalueTB$DBI > median(CvalueTB$DBI), ] %>% row.names() %>% strsplit("_") %>% 
    selected_stock_vec_Gen(input_data = .) -> selected_stock_vec_3
stock_pool_Barchart(input_data = sort(table(selected_stock_vec_3), decreasing = TRUE), 
                    criteria_name = "Selected from DBI")

# (11)
CvalueTB[CvalueTB$PSI > median(CvalueTB$PSI), ] %>% row.names() %>% strsplit("_") %>% 
    selected_stock_vec_Gen(input_data = .) -> selected_stock_vec_4
stock_pool_Barchart(input_data = sort(table(selected_stock_vec_4), decreasing = TRUE), 
                    criteria_name = "Selected from PSI")

selected_stock_vec <- c(
    selected_stock_vec_1, selected_stock_vec_2, 
    selected_stock_vec_3, selected_stock_vec_4
)
stock_pool_Barchart(input_data = sort(table(selected_stock_vec), decreasing = TRUE), 
                    criteria_name = "Selected from Four Criteria Values")
