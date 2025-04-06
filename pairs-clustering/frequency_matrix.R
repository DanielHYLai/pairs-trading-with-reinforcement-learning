# Import necessary packages and customized functions
{
    library(dplyr)
    library(readr)
    library(pheatmap)
    
    source("./utils/utils_frequency_matrix.R")
    source("./utils/utils_clustering_method.R")
    
    set.seed(123)
}

# Load cache file
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

###################################################################################################
# Discretize: alpha = 0.01

discrete01_test_result <- test_result_Discretize(obj = all_test_result, alpha = 0.01)

discrete01_FM <- matrix(0, nrow = length(data_stock), ncol = length(data_stock), 
                        dimnames = list(names(data_stock), names(data_stock))) %>% 
    FM_generator(result_test = discrete01_test_result, result_mat = ., discrete = TRUE) %>% 
    .[!row.names(.) %in% c("BRK-B", "BF-B"), 
      !colnames(.) %in% c("BRK-B", "BF-B")]

pheatmap::pheatmap(
    mat = discrete01_FM, 
    color = colorRampPalette(c("white", "orange", "red"))(100), 
    show_rownames = FALSE, 
    show_colnames = FALSE, 
    # cutree_rows = 3, 
    # cutree_cols = 3, 
    # treeheight_row = 0, 
    # treeheight_col = 0, 
    main = expression(alpha == 0.01)
    )

# clustering_method_Optimal(
#     df = discrete01_FM, K = 4, 
#     assigned_method = "HC", seed = 123, 
#     visualized = FALSE
#     )

DFM01_c2 <- fundamental_Cluster(df = discrete01_FM, K = 2, assigned_method = "HC", seed = 123)
DFM01_c3 <- fundamental_Cluster(df = discrete01_FM, K = 3, assigned_method = "HC", seed = 123)
table(DFM01_c2); table(DFM01_c3)

###################################################################################################

###################################################################################################
# Discretize: alpha = 0.05

discrete05_test_result <- test_result_Discretize(obj = all_test_result, alpha = 0.05)

discrete05_FM <- matrix(0, nrow = length(data_stock), ncol = length(data_stock), 
                        dimnames = list(names(data_stock), names(data_stock))) %>% 
    FM_generator(result_test = discrete05_test_result, result_mat = ., discrete = TRUE) %>% 
    .[!row.names(.) %in% c("BRK-B", "BF-B"), 
      !colnames(.) %in% c("BRK-B", "BF-B")]

pheatmap::pheatmap(
    mat = discrete05_FM, 
    color = colorRampPalette(c("white", "orange", "red"))(100), 
    show_rownames = FALSE, 
    show_colnames = FALSE, 
    # cutree_rows = 3, 
    # cutree_cols = 3, 
    treeheight_row = 0, 
    treeheight_col = 0, 
    main = expression(alpha == 0.05)
)

# clustering_method_Optimal(
#     df = discrete05_FM, K = 4, 
#     assigned_method = "HC", seed = 123, 
#     visualized = FALSE
# )

DFM05_c2 <- fundamental_Cluster(df = discrete05_FM, K = 2, assigned_method = "HC", seed = 123)
DFM05_c3 <- fundamental_Cluster(df = discrete05_FM, K = 3, assigned_method = "HC", seed = 123)
table(DFM05_c2); table(DFM05_c3)

###################################################################################################

###################################################################################################
# threshold (p-value): q = 0.1 ~ 0.2

threshold_FM_all <- list()

for (q in seq(0.1, 0.2, length.out = 10)) {
    cat("loc =", floor(length(all_test_result[[1]]) * q), "\n")
    threshold_FM_all[[as.character(floor(length(all_test_result[[1]]) * q))]] <- matrix(
        0, nrow = length(data_stock), ncol = length(data_stock), 
        dimnames = list(names(data_stock), names(data_stock))) %>% 
        FM_generator(result_test = all_test_result, 
                     result_mat = ., 
                     discrete = FALSE, 
                     threshold_q = q)
}

# for (idx in seq_along(threshold_FM_all)) {
#     cat("idx:", idx, "\n")
#     pheatmap::pheatmap(
#         mat = threshold_FM_all[[idx]], 
#         color = colorRampPalette(c("red", "orange", "white"))(100), 
#         show_rownames = FALSE, 
#         show_colnames = FALSE, 
#         # cutree_rows = 3, 
#         # cutree_cols = 3, 
#         treeheight_row = 0, 
#         treeheight_col = 0
#     )
#     clustering_method_Optimal(
#         df = threshold_FM_all[[idx]], K = 3, 
#         assigned_method = "HC", seed = 123, 
#         visualized = FALSE
#     ) %>% write.csv(., file = paste0("temp-", idx, ".csv"))
# }

###################################################################################################
