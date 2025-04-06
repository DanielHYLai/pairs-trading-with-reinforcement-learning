# 載入套件
{
    library(cluster)
    library(clusterSim)
    library(dplyr)
    library(fpc)
    library(SNFtool)
}

# 計算 PSI 指標
PSI_index <- function(mat_normalized, cluster) {
    
    group <- cluster
    ordered_indices  <- order(group)
    reordered_matrix <- mat_normalized[ordered_indices, ordered_indices]
    
    ## 計算距離 (對角線)
    C      <- list()
    mean_1 <- c()
    l      <- c()
    
    for (idx in 1:length(unique(group))) {
        C[[idx]] <- as.matrix(
            log(
                reordered_matrix[
                    (length(which(group <= idx - 1)) + 1):length(which(group <= idx)), 
                    (length(which(group <= idx - 1)) + 1):length(which(group <= idx))
                    ]
                )
            )
        diag(C[[idx]]) <- 0
        mean_1[idx]    <- sum(C[[idx]])
        l[idx] <- length(C[[idx]])
    }
    
    ## 計算距離 (非對角線)
    compute <- reordered_matrix
    logcompute <- log(compute)
    
    for (idx in 1:length(unique(group))) {
        logcompute[
            (length(which(group <= idx - 1)) + 1):length(which(group <= idx)), 
            (length(which(group <= idx - 1)) + 1):length(which(group <= idx))
            ] <- 0
    }
    
    logical_index <- row(logcompute) < col(logcompute)
    result <- logcompute[logical_index]
    sum_2  <- sum(result)
    
    logical_index <- row(logcompute) > col(logcompute)
    result <- logcompute[logical_index]
    sum_3  <- sum(result)
    
    d <- (sum(mean_1) / (sum(l) - ncol(mat_normalized))) / 
         ((sum_2 + sum_3) / (length(logcompute) - sum(l)))
    
    return(1 / d)
}


# 綜合計算分群方法的評估指標
cluster_criteria_Calculator <- function(df, result) {
    
    SS  <- round(cluster::silhouette(result, dist(df))[, 3] %>% mean(), 2)
    CHI <- round(fpc::calinhara(x = df, clustering = result), 2)
    DBI <- round(clusterSim::index.DB(x = df, cl = result)$DB, 2)
    PSI <- round(PSI_index(
        mat_normalized = SNFtool::affinityMatrix(df, K = length(unique(result))), 
        cluster = result
        ), 2)
    
    return(list(SSI = SS, CHI = CHI, DBI = DBI, PSI = PSI))
}

# 視覺化呈現最佳分群群數
optimal_clusterNum_Visualize <- function(criteria_list, assigned_method) {
    par(mfrow = c(4, 1))
    
    ## 繪製 SS 指標
    plot(criteria_list$SSI, type = "b", 
         main = paste0("Silhouette Coefficient (", assigned_method, ")"), 
         xaxt = "n", xlab = "Number of Cluster", ylab = "Value")
    axis(1, 
         at = 1:length(criteria_list$SSI), 
         labels = seq_along(criteria_list$SSI) + 1
         )
    points(
        which.max(criteria_list$SSI), 
        criteria_list$SSI[which.max(criteria_list$SSI)], 
        col = "red", pch = 19, cex = 1.5
    )
    
    ## 繪製 CHI 指標
    plot(criteria_list$CHI, type = "b", 
         main = paste0("Calinski-Harabasz Index (", assigned_method, ")"), 
         xaxt = "n", xlab = "Number of Cluster", ylab = "Value")
    axis(1, 
         at = 1:length(criteria_list$CHI), 
         labels = seq_along(criteria_list$CHI) + 1
         )
    points(
        which.max(criteria_list$CHI), 
        criteria_list$CHI[which.max(criteria_list$CHI)], 
        col = "red", pch = 19, cex = 1.5
    )
    
    ## 繪製 DBI 指標
    plot(criteria_list$DBI, type = "b", 
         main = paste0("Davies-Bouldin Index (", assigned_method, ")"), 
         xaxt = "n", xlab = "Number of Cluster", ylab = "Value")
    axis(1, 
         at = 1:length(criteria_list$DBI), 
         labels = seq_along(criteria_list$DBI) + 1
         )
    points(
        which.min(criteria_list$DBI), 
        criteria_list$DBI[which.min(criteria_list$DBI)], 
        col = "red", pch = 19, cex = 1.5
    )
    
    ## 繪製 PSI 指標
    plot(criteria_list$PSI, type = "b", 
         main = bquote(Psi ~ "(" ~ .(assigned_method) ~ ")"), 
         xaxt = "n", xlab = "Number of Cluster", ylab = "Value")
    axis(1, 
         at = 1:length(criteria_list$PSI), 
         labels = seq_along(criteria_list$PSI) + 1
         )
    points(
        which.max(criteria_list$PSI), 
        criteria_list$PSI[which.max(criteria_list$PSI)], 
        col = "red", pch = 19, cex = 1.5
    )
}


# 搜尋指定分群演算法的最佳群數
clustering_method_Optimal <- function(df, K, assigned_method, seed = NULL, visualized = FALSE) {
    
    ## 設定 random seed
    if (is.null(seed) == FALSE) {
        set.seed(seed = seed)
    }
    
    ## 儲存評估指標的矩陣
    criteria_result <- matrix(0, nrow = K - 1, ncol = 4)
    
    ## HC 的初始化參數
    dist_matrix <- stats::dist(df, method = "euclidean")
    temp_hc <- stats::hclust(d = dist_matrix, method = "complete")
    
    for (num in c(2:K)) {
        
        ## 使用 Kmeans
        if (assigned_method == "Kmeans") {
            temp_model <- stats::kmeans(x = df, centers = num)
            temp_criteria <- cluster_criteria_Calculator(
                df = df, result = temp_model$cluster
            )
        }
        
        ## 使用 HC
        else if (assigned_method == "HC") {
            temp_model <- stats::cutree(temp_hc, k = num)
            temp_criteria <- cluster_criteria_Calculator(
                df = df, result = temp_model
            )
        }
        
        ## 使用 spec
        else if (assigned_method == "spec") {
            temp_model <- SNFtool::spectralClustering(affinity = df, K = num)
            temp_criteria <- cluster_criteria_Calculator(
                df = df, result = temp_model
            )
        }
        
        else {
            cat("Please check the assigned method.\n")
            break
            return(NULL)
        }
        ## 將評估指標計算結果填入矩陣
        for (col_idx in c(1:length(temp_criteria))) {
            criteria_result[num - 1, col_idx] <- temp_criteria[[col_idx]]
        }
    }
    
    ## 將 matrix 轉為 data.frame 且重新命名 column names and row names
    criteria_result <- as.data.frame(criteria_result)
    row.names(criteria_result) <- paste("Num =", c(2:K))
    names(criteria_result) <- names(temp_criteria)
    
    ## visualization
    if (visualized == TRUE) {
        optimal_clusterNum_Visualize(
            criteria_list = criteria_result, 
            assigned_method = assigned_method
            )
    }
    
    return(criteria_result)
}

# 使用指定的分群演算法
fundamental_Cluster <- function(df, K, assigned_method, seed = NULL) {
    
    ## 設定 random seed
    if (is.null(seed) == FALSE) {
        set.seed(seed = seed)
    }
    
    ## 使用 kmeans
    if (tolower(assigned_method) == "kmeans") {
        
        cat("Assigned method: Kmeans ... ")
        result <- stats::kmeans(x = df, centers = K)$cluster
        cat("Done.\n")
        
        return(result)
    }
    
    ## 使用 HC
    else if (tolower(assigned_method) == "hc") {
        
        cat("Assigned method: Hierarchical Clustering ... ")
        result <- stats::hclust(
            d = stats::dist(x = df, method = "euclidean"), 
            method = "complete"
            ) %>% 
            stats::cutree(., k = K)
        cat("Done.\n")
        
        return(result)
    }
    
    ## 使用 spec
    else if (tolower(assigned_method) == "spec") {
        
        cat("Assigned method: Spectral Clustering ... ")
        result <- SNFtool::spectralClustering(
            affinity = df, K = K
        )
        cat("Done.\n")
        
        return(result)
    }
    
    else {
        
        cat("There does not provide the method called", assigned_method, ".\n")
        return(NULL)
    }
}
