{
    library(ggplot2)
    library(dplyr)
    library(tidyr)
    library(gridExtra)
    library(pheatmap)
    
    source("utils/utils_clustering_method.R")
    source("utils/utils_frequency_matrix.R")
}

{
    data_stock <- get(load("./cache_file/data_stock_cache.rData"))
    data_train <- readr::read_csv("./data_file/train.csv", show_col_types = FALSE)
    data_test  <- read.csv("./data_file/test.csv")
    load("cache_file/all_possible_coTest_cache.rData")
}

#-----
# 展示不同 stock pairs 的 p-value 分佈的差異
data <- data.frame(
    name = c(rep("O-WELL", 84), c(rep("MMM-ACN", 84))), 
    value = c(all_test_result[["O-WELL"]], all_test_result[["MMM-ACN"]])
)

windowsFonts("Times New Roman" = windowsFont("Times New Roman"))

data %>% 
    ggplot(aes(x = name, y = value, fill = name)) + 
    geom_boxplot() + 
    scale_fill_manual(values = c("O-WELL" = "#FF9999", "MMM-ACN" = "#66B2FF")) + 
    theme_minimal() + 
    theme(
        legend.position = "none", 
        axis.title.x = element_text(size = 24, family = "Times New Roman"), 
        axis.title.y = element_text(size = 24, family = "Times New Roman"), 
        axis.text.x = element_text(size=16, family = "Times New Roman", color = "black"), 
        axis.text.y = element_text(size=16, family = "Times New Roman", color = "black")
    ) + 
    xlab("Asset Pair") + 
    ylab(expression(italic(p)*"-value")) -> p; p

# ggsave("pair_pvalue_example.png", plot = p, width = 10, height = 8, dpi = 300)

library(ggplot2)
library(dplyr)

# 先將兩支股票整理成寬格式
wide_data <- data_train %>%
    filter(Ticker %in% c("O", "WELL")) %>%
    dplyr::select(Date, Ticker, Close) %>%
    tidyr::pivot_wider(names_from = Ticker, values_from = Close) %>%
    mutate(Diff = O - WELL)

# 計算差值的 rescale 用於 sec.axis
# 我們將 Diff 轉換為主軸（Close）的範圍內，才可疊線
scale_factor <- max(wide_data$O, wide_data$WELL, na.rm = TRUE) / max(abs(wide_data$Diff), na.rm = TRUE)

ggplot(wide_data, aes(x = Date)) +
    geom_line(aes(y = O, color = "O"), size = 3) +
    geom_line(aes(y = WELL, color = "WELL"), size = 3) +
    geom_line(aes(y = Diff * scale_factor, color = "O - WELL"), size = 3) +
    scale_y_continuous(
        name = "Close Price",
        sec.axis = sec_axis(~ . / scale_factor, name = "Price Difference (O - WELL)")
    ) +
    scale_color_manual(
        name = "",
        values = c("O" = "blue", "WELL" = "green", "O - WELL" = "pink")
    ) +
    labs(
        title = "Closing Prices of O and WELL with Difference",
        x = "Date"
    ) +
    theme_minimal() +
    theme(
        text = element_text(family = "Times New Roman"),
        axis.text = element_text(size = 20),
        axis.title = element_text(size = 24),
        legend.text = element_text(size = 20), 
        plot.title = element_text(size = 28, hjust = 0.5),
        legend.position = "bottom"
    ) -> pp; pp
ggsave("pair_example.png", plot = pp, width = 25, height = 6, dpi = 300)

#-----
# 展示固定相同 q 之下，觀察 K 的效果
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

## empirical study 放的圖片
criteria_values <- clustering_method_Optimal(
    df = threshold_FM_all[[2]], 
    K = 6, 
    assigned_method = "HC", 
    seed = 123, 
    visualized = FALSE
)
criteria_values[["Num"]] <- as.numeric(substr(row.names(criteria_values), 7, 7))
criteria_values <- criteria_values[c("Num", "SSI", "CHI", "PSI", "DBI")]
criteria_values <- pivot_longer(
    criteria_values, 
    cols = -Num, 
    names_to = "Metric", 
    values_to = "Value"
)

highlight <- criteria_values %>% 
    group_by(Metric) %>% 
    filter(
        (Metric %in% c("SSI", "CHI", "PSI") & Value == max(Value)) | 
            (Metric == "DBI" & Value == min(Value))
    )

titles <- c(SSI = "Silhouette Score",
            CHI = "Calinski-Harabasz Index",
            PSI = expression(Psi ~ "Index"), 
            DBI = "Davies-Bouldin Index")

windowsFonts("Times New Roman" = windowsFont("Times New Roman"))

p <- lapply(unique(criteria_values$Metric), function(metric) {
    ggplot(subset(criteria_values, Metric == metric), aes(x = Num, y = Value)) + 
        geom_line(size = 1.5) + 
        geom_point(size = 2.5) + 
        geom_point(data = subset(highlight, Metric == metric), color = "red", size = 4) + 
        labs(x = "Number of Cluster (K)", y = "Value", title = titles[[metric]]) + 
        theme_minimal() + 
        theme(
            plot.title = element_text(
                hjust = 0.5, size = 24, family = "Times New Roman", color = "black"
            ), 
            axis.title.x = element_text(size = 20, family = "Times New Roman"), 
            axis.title.y = element_text(size = 20, family = "Times New Roman"), 
            axis.text.x = element_text(size=18, family = "Times New Roman", color = "black"), 
            axis.text.y = element_text(size=18, family = "Times New Roman", color = "black")
        )
})

grid.arrange(grobs = p, ncol = 2) -> p; p

ggsave("cluster_num_example.png", plot = p, width = 16, height = 12, dpi = 300)

for (idx in seq_along(threshold_FM_all)) {
    criteria_values <- clustering_method_Optimal(
        df = threshold_FM_all[[idx]], 
        K = 6, 
        assigned_method = "HC", 
        seed = 123, 
        visualized = FALSE
    )
    criteria_values[["Num"]] <- as.numeric(substr(row.names(criteria_values), 7, 7))
    criteria_values <- criteria_values[c("Num", "SSI", "CHI", "PSI", "DBI")]
    criteria_values <- pivot_longer(
        criteria_values, 
        cols = -Num, 
        names_to = "Metric", 
        values_to = "Value"
    )
    highlight <- criteria_values %>% 
        group_by(Metric) %>% 
        filter(
            (Metric %in% c("SSI", "CHI", "PSI") & Value == max(Value)) | 
                (Metric == "DBI" & Value == min(Value))
        )
    
    titles <- c(SSI = "Silhouette Score",
                CHI = "Calinski-Harabasz Index",
                PSI = expression(Psi ~ "Index"), 
                DBI = "Davies-Bouldin Index")
    p <- lapply(unique(criteria_values$Metric), function(metric) {
        ggplot(subset(criteria_values, Metric == metric), aes(x = Num, y = Value)) + 
            geom_line(size = 1.5) + 
            geom_point(size = 2.5) + 
            geom_point(data = subset(highlight, Metric == metric), color = "red", size = 4) + 
            labs(x = "Number of Cluster (K)", y = "Value", title = titles[[metric]]) + 
            theme_minimal() + 
            theme(
                plot.title = element_text(
                    hjust = 0.5, size = 24, family = "Times New Roman", color = "black"
                ), 
                axis.title.x = element_text(size = 20, family = "Times New Roman"), 
                axis.title.y = element_text(size = 20, family = "Times New Roman"), 
                axis.text.x = element_text(size=18, family = "Times New Roman", color = "black"), 
                axis.text.y = element_text(size=18, family = "Times New Roman", color = "black")
            )
    })
    
    grid.arrange(grobs = p, ncol = 2) -> p; p
    
    ggsave(paste0("q_", idx, "_decide_K.png"), plot = p, width = 16, height = 12, dpi = 300)
}
#-----
# 展示設定不同的 q 對 frequency matrix 的影響
# q = 0.05
# 
# matrix(
#     0, nrow = length(data_stock), ncol = length(data_stock), 
#     dimnames = list(names(data_stock), names(data_stock))) %>% 
#     FM_generator(result_test = all_test_result, 
#                  result_mat = ., 
#                  discrete = FALSE, 
#                  threshold_q = q) -> mat
# 
# mat[!row.names(mat) %in% c("BRK-B", "BF-B"), 
#     !colnames(mat) %in% c("BRK-B", "BF-B")] -> mat
# 
# windowsFonts("Times New Roman" = windowsFont("Times New Roman"))
# 
# png("heatmap_005.png", width = 1000, height = 1000, res = 150, family = "Times New Roman")
# 
# pheatmap::pheatmap(
#     mat = mat,
#     color = colorRampPalette(c("blue", "white", "red"))(100),
#     breaks = seq(0, 0.05, length.out = 101),  # extreme values
#     show_rownames = FALSE,
#     show_colnames = FALSE,
#     treeheight_row = 0,
#     treeheight_col = 0, 
#     legend = FALSE,
#     # main = paste0("Percentile q = ", q)
# )
# 
# dev.off()

#-----
input_data = sort(table(selected_stock_vec), decreasing = TRUE)
windowsFonts("Times New Roman" = windowsFont("Times New Roman"))
# par(family = "Times New Roman")
png("pair_pool.png", width = 2500, height = 1600, res = 300, family = "Times New Roman")
fig <- barplot(
    input_data,
    border = TRUE, 
    col = adjustcolor("#46A3FF", alpha.f = 1), 
    ylim = c(0, max(input_data) + 3), 
    xlab = "Stock", 
    ylab = "Count", 
    cex.lab = 1.5,
    cex.axis = 1.2,
    cex.names = 1.2
)
abline(v = fig[61] + 0.3, col = "red", lty = 2, lwd = 2)
counts <- rev(table(input_data))
loc <- c()
start_idx <- 1
for (end_idx in cumsum(counts)) {
    loc <- c(loc, median(fig[start_idx:end_idx]))
    start_idx <- end_idx + 1
}
text(loc, as.integer(names(counts)) + 0.5, labels = counts, family = "Times New Roman")
dev.off()