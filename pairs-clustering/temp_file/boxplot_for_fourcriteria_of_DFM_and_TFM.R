library(ggplot2)
data.frame(
    value = reshape2::melt(CvalueTB)$value, 
    criteria = reshape2::melt(CvalueTB)$variable, 
    type = c(rep("DFM", 20), rep("TFM", 18), 
             rep("DFM", 20), rep("TFM", 18),
             rep("DFM", 20), rep("TFM", 18), 
             rep("DFM", 20), rep("TFM", 18))
) %>% 
    ggplot2::ggplot(., aes(x = criteria, y = value, fill = type)) + 
    geom_boxplot()

data.frame(
    value = reshape2::melt(CvalueTB[, -2])$value, 
    criteria = reshape2::melt(CvalueTB[, -2])$variable, 
    type = c(rep("DFM", 20), rep("TFM", 18), 
             # rep("DFM", 20), rep("TFM", 18), 
             rep("DFM", 20), rep("TFM", 18), 
             rep("DFM", 20), rep("TFM", 18))
) %>% 
    ggplot2::ggplot(., aes(x = criteria, y = value, fill = type)) + 
    geom_boxplot()