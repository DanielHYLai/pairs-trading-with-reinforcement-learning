# Import necessary packages
{
    library(dplyr)
    library(rvest)
}

# Crawler the Wikipedia page for a list of stocks
stocklist_Crawler <- function(url) {
    
    ## Crawler the page content
    webpage <- rvest::read_html(url)
    
    ## Search for content with the tag "table" and filter out specific columns
    component_list <- webpage %>% 
        rvest::html_node("table") %>% 
        rvest::html_table(fill = TRUE) %>% 
        dplyr::select(Symbol, Security, `GICS Sector`, Founded)
    
    return(component_list)
}

url <- "https://en.wikipedia.org/w/index.php?title=List_of_S%26P_500_companies&oldid=1242054734"
component_list <- stocklist_Crawler(url = url)

# Adjust the stock tickers on Yahoo Finance and Wikipedia that don't match
component_list[
    component_list$Symbol %in% c("BRK.B", "BF.B"), "Symbol"
] <- c("BRK-B", "BF-B")

# Export the csv file
write.csv(component_list, "SP500_list.csv", row.names = FALSE)

# Clear up the environment variables
rm(list = ls())
