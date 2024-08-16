getwd()
setwd("C:/Users/admin/Desktop/p2_Natibundit_Suntraranusorn/data")
data <- read.csv("CleanedInR_LaptopPrice.csv")

categorical_columns <- c( "model","ram_gb","ssd","hdd","graphic_card_gb")
# Convert the specified columns to numeric using lapply
data[, categorical_columns] <- lapply(data[, categorical_columns], function(x) as.numeric(factor(x)))
summary(data)
categorical_columns <- c( "ram_gb","ssd",
                          "graphic_card_gb","latest_price","star_rating")
sdata<- data[,c(categorical_columns)]

# Change scaling to normal distribution which all data are Mean 0
pmatrix <- scale(sdata)
summary(pmatrix)

# K-Mean
k <- 3
pCluster <- kmeans(pmatrix, k, nstart = 100, iter.max = 100)
summary(pCluster)
pCluster$centers
pCluster$size

printCluster <- function(data, groups, columns){
  groupedD <- split(data, groups)
  lapply(groupedD,
         function(df) df[,columns])
}
cols_print <- c("brand","processor_name","ram_gb","ssd",
                "graphic_card_gb","latest_price")
groups <- pCluster$cluster
printCluster(data, groups, cols_print)

# Total within sum of square (WSS) for considering k number
# Creating customized function by calculating distance between two points
sqr_edist <- function(x,y){
  sum((x-y)^2)
}
# Calculate sum of square for each cluster
wss_cluster <- function(clust_matrix){
  c0 <- colMeans(clust_matrix)
  sum(apply(
    clust_matrix, 1, FUN = function(row){
      sqr_edist(row, c0)
    }
  ))
}
# Calculate sum of square for all clusters
wss_total <- function(dmatrix, labels){
  wsstot <- 0
  k <- length(unique(labels))
  for(i in 1:k)
    wsstot <- wsstot + wss_cluster(subset(dmatrix, labels == i))
  wsstot
}
# WSS use to consider number of k when we add more k the WSS will be reduce (k = n, WSS = 0) 
wss_total(pmatrix, groups)

# Group up Wss in different k
get_wss <- function(dmatrix, max_cluster){
  wss = numeric(max_cluster)
  wss[1] <- wss_cluster(dmatrix)
  for(k in 2:max_cluster){
    pcluster1 <- kmeans(pmatrix, k, nstart = 100, iter.max = 100)
    groups <- pcluster1$cluster
    wss[k] <- wss_total(dmatrix, groups)
  }
  wss
}

kmax <- 10
cluster_meas <- data.frame(ncluster = 1:kmax,
                           wss = get_wss(pmatrix, kmax))

# plot Elbow method
library(ggplot2)
breaks <- 1:kmax
ggplot(cluster_meas, aes(x=ncluster, y=wss)) +
  geom_point() + geom_line() +
  scale_x_continuous(breaks = breaks)

# "ch" 
library(fpc)
clustering_ch <- kmeansruns(pmatrix, krange = 1:10, criterion = "ch")
# Display clustering index by choosing the maximum value is the best k (best k = 3)
clustering_ch$crit
clustering_ch$bestk

# "asw"
clustering_asw <- kmeansruns(pmatrix, krange = 1:10, criterion = "asw")
# best k = 10
clustering_asw$crit
clustering_asw$bestk


