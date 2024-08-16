library("dplyr")
library("caret")
library("wrapr")
library(class)
library(rpart)
library("randomForest")
library("e1071")
library("vtreat")

setwd("C:/Users/admin/Desktop/p2_Natibundit_Suntraranusorn/data")
ldata <- read.csv("Laptop_data.csv")

summary(ldata)

# PREPROCESSING DATA

# Missing values in the data using "Missing" therefore change to NA.
ldata[ldata == "Missing"] <- NA

# Combine
ldata <- ldata %>% mutate(brand = ifelse(brand=='lenovo', 'Lenovo',brand))

# Missing values
count_missing = function(df) {
  sapply(df, FUN=function(col) sum(is.na(col)))
}
nacounts <- count_missing(ldata)
hasNA <- which(nacounts > 0)
nacounts[hasNA]

# Handle missing values
t_plan <- design_missingness_treatment(ldata)
data_prepared <- prepare(t_plan, ldata)

nacounts <- count_missing(data_prepared)
hasNA <- which(nacounts > 0)
nacounts[hasNA]

# Check the data types
str(data_prepared)

# Convert string columns (that can be converted to num) to numeric columns
data_prepared$ssd <- as.numeric(gsub(" GB", "", data_prepared$ssd))
data_prepared$hdd <- as.numeric(gsub(" GB", "", data_prepared$hdd))
data_prepared$processor_gnrtn <- as.numeric(gsub("th", "", data_prepared$processor_gnrtn))
data_prepared$display_size <- as.numeric(data_prepared$display_size)
data_prepared$ram_gb <- as.numeric(gsub(" GB GB", "", data_prepared$ram_gb))

# Replace the price currency from Indian rupee to CAD
data_prepared$latest_price <- data_prepared$latest_price/62
# summary(data_prepared)

# write.csv(data_prepared, "cleaned_v1.csv")

# Missing values
count_missing = function(df) {
  sapply(df, FUN=function(col) sum(is.na(col)))
}
nacounts <- count_missing(data_prepared)
hasNA <- which(nacounts > 0)
nacounts[hasNA]

# Handle missing values
t_plan <- design_missingness_treatment(data_prepared)
data_prepared <- prepare(t_plan, data_prepared)
nacounts <- count_missing(data_prepared)
hasNA <- which(nacounts > 0)
nacounts[hasNA]

# Select only brands greater than 3 models
brand_counts <- table(data_prepared$brand)
selected_brands <- names(brand_counts[brand_counts >= 3])
# filtered_data <- data_prepared[data_prepared$brand %in% selected_brands, ]
data_prepared <- data_prepared[data_prepared$brand %in% selected_brands, ]
brand_counts <- table(data_prepared$brand)
brand_counts

# Check outliers using boxplot
boxplot(data_prepared$latest_price)
q1 <- quantile(data_prepared$latest_price, 0.25)
q3 <- quantile(data_prepared$latest_price, 0.75)
iqr <- IQR(data_prepared$latest_price)
# Remove outliers
new_data <-
  subset(data_prepared,
         data_prepared$latest_price > (q1 - 1.5*iqr) &
           data_prepared$latest_price < (q3 +1.5*iqr)
  )
boxplot(new_data$latest_price)
summary(new_data$latest_price)

# Check coefficient
num_col <- unlist(lapply(new_data,
                         is.numeric),
                  use.names = FALSE)
cor(new_data[num_col])

corr_matrix <- round(cor(new_data[num_col]), 2)
print(corr_matrix)

# create heatmap
library(reshape2)
melted_cormat <- melt(corr_matrix)

ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) +
  geom_tile() +
  ggtitle("Correlation Heatmap")

# write.csv(new_data, "CleanedInR_LaptopPrice.csv")

# Count the number of models for each brand
brand_counts <- table(new_data$brand)

# Print the counts
print(brand_counts)


# Create categorical variables
# 3 labels
new_data$latest_price_cat <- ifelse(new_data$latest_price > 1722, "high",
                                        ifelse(new_data$latest_price > 1131.6, "mid", "low")
                                        )
new_data$latest_price_cat <- as.factor(new_data$latest_price_cat)

# Prepare variables
labels = c("high","mid", "low")
y = c("latest_price_cat")
x = c("brand","graphic_card_gb", "ssd","ram_gb", "star_rating")
fmla <- mk_formula(y,x)
print(fmla)
cleaned_data = new_data[, c(y,x)]
summary(cleaned_data)

# Process to normalize the data
process <- preProcess(cleaned_data, method=c("range"))
norm_cleaned_data <- predict(process, cleaned_data)
class(norm_cleaned_data$latest_price_cat)
################################################################################

# MODELING
# set seed and split the dataset into 70%
set.seed(3360)
training_obs <- cleaned_data$latest_price_cat %>% createDataPartition(
  p=0.6, list=F)

# not normalized.
train <- cleaned_data[training_obs,]
test <- cleaned_data[-training_obs,]

# normalized data to be range from 0-1
norm_train <- norm_cleaned_data[training_obs,]
norm_test <- norm_cleaned_data[-training_obs,]

# create a logistic model
logit_model <- glm(fmla, data=train, family=binomial(link = "logit"))

train$pred_prob <- predict(logit_model, newdata = train, type ="response")
test$pred_prob <- predict(logit_model, newdata = test, type ="response")

# 3 labels
train$pred <- factor(ifelse(train$pred_prob > 0.6 , labels[3],
                                 ifelse(train$pred_prob > 0.35, labels[2], labels[1]))
                                 )
test$pred <- factor(ifelse(test$pred_prob > 0.6, labels[3],
                                 ifelse(test$pred_prob > 0.35, labels[2], labels[1]))
                          )
# create confusion matrix
cfm_train <- confusionMatrix(data = as.factor(train$pred),reference = as.factor(train$latest_price_cat))
cfm_test <- confusionMatrix(data = as.factor(test$pred),reference = as.factor(test$latest_price_cat))

# Get the True Positive (TP) values for each class in the training set
tp_train <- cfm_train$byClass[, "Sensitivity"]
tp_test <- cfm_test$byClass[, "Sensitivity"]
print(tp_train)
print(tp_test)

# Create a data frame to store the accuracy values
result_accuracy <- data.frame(Label = c("high", "mid", "low"),
                              Accuracy_Train = tp_train,
                              Accuracy_Test = tp_test)
print(result_accuracy)

# # Improve accuracy by trainControl with repeated cross-validation method
# train_control <- trainControl(method = "repeatedcv",
#                               number = 10,)
# model <- train(fmla, data = train, method = "multinom", family= binomial,
#                trControl = train_control, metric = "Accuracy")
# print(model)
# model$resample
# mean(model$resample$Accuracy)
# getTrainPerf(model)

# Visualize the confusion matrix
library(ggplot2)

# Convert the confusion matrices to data frames
cfm_train_df <- as.data.frame(as.table(cfm_train$table))
cfm_test_df <- as.data.frame(as.table(cfm_test$table))

# Function to create a heatmap for the confusion matrix
create_confusion_matrix_heatmap <- function(confusion_matrix_df) {
  ggplot(confusion_matrix_df, aes(x = Reference, y = Prediction)) +
    geom_tile(aes(fill = Freq), color = "white") +
    geom_text(aes(label = Freq), vjust = 1) +
    scale_fill_gradient(low = "white", high = "blue") +
    labs(x = "Reference", y = "Prediction", fill = "Frequency") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# Create heatmap for confusion matrix
train_confusion_heatmap <- create_confusion_matrix_heatmap(cfm_train_df)
print(train_confusion_heatmap)
test_confusion_heatmap <- create_confusion_matrix_heatmap(cfm_test_df)
print(test_confusion_heatmap)

# get only the accuracy
method <- c("logit")
accuracy_train <- c(as.numeric(cfm_train$overall["Accuracy"]))
accuracy_test <- c(as.numeric(cfm_test$overall["Accuracy"]))
result_modeling <- data.frame(method, accuracy_train, accuracy_test)
result_modeling

# Random Forest with type class
library("randomForest")
rf_model <- randomForest(fmla, data=train)
train$pred <- predict(rf_model, newdata = train, type="class") 
test$pred <- predict(rf_model, newdata = test, type="class")
cfm_train <- confusionMatrix(data = as.factor(train$pred),reference = as.factor(train$latest_price_cat))
cfm_test <- confusionMatrix(data = as.factor(test$pred),reference = as.factor(test$latest_price_cat))
method <- append(method, "Random Forest with type class")
accuracy_train <- append(accuracy_train, as.numeric(cfm_train$overall["Accuracy"]))
accuracy_test <- append(accuracy_test, as.numeric(cfm_test$overall["Accuracy"]))
result_modeling <- data.frame(method, accuracy_train, accuracy_test)

# Random Forest with type response
library("randomForest")
rf_model <- randomForest(fmla, data=train)
train$pred <- predict(rf_model, newdata = train, type="response") 
test$pred <- predict(rf_model, newdata = test, type="response")
cfm_train <- confusionMatrix(data = as.factor(train$pred),reference = as.factor(train$latest_price_cat))
cfm_test <- confusionMatrix(data = as.factor(test$pred),reference = as.factor(test$latest_price_cat))
method <- append(method, "Random Forest with type response")
accuracy_train <- append(accuracy_train, as.numeric(cfm_train$overall["Accuracy"]))
accuracy_test <- append(accuracy_test, as.numeric(cfm_test$overall["Accuracy"]))
result_modeling <- data.frame(method, accuracy_train, accuracy_test)

# Tuning Random Forest (ntree, mtry, nodesize)
# help("randomForest")
rf_model <- randomForest(fmla, data=norm_train, ntree=350, mtry=3, nodesize=15)
train$pred <- predict(rf_model, newdata = norm_train, type="response")
test$pred <- predict(rf_model, newdata = norm_test, type="response")
cfm_train <- confusionMatrix(data = as.factor(train$pred),reference = as.factor(train$latest_price_cat))
cfm_test <- confusionMatrix(data = as.factor(test$pred),reference = as.factor(test$latest_price_cat))
method <- append(method, "Tuning Random Forest (ntree, mtry, nodesize)")
accuracy_train <- append(accuracy_train, as.numeric(cfm_train$overall["Accuracy"]))
accuracy_test <- append(accuracy_test, as.numeric(cfm_test$overall["Accuracy"]))
result_modeling <- data.frame(method, accuracy_train, accuracy_test)

result_modeling

# Export the result to a csv file
# write.csv(result_modeling, file = "result_modeling.csv", row.names = FALSE)