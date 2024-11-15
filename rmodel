# Load required libraries
library(dplyr)
library(ggplot2)
library(caret)
library(xgboost)

# Load the cleaned dataset
cleaned_data <- read.csv('/Users/annabelleshea/Downloads/cleaned_data.csv')

# Define features and target
features <- c(
  'gdp_2019', 'livingArea', 'longitude', 'latitude', 'resoFacts.homeType_SingleFamily',
  'resoFacts.hasGarage_True', 'lotSizeInSqft', 'nearest_hospital_distance_km',
  'resoFacts.bedrooms', 'bathrooms', 'yearBuilt', 'distance_to_beach_km',
  'distance_to_city_center', 'schools.0.distance'
)

X <- cleaned_data[, features]
y <- cleaned_data$price

# Split the data into training and testing sets
set.seed(42)
train_indices <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_indices, ]
X_test <- X[-train_indices, ]
y_train <- y[train_indices]
y_test <- y[-train_indices]

# Convert data to DMatrix format for XGBoost
train_matrix <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
test_matrix <- xgb.DMatrix(data = as.matrix(X_test), label = y_test)

# Define the model parameters
params <- list(
  objective = "reg:squarederror",
  subsample = 0.6,
  eta = 0.05,
  min_child_weight = 3,
  max_depth = 5,
  gamma = 0.1,
  colsample_bytree = 1.0
)

# Train the XGBoost model
set.seed(42)
model <- xgb.train(
  params = params,
  data = train_matrix,
  nrounds = 150,
  watchlist = list(train = train_matrix, test = test_matrix),
  print_every_n = 10
)

# Make predictions
y_train_pred <- predict(model, train_matrix)
y_test_pred <- predict(model, test_matrix)

# Calculate metrics manually
train_mse <- mean((y_train - y_train_pred)^2)
test_mse <- mean((y_test - y_test_pred)^2)
train_r2 <- 1 - (sum((y_train - y_train_pred)^2) / sum((y_train - mean(y_train))^2))
test_r2 <- 1 - (sum((y_test - y_test_pred)^2) / sum((y_test - mean(y_test))^2))

# Print metrics
cat(sprintf("Training MSE: %.2f\n", train_mse))
cat(sprintf("Test MSE: %.2f\n", test_mse))
cat(sprintf("Training R²: %.4f\n", train_r2))
cat(sprintf("Test R²: %.4f\n", test_r2))

# Save the model as an RDS file
saveRDS(model, file = "xgboost_model.rds")
