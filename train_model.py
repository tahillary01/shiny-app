# IMPORT PACKAGES
import pandas as pd
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
import requests
import logging
import shap
import concurrent.futures
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from scipy.spatial import cKDTree
from geopy.distance import distance
from geopy.distance import geodesic

cleaned_data = pd.read_csv('cleaned_data.csv')

# Define features and target
features = [
    'schools/0/distance', 'schools/1/rating', 'bathrooms',
    'resoFacts/bedrooms', 'yearBuilt', 'latitude',
    'longitude', 'lotSizeInSqft', 'livingArea', 'resoFacts/hasCooling',
    'resoFacts/hasHeating', 'resoFacts/parkingCapacity',
    'sale_month', 'sale_week', 'mortgageRates/thirtyYearFixedRate',
    'high_value_flooring', 'monthlyHoaFee_imputed', 'distance_to_city_center',
    'resoFacts/hasGarage_True',
    'resoFacts/hasSpa_1.0', 'resoFacts/hasHomeWarranty_True',
    'resoFacts/hasView_True',
    'resoFacts/homeType_Condo', 'resoFacts/homeType_Cooperative',
    'resoFacts/homeType_MobileManufactured', 'resoFacts/homeType_MultiFamily',
    'resoFacts/homeType_SingleFamily', 'resoFacts/homeType_Townhouse',
    'resoFacts/homeType_Unknown',
    'resoFacts/hasAttachedProperty_True', 'nearest_hospital_distance_km',
    'nearest_airport_distance_km', 'distance_to_beach_km', 'gdp_2019'
]

X = cleaned_data[features]
y = cleaned_data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor()

param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3]
}

random_search = RandomizedSearchCV(estimator=model,
                                   param_distributions=param_grid,
                                   n_iter=100,  # Number of different combinations to try
                                   scoring='neg_mean_squared_error',
                                   cv=5,
                                   verbose=1,
                                   n_jobs=-1,
                                   random_state=42)

# For Randomized Search
random_search.fit(X_train, y_train)

best_params = random_search.best_params_
best_score = random_search.best_score_

# Perform cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

# Convert scores to positive values
cv_scores = np.abs(cv_scores)

print(f"Cross-Validation MSE Scores: {cv_scores}")
print(f"Mean MSE: {cv_scores.mean()}")
print(f"Standard Deviation of MSE: {cv_scores.std()}")

# DELETE FOLLOWING -----

# Get the best model from RandomizedSearchCV
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

# Predict on training data to calculate train R^2
y_train_pred = best_model.predict(X_train)
train_r2 = r2_score(y_train, y_train_pred)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Best Parameters: {best_params}")
print(f"Best Score (MSE): {best_score}")
print(f"Test MSE: {mse}")
print(f"Training R²: {train_r2}")
print(f"Test R²: {r2}")

import joblib

# Save best model
joblib.dump(best_model, 'best_xgboost_model.pkl')
# Save best parameters and scores as a dictionary
results = {
    "best_params": best_params,
    "best_score": best_score,
    "train_r2": train_r2,
    "test_mse": mse,
    "test_r2": r2,
    "cv_scores": cv_scores
}
joblib.dump(results, 'model_results.pkl')

# Load model and results
best_model = joblib.load('best_xgboost_model.pkl')
results = joblib.load('model_results.pkl')