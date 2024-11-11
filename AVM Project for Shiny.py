import pandas as pd
import joblib

# Load the input data (replace 'your_input_data.csv' with your actual input data file)
new_data = pd.read_csv('lotwize_case_validation-1.csv')

# Define the expected features from the model
expected_features = [
    'schools/0/distance', 'schools/1/rating', 'bathrooms', 'resoFacts/bedrooms', 'yearBuilt',
    'latitude', 'longitude', 'lotSizeInSqft', 'livingArea', 'resoFacts/hasCooling',
    'resoFacts/hasHeating', 'resoFacts/parkingCapacity', 'sale_month', 'sale_week',
    'mortgageRates/thirtyYearFixedRate', 'high_value_flooring', 'monthlyHoaFee_imputed',
    'distance_to_city_center', 'resoFacts/hasGarage_True', 'resoFacts/hasSpa_1.0',
    'resoFacts/hasHomeWarranty_True', 'resoFacts/hasView_True', 'resoFacts/homeType_Condo',
    'resoFacts/homeType_Cooperative', 'resoFacts/homeType_MobileManufactured',
    'resoFacts/homeType_MultiFamily', 'resoFacts/homeType_SingleFamily',
    'resoFacts/homeType_Townhouse', 'resoFacts/homeType_Unknown',
    'resoFacts/hasAttachedProperty_True', 'nearest_hospital_distance_km',
    'nearest_airport_distance_km', 'distance_to_beach_km', 'gdp_2019'
]

# Step 1: Add Missing Features
for col in expected_features:
    if col not in new_data.columns:
        new_data[col] = 0  # Fill missing columns with default value (0)

# Step 2: Convert Data Types
# Ensure all features are numeric (int, float)
for col in expected_features:
    new_data[col] = pd.to_numeric(new_data[col], errors='coerce').fillna(0)

# Step 3: Filter the Data to Keep Only Expected Features
new_data = new_data[expected_features]

# Step 4: Load the Trained Model
best_model = joblib.load('best_xgboost_model.pkl')

# Step 5: Make Predictions
print("Making predictions...")
predicted_prices = best_model.predict(new_data)

# Step 6: Display Predicted Prices
print("Predicted Prices:", predicted_prices)

# Save the predicted prices to a CSV file
output = pd.DataFrame(predicted_prices, columns=['Predicted Price'])
output.to_csv('predicted_prices.csv', index=False)
print("Predicted prices saved to 'predicted_prices.csv'.")

'''# Load necessary libraries
library(shiny)
library(reticulate)

# Set up reticulate to use the Python virtual environment
use_virtualenv("myenv", required = TRUE)

# Load Python model and libraries
py_run_string("import joblib")
py_run_string("import pandas as pd")

# Load the trained model
model <- py$joblib$load("/srv/shiny-server/your-app/best_xgboost_model.pkl")

# Define UI for the app
ui <- fluidPage(
    titlePanel("House Price Prediction"),
    
    sidebarLayout(
        sidebarPanel(
            numericInput("schools_distance", "Distance to School 0 (km):", value = 1),
            numericInput("schools_rating", "Rating of School 1 (1-10):", value = 8),
            numericInput("bathrooms", "Number of Bathrooms:", value = 2),
            numericInput("bedrooms", "Number of Bedrooms:", value = 3),
            numericInput("year_built", "Year Built:", value = 2000),
            numericInput("latitude", "Latitude:", value = 37.77),
            numericInput("longitude", "Longitude:", value = -122.42),
            numericInput("lot_size", "Lot Size (sqft):", value = 5000),
            numericInput("living_area", "Living Area (sqft):", value = 2000),
            numericInput("property_tax_rate", "Property Tax Rate (%):", value = 1.2),
            actionButton("predict", "Predict Price")
        ),
        
        mainPanel(
            h3("Predicted House Price"),
            verbatimTextOutput("prediction")
        )
    )
)

# Define server logic
server <- function(input, output) {
    
    # Reactively predict the house price based on user inputs
    prediction <- eventReactive(input$predict, {
        
        # Create a data frame with user inputs
        user_data <- data.frame(
            `schools/0/distance` = input$schools_distance,
            `schools/1/rating` = input$schools_rating,
            `bathrooms` = input$bathrooms,
            `resoFacts/bedrooms` = input$bedrooms,
            `yearBuilt` = input$year_built,
            `latitude` = input$latitude,
            `longitude` = input$longitude,
            `lotSizeInSqft` = input$lot_size,
            `livingArea` = input$living_area,
            `propertyTaxRate` = input$property_tax_rate
        )
        
        # Convert the data frame to a pandas DataFrame
        py$user_data <- r_to_py(user_data)
        
        # Make prediction using the Python model
        predicted_price <- py$model$predict(py$user_data)[[1]]
        
        # Return the predicted price
        paste("The estimated house price is: $", round(predicted_price, 2))
    })
    
    # Display the prediction
    output$prediction <- renderText({
        prediction()
    })
}

# Run the application
shinyApp(ui = ui, server = server)
'''