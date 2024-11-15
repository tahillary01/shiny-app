library(shiny)
library(leaflet)

# Load the pre-trained XGBoost model
model <- readRDS("xgboost_model.rds")

# Define UI for the app
ui <- fluidPage(
  titlePanel("House Price Prediction with Location Map"),
  
  sidebarLayout(
    sidebarPanel(
      numericInput("gdp_2019", "GDP (2019):", value = 60000),
      numericInput("livingArea", "Living Area (sqft):", value = 2000),
      numericInput("longitude", "Longitude:", value = -122.42),
      numericInput("latitude", "Latitude:", value = 37.77),
      checkboxInput("homeType_SingleFamily", "Is Single-Family Home?", value = TRUE),
      checkboxInput("hasGarage", "Has Garage?", value = TRUE),
      numericInput("lotSizeInSqft", "Lot Size (sqft):", value = 5000),
      numericInput("nearest_hospital_distance_km", "Distance to Nearest Hospital (km):", value = 2),
      numericInput("bedrooms", "Number of Bedrooms:", value = 3),
      numericInput("bathrooms", "Number of Bathrooms:", value = 2),
      numericInput("yearBuilt", "Year Built:", value = 2000),
      numericInput("distance_to_beach_km", "Distance to Beach (km):", value = 5),
      numericInput("distance_to_city_center", "Distance to City Center (km):", value = 10),
      numericInput("schools_distance", "Distance to School 0 (km):", value = 1),
      actionButton("predict", "Predict Price")
    ),
    
    mainPanel(
      h3("Predicted House Price"),
      verbatimTextOutput("prediction"),
      h3("House Location"),
      leafletOutput("map")  # Output for the map
    )
  )
)

# Define server logic
server <- function(input, output, session) {
  
  # Reactively predict the house price based on user inputs
  prediction <- eventReactive(input$predict, {
    
    # Create a data frame with user inputs
    user_data <- data.frame(
      gdp_2019 = input$gdp_2019,
      livingArea = input$livingArea,
      longitude = input$longitude,
      latitude = input$latitude,
      resoFacts.homeType_SingleFamily = as.integer(input$homeType_SingleFamily),
      resoFacts.hasGarage_True = as.integer(input$hasGarage),
      lotSizeInSqft = input$lotSizeInSqft,
      nearest_hospital_distance_km = input$nearest_hospital_distance_km,
      resoFacts.bedrooms = input$bedrooms,
      bathrooms = input$bathrooms,
      yearBuilt = input$yearBuilt,
      distance_to_beach_km = input$distance_to_beach_km,
      distance_to_city_center = input$distance_to_city_center,
      schools.0.distance = input$schools_distance
    )
    
    # Prepare the data for prediction
    user_data_matrix <- as.matrix(user_data)
    
    # Make prediction using the loaded XGBoost model
    predicted_price <- predict(model, user_data_matrix)
    
    # Return the predicted price
    paste("The estimated house price is: $", format(round(predicted_price, 2), big.mark = ","))
  })
  
  # Display the prediction
  output$prediction <- renderText({
    prediction()
  })
  
  # Render the map
  output$map <- renderLeaflet({
    req(input$latitude, input$longitude)  # Ensure inputs are available
    
    leaflet() %>%
      addTiles() %>%
      setView(lng = input$longitude, lat = input$latitude, zoom = 12) %>%
      addMarkers(lng = input$longitude, lat = input$latitude, popup = "House Location")
  })
}

# Run the application
shinyApp(ui = ui, server = server)
