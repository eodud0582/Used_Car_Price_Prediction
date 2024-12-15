#### Import Libraries ####
# Data Import/Export
library(haven)       

# Data Manipulation and Visualization
library(tidyverse)   
library(dplyr)      
library(ggplot2)    
library(plotly)      
library(viridis)    
library(recipes)

# Benchmark Datasets and Preprocessing
library(mlbench)     
library(recipes)    

# Machine Learning Framework: Training and tuning models
library(caret)       

# Random Forest
library(randomForest)
library(doParallel)   

# XGBoost and Gradient Boosting
library(xgboost)     
library(Matrix)     

# ElasticNet and Lasso
library(glmnet)     

# Support Vector Machines (including regression)
library(e1071)      

# MARS (Multivariate Adaptive Regression Splines)
library(earth)      

# Model Interpretation: Partial dependence plots
library(pdp)         

# Quantile Regression
library(quantreg)   

set.seed(2024)

#### Data Preprocessing ####

dt <- read.csv("train.csv")

dt <- as_tibble(dt)


dt <- dt %>%
  mutate(car_grade = case_when(
    # Economic brands
    tolower(brand) %in% c("toyota", "honda", "chevrolet", "ford", "nissan", "hyundai", "kia", "mazda", 
                          "subaru", "volkswagen", "mitsubishi", "fiat", "saturn", "scion", "suzuki", 
                          "chrysler", "plymouth", "smart") ~ "Economic",
    # Luxury brands
    tolower(brand) %in% c("tesla", "bmw", "mercedes-benz", "audi", "lexus", "infiniti", "acura", 
                          "cadillac", "volvo", "lincoln", "genesis", "buick", "alfa", "polestar", "lucid") ~ "Luxury",
    # Sport/Luxury brands
    tolower(brand) %in% c("porsche", "ferrari", "lamborghini", "maserati", "aston", "mclaren", 
                          "bugatti", "lotus", "jaguar", "bentley", "rolls-royce", "maybach", 
                          "karma", "rivian") ~ "Sport/Luxury",
    # Off-road/Utility brands
    tolower(brand) %in% c("jeep", "ram", "gmc", "land", "hummer") ~ "Off-road/Utility",
    # Default category
    TRUE ~ "Other"
  ))

dt <- dt %>%
  mutate(fuel_category = case_when(
    fuel_type %in% c("Gasoline") ~ "Gasoline",
    fuel_type %in% c("Hybrid", "Plug-In Hybrid") ~ "Hybrid",
    fuel_type %in% c("Diesel") ~ "Diesel",
    fuel_type %in% c("E85 Flex Fuel") ~ "Flex Fuel",
    fuel_type %in% c("not supported", "-", "") ~ "Unknown",
    TRUE ~ "Other"
  ))

dt %>%
  summarise(
    unique_brands = list(unique(brand)),
    unique_fuel_types = list(unique(fuel_type)),
    unique_transmissions = list(unique(transmission)),
    unique_accidents = list(unique(accident)),
    unique_engine = list(unique(engine)),
    unique_ext_col = list(unique(ext_col)),
    unique_int_col = list(unique(int_col))
  )

dt <- dt %>%
  mutate(transmission_category = case_when(
    grepl("A/T|Automatic", transmission, ignore.case = TRUE) ~ "Automatic",
    grepl("M/T|Manual", transmission, ignore.case = TRUE) ~ "Manual",
    grepl("CVT", transmission, ignore.case = TRUE) ~ "CVT",
    grepl("Auto-Shift|Dual Shift Mode", transmission, ignore.case = TRUE) ~ "Auto-Shift",
    TRUE ~ "Other"
  ))

dt <- dt %>%
  mutate(accident_status = case_when(
    accident == "None reported" ~ "No Accident",
    accident == "At least 1 accident or damage reported" ~ "Accident Reported",
    TRUE ~ "Unknown"
  ))

dt <- dt %>%
  mutate(hp = as.numeric(gsub("HP.*", "", engine)),  # Extract horsepower
         displacement = as.numeric(gsub(".* ([0-9]\\.[0-9])L.*", "\\1", engine)),  # Extract displacement in liters
         engine_type = case_when(  # Classify engine types
           grepl("V6", engine, ignore.case = TRUE) ~ "V6",
           grepl("V8", engine, ignore.case = TRUE) ~ "V8",
           grepl("I4|4 Cylinder", engine, ignore.case = TRUE) ~ "I4",
           grepl("Flat 6", engine, ignore.case = TRUE) ~ "Flat 6",
           grepl("V10", engine, ignore.case = TRUE) ~ "V10",
           grepl("V12", engine, ignore.case = TRUE) ~ "V12",
           TRUE ~ "Other"
         ),
         fuel_type = case_when(  # Simplify fuel type
           grepl("Electric", engine, ignore.case = TRUE) ~ "Electric",
           grepl("Gas/Electric Hybrid", engine, ignore.case = TRUE) ~ "Hybrid",
           grepl("Gasoline", engine, ignore.case = TRUE) ~ "Gasoline",
           grepl("Diesel", engine, ignore.case = TRUE) ~ "Diesel",
           grepl("Flex Fuel", engine, ignore.case = TRUE) ~ "Flex Fuel",
           TRUE ~ "Other"
         ),
         hp_category = case_when(  # Categorize horsepower
           hp < 200 ~ "Low HP",
           hp >= 200 & hp <= 400 ~ "Medium HP",
           hp > 400 ~ "High HP",
           TRUE ~ "Unknown"
         ),
         displacement_category = case_when(  # Categorize engine displacement
           displacement < 2.5 ~ "Small Engine",
           displacement >= 2.5 & displacement <= 4.0 ~ "Medium Engine",
           displacement > 4.0 ~ "Large Engine",
           TRUE ~ "Unknown"
         ))

dt <- dt %>%
  mutate(color_category = case_when(
    grepl("Black", ext_col, ignore.case = TRUE) ~ "Black",
    grepl("White", ext_col, ignore.case = TRUE) ~ "White",
    grepl("Gray|Grey|Silver", ext_col, ignore.case = TRUE) ~ "Gray",
    TRUE ~ "Other"  # All other colors fall under 'Other'
  ))

dt <- dt %>%
  mutate(int_color_category = case_when(
    grepl("Black", int_col, ignore.case = TRUE) ~ "Black",
    grepl("Gray|Grey|Slate", int_col, ignore.case = TRUE) ~ "Gray",
    grepl("Beige|Tan|Parchment|Oyster|Sand", int_col, ignore.case = TRUE) ~ "Beige",
    TRUE ~ "Other"  # All other colors fall under 'Other'
  ))

dt <- dt %>%
  group_by(engine_type) %>%
  mutate(
    hp = ifelse(is.na(hp), mean(hp, na.rm = TRUE), hp),
    displacement = ifelse(is.na(displacement), mean(displacement, na.rm = TRUE), displacement)
  ) %>%
  ungroup()

# Replace empty strings in "clean_title" with "Unknown"
dt <- dt %>%
  mutate(clean_title = ifelse(clean_title == "", "Unknown", clean_title))

dt <- dt %>%
  mutate(color_category = relevel(factor(color_category), ref = "Other"),
         int_color_category = relevel(factor(int_color_category), ref = "Other"))


dat <- dt %>% 
  select(-c(brand, model, fuel_type, engine, transmission, ext_col, int_col, accident, hp_category, displacement_category)) %>% 
  filter(hp != "NaN")

write.csv(dat, "train_clean.csv")



#### Exploaratory Data Analysis ####

## These EDA sections were initially run using Python.
## Some parts of the code do not work seamlessly on the R side.

# Function to display the full DataFrame with its dimensions
print_all <- function(df) {
  # Show the number of rows and columns in the DataFrame
  print(dim(df)) 
  # Print the entire DataFrame
  print(df)
}

# Function to display the first 'n' rows of the DataFrame with its dimensions
print_cols <- function(df, n = 5) {
  # Show the number of rows and columns in the DataFrame
  print(dim(df))
  # Print the first 'n' rows of the DataFrame
  print(head(df, n))
}

# Read the CSV file into a DataFrame
train <- read_csv('train.csv') %>%
  column_to_rownames(var = "id") # Use the 'id' column as the row names

# Display the first 2 rows of the DataFrame
print_cols(train, 2)

# Check for missing values in the DataFrame
temp <- colSums(is.na(train)) # Count missing values for each column

# Print the columns with missing values
print(temp[temp != 0])

# Create a horizontal bar plot for columns with missing values
missing_values <- temp[temp != 0]
barplot(missing_values, horiz = TRUE, las = 1, main = "Missing Values per Column")

# fuel_type
table(train$fuel_type)

# clean_title
table(train$clean_title)

# accident
table(train$accident)

# Load necessary libraries
library(ggplot2)
library(reshape2)
library(viridis)

# Get column names with missing values
null_cols <- colnames(train)[colSums(is.na(train)) > 0]

# Set up the plotting layout (3 plots side by side)
par(mfrow = c(1, 3), mar = c(5, 5, 2, 2))

# Plot heatmap for 'fuel_type'
fuel_type_data <- train[null_cols]
fuel_type_data <- fuel_type_data[order(fuel_type_data$fuel_type), ]
heatmap_data1 <- as.matrix(is.na(fuel_type_data))
image(1:ncol(heatmap_data1), 1:nrow(heatmap_data1), t(heatmap_data1), col = viridis(100), axes = FALSE)
title('fuel_type')

# Plot heatmap for 'accident'
accident_data <- train[null_cols]
accident_data <- accident_data[order(accident_data$accident), ]
heatmap_data2 <- as.matrix(is.na(accident_data))
image(1:ncol(heatmap_data2), 1:nrow(heatmap_data2), t(heatmap_data2), col = viridis(100), axes = FALSE)
title('accident')

# Plot heatmap for 'clean_title'
clean_title_data <- train[null_cols]
clean_title_data <- clean_title_data[order(clean_title_data$clean_title), ]
heatmap_data3 <- as.matrix(is.na(clean_title_data))
image(1:ncol(heatmap_data3), 1:nrow(heatmap_data3), t(heatmap_data3), col = viridis(100), axes = FALSE)
title('clean_title')

# Reset layout
par(mfrow = c(1, 1))

# Get the list of numeric columns
numeric_cols <- names(train)[sapply(train, is.numeric)]

# Print the list of numeric columns
print(numeric_cols)


library(ggplot2)
library(e1071)

# Function to plot distribution of a numerical column
plot_dist <- function(df, column) {
  # Extract the column data
  series <- df[[column]]
  
  # Calculate basic statistics
  stats <- list(
    min = min(series),
    max = max(series),
    mean = mean(series),
    median = median(series),
    q25 = quantile(series, 0.25),
    q75 = quantile(series, 0.75)
  )
  
  # Print basic statistics
  cat(paste(column, "distribution statistics:\n"))
  for (key in names(stats)) {
    cat(paste(key, ":", round(stats[[key]], 2), "\n"))
  }
  
  # Plot 1: Histogram with KDE
  par(mfrow = c(1, 2))  # Set the plot area to 1 row, 2 columns
  hist(series, probability = TRUE, main = paste(column, "Histogram"), 
       xlab = column, ylab = "Frequency", col = "lightblue", border = "black")
  lines(density(series), col = "blue")  # Add kernel density estimate (KDE)
  
  # Add normal distribution lines
  mean_val <- mean(series)
  std_val <- sd(series)
  abline(v = mean_val, col = "red", lwd = 2)  # Mean line
  abline(v = mean_val - std_val, col = "red", lty = 2)  # Mean - 1 Std line
  abline(v = mean_val + std_val, col = "red", lty = 2)  # Mean + 1 Std line
  
  # Plot 2: Boxplot
  boxplot(series, main = paste(column, "Boxplot"), horizontal = TRUE, col = "lightgreen")
  
  # Show skewness and kurtosis
  cat("Skewness:", skewness(series), "\n")
  cat("Kurtosis:", kurtosis(series), "\n")
}

# Example usage for numeric columns in 'train' dataframe
for (col in numeric_cols) {
  plot_dist(train, col)
}


# IQR Calculation
Q1 <- apply(train[, numeric_cols], 2, function(x) quantile(x, 0.25))
Q3 <- apply(train[, numeric_cols], 2, function(x) quantile(x, 0.75))
IQR <- Q3 - Q1

# Condition for outliers
cond <- (train[, numeric_cols] < (Q1 - 1.5 * IQR)) | (train[, numeric_cols] > (Q3 + 1.5 * IQR))

# Extract rows with outliers
temp <- train[cond, numeric_cols, drop = FALSE]

# Find rows where any column has an outlier
outliers <- temp[apply(temp, 1, function(x) any(!is.na(x))), ]
outliers

# Assuming 'train' is the dataset and 'numeric_cols' is the list of numeric columns
# Create an empty list to store the plots
plots <- list()

# Loop over each numeric column
for (col in numeric_cols) {
  
  # Create the histogram with density curve
  p <- ggplot(train, aes_string(x = col)) +
    geom_histogram(aes(y = ..density..), bins = 30, fill = "blue", alpha = 0.5) +
    geom_density(color = "black") +
    
    # Add scatter plot for outliers
    geom_point(data = outliers, aes_string(x = col, y = rep(0, nrow(outliers))),
               color = "red", shape = 4) +
    
    # Set plot title and remove x-axis label
    ggtitle(paste("Histogram of", col, "with Outliers")) +
    theme(axis.title.x = element_blank())
  
  # Store the plot in the list
  plots[[col]] <- p
}

# Display the plots
gridExtra::grid.arrange(grobs = plots, ncol = 1)


# Setting independent and dependent variables
X <- train[, numeric_cols]  # Selecting numeric columns
X <- X[, !colnames(X) %in% 'price']  # Dropping 'price' column
y <- train$price  # Dependent variable 'price'

# Adding constant term (intercept) to the model
X <- cbind(1, X)  # Adding a column of 1s as constant

# Fitting the linear regression model
model <- lm(y ~ ., data = X)

# Calculating Cook's Distance
cooks_d <- cooks.distance(model)

# Visualizing Cook's Distance
plot(cooks_d, type = "h", main = "Cook's Distance for Outlier Detection", 
     xlab = "Observation Index", ylab = "Cook's Distance", lwd = 2)

# Detecting outliers (Typically, Cook's Distance > 4/n is considered an outlier)
n <- nrow(train)
outliers <- which(cooks_d > 4/n)
cat("Detected outliers at indices:", outliers, "\n")

# Load required libraries
library(ggplot2)
library(GGally)  # for pairplot equivalent
library(corrplot)  # for correlation heatmap
library(dplyr)

# Ensure all columns are numeric before correlation
# First, select only numeric columns
train_numeric <- train[, sapply(train, is.numeric)]

# Pairplot to visualize relationships between features
tryCatch({
  ggpairs(train_numeric)
}, error = function(e) {
  print("Error creating pairplot:")
  print(e)
})

# Correlation matrix 
tryCatch({
  corr_matrix <- cor(train_numeric, use = "complete.obs")
  
  # Correlation heatmap
  corrplot(corr_matrix, 
           method = "color",     # Color-based representation
           type = "lower",       # Only show lower triangle
           diag = FALSE,         # Exclude diagonal 
           addCoef.col = "black",# Add correlation coefficients 
           tl.col = "black",     # Text label color
           tl.srt = 45,          # Rotate text labels
           col = colorRampPalette(c("blue", "white", "red"))(200),  # Color scheme
           title = "Correlation Heatmap")
}, error = function(e) {
  print("Error creating correlation matrix or heatmap:")
  print(e)
})

# Adjust the plot to remove the upper triangle (using mask)
# Note: The mask isn't a simple feature in ggplot, so we filter manually.
corr[mask] <- NA

# Draw the heatmap with the mask applied
library(reshape2)  # for melt function
corr_melted <- melt(corr, na.rm = TRUE)

ggplot(corr_melted, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Correlation Heatmap") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# EDA on categorical variables
eda_categorical <- function(df, categorical_cols, target) {
  for (col in categorical_cols) {
    cat(rep("=", 50), "\n")
    cat(paste("--- EDA for", col, "---\n"))
    
    unique_count <- length(unique(df[[col]]))
    if (unique_count > 30) {
      cat(paste("Column", col, "has too many unique values (", unique_count, "). Consider binning or filtering.\n"))
    }
    
    # Display unique values and counts (Top 20)
    unique_values_counts <- head(sort(table(df[[col]]), decreasing = TRUE), 20)
    cat(paste("Unique values and counts for", col, "(Top 20):\n"))
    print(unique_values_counts)
    
    # Missing values count
    missing_count <- sum(is.na(df[[col]]))
    cat(paste("Missing values for", col, ":", missing_count, "\n"))
    
    # Plot unique values and counts
    par(mfrow = c(1, 2), mar = c(5, 5, 2, 2), cex = 0.8)
    
    # 1) Table visualization of unique values and counts
    plot(1, type = "n", axes = FALSE, xlab = "", ylab = "", xlim = c(0, 1), ylim = c(0, length(unique_values_counts)))
    text(0.5, 1:length(unique_values_counts), labels = paste(names(unique_values_counts), unique_values_counts), cex = 0.8)
    title(main = paste("Unique values and counts for", col, "(Top 20)"))
    
    # 2) Bar plot visualization
    barplot(unique_values_counts, main = paste("Count plot for", col, "(Top 20)"), las = 2, cex.names = 0.7)
    
    # Analyze relationship between column and target variable
    if (target %in% colnames(df)) {
      target_mean <- aggregate(df[[target]] ~ df[[col]], data = df, FUN = mean)
      colnames(target_mean) <- c(col, paste("Mean of", target))
      cat(paste("Analyzing relationship between", col, "and", target, "...\n"))
      
      # Plot mean values
      par(mfrow = c(1, 2), mar = c(5, 5, 2, 2), cex = 0.8)
      
      # 1) Table of means
      plot(1, type = "n", axes = FALSE, xlab = "", ylab = "", xlim = c(0, 1), ylim = c(0, nrow(target_mean)))
      text(0.5, 1:nrow(target_mean), labels = paste(target_mean[[col]], round(target_mean[[paste("Mean of", target)]], 2)), cex = 0.8)
      title(main = paste("Mean of target variable", target, "by", col, "(Top 20)"))
      
      # 2) Bar plot of means
      barplot(target_mean[[paste("Mean of", target)]], names.arg = target_mean[[col]], col = "skyblue", main = paste("Mean of target variable", target, "by", col, "(Top 20)"))
      
      # Boxplot and Violin plot
      top_categories <- target_mean[[col]]
      cond <- df[[col]] %in% top_categories
      par(mfrow = c(1, 2), mar = c(5, 5, 2, 2))
      
      # 1) Boxplot
      boxplot(df[[target]] ~ df[[col]], data = df[cond, ], col = "skyblue", main = paste("Boxplot of", target, "by", col, "(Top 20)"))
      
      # 2) Violin plot
      library(ggplot2)
      ggplot(df[cond, ], aes(x = factor(df[[col]]), y = df[[target]], fill = factor(df[[col]]))) +
        geom_violin(trim = FALSE) +
        theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
        ggtitle(paste("Violin plot of", target, "by", col, "(Top 20)"))
      
    }
  }
}

# Example usage:
categorical_cols <- names(train)[sapply(train, is.factor)]
eda_categorical(train, categorical_cols, 'price')



#### Main Analysis ####

#### Import Cleaned Train Dataset ####
dat <- read.csv("train_clean.csv")
dat <- as_tibble(dat)

#### Revised OLS with log-transforemd price ####

#### Split the Data into Training and Test Sets ####
train_index <- createDataPartition(dat$price, p = 0.8, list = FALSE)
train <- dat[train_index, ]
test <- dat[-train_index, ]

#### Visualize Distributions of Raw Price and Log-Transformed Price ####
ggplot(train, aes(x = price)) +
  geom_histogram(bins = 100, fill = "blue", alpha = 0.7) +
  labs(title = "Distribution of Raw Price", x = "Price", y = "Count") +
  theme_minimal()

ggplot(train, aes(x = log(price))) +
  geom_histogram(bins = 100, fill = "red", alpha = 0.7) +
  labs(title = "Distribution of Log-Transformed Price", x = "Log(Price)", y = "Count") +
  theme_minimal()

#### OLS Regression with Log-Transformed Price ####
# Base OLS model using log(price) as the dependent variable
ols_model <- lm(log(price) ~ model_year + milage + clean_title + car_grade + 
                  fuel_category + transmission_category + accident_status + 
                  hp + displacement + engine_type + color_category + int_color_category, 
                data = train)

ols_og <- summary(ols_model)

#### Outlier Treatment ####
# Step 2: Identify influential observations in training data
influence_measures <- influence.measures(ols_model)
influence_df <- as.data.frame(influence_measures$infmat)

# Calculate Cook's Distance for Multivariate Outlier Detection
train <- train %>%
  mutate(cooks_distance = influence_df$cook.d) %>%
  mutate(outlier_flag = ifelse(cooks_distance > (4 / nrow(train)), TRUE, FALSE))

# Investigate outliers in training data
outlier_summary <- train %>%
  filter(outlier_flag == TRUE) %>%
  select(price, cooks_distance, everything())  # Examine flagged observations

# Remove flagged outliers from training data
train <- train %>%
  filter(outlier_flag == FALSE) %>%
  select(-c(cooks_distance, outlier_flag))  # Drop helper columns

#### Plot before & after Outlier Treatment ####
# Create a column to differentiate the datasets
train_before <- dat[train_index, ] %>%
  mutate(dataset = "Before Outlier Treatment")

train_after <- train %>%
  mutate(dataset = "After Outlier Treatment")

# Combine both datasets for visualization
combined_data <- bind_rows(train_before, train_after)

# Plot boxplot to compare price distribution
ggplot(combined_data, aes(x = dataset, y = log(price), fill = dataset)) +
  geom_boxplot(outlier.color = "red", outlier.size = 1.2, alpha = 0.6) +
  labs(
    title = "Log-Transformed Price Distribution Before and After Outlier Treatment",
    x = "Dataset",
    y = "Log(Price)"  # Corrected y-axis label
  ) +
  scale_y_continuous(labels = scales::comma) +
  theme_minimal() +
  theme(legend.position = "none")

summary(log(train_before$price))
summary(log(train_after$price))
#### OLS Model After Outlier Treatment ####
ols_model_refined <- lm(log(price) ~ milage * model_year + clean_title + car_grade + 
                          fuel_category + transmission_category + accident_status + 
                          hp + displacement + engine_type + color_category + int_color_category, 
                        data = train)

ols_ref <- summary(ols_model_refined)

#### OLS: Performance Metrics ####

# Performance on training data (refined model)
predict_log_ref_train <- predict(ols_model_refined, newdata = train)
predict_price_train <- exp(predict_log_ref_train)  # Roll back log(price) to price
mae_ref_train <- mean(abs(predict_price_train - train$price))
rmse_ref_train <- sqrt(mean((predict_price_train - train$price)^2))
r_squared_ref_train <- ols_ref$r.squared

cat("Training Performance:\n")
cat("R-squared:", r_squared_ref_train, "MAE:", mae_ref_train, "RMSE:", rmse_ref_train, "\n")

# Performance on test data
predict_log_test <- predict(ols_model_refined, newdata = test)
predict_price_test <- exp(predict_log_test)  # Roll back log(price) to price
mae_test <- mean(abs(predict_price_test - test$price))
rmse_test <- sqrt(mean((predict_price_test - test$price)^2))
r_squared_test <- 1 - sum((predict_price_test - test$price)^2) / 
  sum((mean(train$price) - test$price)^2)

cat("Test Performance:\n")
cat("R-squared:", r_squared_test, "MAE:", mae_test, "RMSE:", rmse_test, "\n")



#### ElasticNet ####

# Split the Data into Training and Test Sets
train_index <- createDataPartition(dat$price, p = 0.8, list = FALSE)
train <- dat[train_index, ]
test <- dat[-train_index, ]

# Apply log transformation to the target variable
train <- train %>%
  mutate(log_price = log(price))  # Log-transform the price only in training data

# Define preprocessing steps using a recipe
elastic_net_recipe <- recipe(log_price ~ ., data = train) %>%
  step_rm(price) %>%  
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>%  
  step_center(all_predictors()) %>%  
  step_scale(all_predictors())      

# Prepare the Recipe
prepped_recipe <- prep(elastic_net_recipe, training = train)

# Bake Train and Test Sets
train_prepped <- bake(prepped_recipe, new_data = train)
test_prepped <- bake(prepped_recipe, new_data = test)

# Set Control Parameters for Model Training
fitCtrl <- trainControl(
  method = "repeatedcv", 
  number = 5,           
  repeats = 3            
)

# Set the Tuning Grid for ElasticNet
glmnetGrid <- expand.grid(
  alpha = seq(0, 1, by = 0.1),  
  lambda = seq(0.01, 0.1, by = 0.01)  
)

# Train the Model
elastic_net_model <- train(
  log_price ~ .,  
  data = train_prepped,
  method = "glmnet",
  trControl = fitCtrl,
  tuneGrid = glmnetGrid
)

# Model Results
print(elastic_net_model)
# alpha 0.1    lambda 0.01    RMSE 0.5162244  RS 0.6144728  MAE 0.3657496

# Evaluate Model on Test Data
predicted_log_prices <- predict(elastic_net_model, newdata = test_prepped)

# Back-transform predictions to the original price scale
predicted_prices <- exp(predicted_log_prices)

# Evaluate performance on the original price scale
rmse_original <- sqrt(mean((test$price - predicted_prices)^2))
mae_original <- mean(abs(test$price - predicted_prices))

cat("RMSE (Original Scale):", rmse_original, "\n")
cat("MAE (Original Scale):", mae_original, "\n")

# Calculate R-squared for test set predictions
ss_residual <- sum((test$price - predicted_prices)^2)  # Sum of squared residuals
ss_total <- sum((test$price - mean(test$price))^2)     # Total sum of squares

r_squared_test <- 1 - (ss_residual / ss_total)

cat("R-squared (Original Scale):", r_squared_test, "\n")



#### Random Forest ####

dat_rf <- dat %>%
  select(-c(X, id))

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# Split data into training and testing sets
trainIndex <- createDataPartition(dat_rf$price, p = 0.8, list = FALSE)
train <- dat_rf[trainIndex, ]
test <- dat_rf[-trainIndex, ]

# Set control parameters for lighter model training
fitCtrl <- trainControl(method = "cv",   
                        number = 3,     
                        search = "grid", 
                        allowParallel = TRUE)

# Set limited grid search space 
rfGrid <- expand.grid(mtry = c(4, 6),       
                      min.node.size = 5,    
                      splitrule = "variance") 

# Fit random forest model with moderate number of trees
rf.res <- train(price ~ .,                           
                data = train, 
                method = "ranger",                   
                trControl = fitCtrl,                 
                tuneGrid = rfGrid,                   
                num.trees = 200,                     
                importance = "impurity",             
                metric = "RMSE",                     
                verbose = FALSE)

print(rf.res)
plot(rf.res)

# Predict on the test set
predictions <- predict(rf.res, test)

# Evaluate performance on test data
test_performance <- postResample(predictions, test$price)
print(test_performance)

# Check variable importance
rfImp <- varImp(rf.res)
plot(rfImp)

# Stop parallel cluster
stopCluster(cl)
registerDoSEQ()

# Feature importance plot
importance_df <- as.data.frame(varImp(rf.res)$importance)
importance_df$Feature <- rownames(importance_df)



#### XGBoost ####
dat_xg <- dat_rf %>%
  mutate(across(where(is.character), as.factor))

dmy <- dummyVars(" ~ .", data = dat_xg)
dat_xg <- data.frame(predict(dmy, newdata = dat_xg))

# Prepare the data (convert to matrix for XGBoost)
X <- as.matrix(dat_xg[, -which(colnames(dat_xg) == "price")])  
y <- dat_xg$price  

# Split the data into training and testing sets
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
train_X <- X[trainIndex, ]
test_X <- X[-trainIndex, ]
train_y <- y[trainIndex]
test_y <- y[-trainIndex]

# Convert training and testing sets 
dtrain <- xgb.DMatrix(data = train_X, label = train_y)
dtest <- xgb.DMatrix(data = test_X, label = test_y)

# Set parameters 
params <- list(
  booster = "gbtree",
  objective = "reg:squarederror", 
  eta = 0.05, 
  max_depth = 4, 
  subsample = 0.8,  
  colsample_bytree = 0.8 
)

# Use cross-validation to determine the optimal nrounds
xgb_cv <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = 200,  
  nfold = 3,  
  early_stopping_rounds = 10,  
  metrics = "rmse",
  verbose = 1
)

# Get the best number of rounds based on CV
best_nrounds <- xgb_cv$best_iteration

# Train the final model with the optimal number of rounds
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = best_nrounds,  
  watchlist = list(train = dtrain, eval = dtest),  
  eval_metric = "rmse",
  verbose = 1
)

# Make predictions on the test set
predictions <- predict(xgb_model, newdata = test_X)

# Make predictions on the training set
train_predictions <- predict(xgb_model, newdata = train_X)

# Evaluate performance on training data
train_rmse <- sqrt(mean((train_y - train_predictions)^2))
train_mae <- mean(abs(train_y - train_predictions))
train_ss_residual <- sum((train_y - train_predictions)^2)
train_ss_total <- sum((train_y - mean(train_y))^2)
train_r_squared <- 1 - (train_ss_residual / train_ss_total)

# Print training performance
cat("Training Performance:\n")
cat("RMSE:", train_rmse, "\n")
cat("MAE:", train_mae, "\n")
cat("R-squared:", train_r_squared, "\n")

# Evaluate performance on test data
test_performance <- postResample(predictions, test_y)
print(test_performance)

# Plot feature importance
importance_matrix <- xgb.importance(feature_names = colnames(train_X), model = xgb_model)
xgb.plot.importance(importance_matrix)


#### Bootstrapping ####

# Evaluate model performance (stability, RMSE) and prediction confidence intervals
# for reliable evaluation of model performance while considering uncertainty in predictions

# Set up bootstrap
n_iterations <- 100
bootstrap_rmse <- numeric(n_iterations)
bootstrap_preds <- matrix(0, nrow = n_iterations, ncol = length(test_y))

# Perform bootstrap iterations
for (i in 1:n_iterations) {
  # Resample the training data using bootstrap sampling
  resampled_indices <- sample(1:length(train_y), length(train_y), replace = TRUE)
  X_resampled <- train_X[resampled_indices, ]
  y_resampled <- train_y[resampled_indices]
  
  # Initialize a new model and train it for each iteration
  bootstrap_model <- xgboost(
    data = as.matrix(X_resampled),
    label = y_resampled,
    booster = "gbtree",
    objective = "reg:squarederror",
    eta = 0.05,
    max_depth = 4,
    subsample = 0.8,
    colsample_bytree = 0.8,
    nrounds = best_nrounds
  )
  
  # Predict using the test data
  y_pred_bootstrap <- predict(bootstrap_model, as.matrix(test_X))
  
  # Compute RMSE and save the result
  rmse_bootstrap <- sqrt(mean((test_y - y_pred_bootstrap)^2))
  bootstrap_rmse[i] <- rmse_bootstrap
  
  # Store predictions from each iteration
  bootstrap_preds[i, ] <- y_pred_bootstrap
}

# Compute mean and standard deviation of RMSE to evaluate variability
bootstrap_rmse_mean <- mean(bootstrap_rmse)
bootstrap_rmse_std <- sd(bootstrap_rmse)

cat("Bootstrap RMSE:", bootstrap_rmse_mean, "±", bootstrap_rmse_std, "\n")

# Compute confidence intervals (95% confidence interval) for predictions
bootstrap_mean <- colMeans(bootstrap_preds)
bootstrap_ci_lower <- apply(bootstrap_preds, 2, function(x) quantile(x, 0.025))
bootstrap_ci_upper <- apply(bootstrap_preds, 2, function(x) quantile(x, 0.975))

cat("Bootstrap Mean Prediction:", bootstrap_mean, "\n")
cat("95% Confidence Interval:", bootstrap_ci_lower, "to", bootstrap_ci_upper, "\n")

