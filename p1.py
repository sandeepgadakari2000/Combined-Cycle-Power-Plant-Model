import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib

# --- 1. Setup and Data Loading ---

# Define constants
FILE_NAME = r"C:\Users\sande\Downloads\CCPP_data.csv"
TARGET_COLUMN = 'PE'
FEATURE_COLUMNS = ['AT', 'V', 'AP', 'RH'] # AT (Temperature), V (Exhaust Vacuum), AP (Ambient Pressure), RH (Relative Humidity)

# Primary metric for comparison (RMSE)
def rmse(y_true, y_pred):
    """Calculates the Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

print(f"Loading data from {FILE_NAME}...")

try:
    # Load the dataset
    data = pd.read_csv(FILE_NAME)
    print(f"Data loaded successfully. Total records: {len(data)}")
    
    # Clean column names (replace spaces and standardize for better access)
    data.columns = data.columns.str.strip().str.replace(' ', '_')
    
    # Check for missing values (imputation is usually needed, but this dataset is clean)
    if data.isnull().sum().any():
        print("\nWarning: Missing values detected. Simple removal for this example.")
        data.dropna(inplace=True)

    # Define features (X) and target (y)
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

except FileNotFoundError:
    print(f"Error: The file '{FILE_NAME}' was not found. Please ensure it is in the correct directory.")
    exit()
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()


# --- 2. Data Splitting and Validation Strategy ---

# Split data into training (80%) and testing (20%) sets
# The test set is strictly held out for final evaluation.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {len(X_train)} records")
print(f"Test set size: {len(X_test)} records")

# Validation Strategy: Use 5-Fold Cross-Validation (CV) on the training data.
N_SPLITS = 5
cv_strategy = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)


# --- 3. Model Comparison (Using Cross-Validation) ---

# Model 1: Linear Regression (Simple, explainable baseline)
model_lr = LinearRegression()
print("\n--- Model Comparison (Cross-Validation) ---")

# Calculate RMSE scores using 5-Fold CV. 'neg_mean_squared_error' is used because cross_val_score maximizes the score.
# We then take the negative, find the mean, and take the square root to get RMSE.
lr_scores = -cross_val_score(
    model_lr, 
    X_train, 
    y_train, 
    cv=cv_strategy, 
    scoring='neg_mean_squared_error',
    n_jobs=-1 # Use all processors
)
lr_rmse_scores = np.sqrt(lr_scores)
lr_mean_rmse = lr_rmse_scores.mean()

print(f"1. Linear Regression (Baseline):")
print(f"   Avg. CV RMSE: {lr_mean_rmse:.3f} MW (Std Dev: {lr_rmse_scores.std():.3f})")

# Model 2: Random Forest Regressor (Powerful non-linear model)
# Comparison 2a: Default Hyperparameters (n_estimators=100)
model_rf_default = RandomForestRegressor(random_state=42, n_jobs=-1)

rf_scores_default = -cross_val_score(
    model_rf_default, 
    X_train, 
    y_train, 
    cv=cv_strategy, 
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
rf_rmse_scores_default = np.sqrt(rf_scores_default)
rf_mean_rmse_default = rf_rmse_scores_default.mean()

print(f"2a. Random Forest (Default, n_estimators=100):")
print(f"    Avg. CV RMSE: {rf_mean_rmse_default:.3f} MW (Std Dev: {rf_rmse_scores_default.std():.3f})")


# Comparison 2b: Hyperparameter Tuning (Increased n_estimators)
# This satisfies the requirement to compare models with different hyperparameters.
N_ESTIMATORS_TUNED = 300
model_rf_tuned = RandomForestRegressor(n_estimators=N_ESTIMATORS_TUNED, random_state=42, n_jobs=-1)

rf_scores_tuned = -cross_val_score(
    model_rf_tuned, 
    X_train, 
    y_train, 
    cv=cv_strategy, 
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
rf_rmse_scores_tuned = np.sqrt(rf_scores_tuned)
rf_mean_rmse_tuned = rf_rmse_scores_tuned.mean()

print(f"2b. Random Forest (Tuned, n_estimators={N_ESTIMATORS_TUNED}):")
print(f"    Avg. CV RMSE: {rf_mean_rmse_tuned:.3f} MW (Std Dev: {rf_rmse_scores_tuned.std():.3f})")

# Select the best model based on the lowest average CV RMSE
best_model_name = ""
best_model_instance = None
best_rmse = float('inf')

model_options = {
    "Linear Regression": (model_lr, lr_mean_rmse),
    "Random Forest (Default)": (model_rf_default, rf_mean_rmse_default),
    "Random Forest (Tuned)": (model_rf_tuned, rf_mean_rmse_tuned),
}

for name, (model, rmse_val) in model_options.items():
    if rmse_val < best_rmse:
        best_rmse = rmse_val
        best_model_instance = model
        best_model_name = name

print(f"\nSelected Final Model: {best_model_name} with Validation RMSE: {best_rmse:.3f} MW")


# --- 4. Final Model Training and Evaluation ---

# Train the selected best model on the entire training set
print(f"\nTraining the final model ({best_model_name}) on the full training set...")
best_model_instance.fit(X_train, y_train)


# Evaluate the final model on the unseen test set
print("Evaluating final model performance on the test set...")
y_pred = best_model_instance.predict(X_test)

# Calculate metrics
final_rmse = rmse(y_test, y_pred)
final_mae = mean_absolute_error(y_test, y_pred)
final_r2 = r2_score(y_test, y_pred)


# --- 5. Project Summary and Output ---

print("\n" + "="*50)
print(f"PROJECT SUMMARY: CCPP ENERGY OUTPUT PREDICTION")
print("="*50)
print("1. ML Approach & Metric:")
print(f"   - Task Type: Regression")
print(f"   - Target: Net Hourly Electrical Energy Output (PE, MW)")
print(f"   - Primary Evaluation Metric: RMSE (Root Mean Squared Error)")

print("\n2. Feature Selection:")
print(f"   - Features Used: {', '.join(FEATURE_COLUMNS)}")

print("\n3. Models Compared (using 5-Fold Cross-Validation):")
print(f"   - Linear Regression: Avg. CV RMSE = {lr_mean_rmse:.3f} MW")
print(f"   - Random Forest (Default): Avg. CV RMSE = {rf_mean_rmse_default:.3f} MW")
print(f"   - Random Forest (Tuned): Avg. CV RMSE = {rf_mean_rmse_tuned:.3f} MW")

print("\n4. Final Model Selected:")
print(f"   - Model: {best_model_name}")

print("\n5. Final Evaluation on Unseen Test Set:")
print(f"   - Test Set RMSE (Primary Metric): {final_rmse:.3f} MW")
print(f"   - Test Set MAE (Average Error): {final_mae:.3f} MW")
print(f"   - Test Set R² Score (Variance Explained): {final_r2:.4f}")

print("\nInterpretation:")
print(f"The average prediction error (MAE) is only {final_mae:.3f} MW, and the model explains {final_r2*100:.2f}% of the variance in the energy output, indicating excellent performance.")

# Save the final model for deployment (optional but good practice)
joblib.dump(best_model_instance, f'{best_model_name.replace(" ", "_")}_final_model.joblib')
print(f"Final model saved as {best_model_name.replace(' ', '_')}_final_model.joblib")

# Example of a single prediction
example_data = X_test.iloc[0:1]
example_prediction = best_model_instance.predict(example_data)[0]
example_actual = y_test.iloc[0]

print("\nQuick Demo:")
print(f"   - Example Input (AT, V, AP, RH): {example_data.values[0]}")
print(f"   - Actual Output (PE): {example_actual:.2f} MW")
print(f"   - Predicted Output (PE): {example_prediction:.2f} MW")



print(model_rf_tuned.predict([[26, 59, 1012.23, 58.77]]))

print(model_rf_tuned.predict([[13.97, 38.47	,1015.15,55.28]]))