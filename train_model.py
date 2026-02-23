import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shap

def train_and_evaluate(data_path='patpat_ML_Ready_v2.csv', model_save_path='xgboost_car_price_model.json'):
    if not os.path.exists(data_path):
        print(f"Error: Could not find '{data_path}'. Please run preprocess.py first.")
        return
        
    # 1. Load the cleaned data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    
    df['Location'] = df['Location'].astype('category')
    df['Fuel_Type'] = df['Fuel_Type'].astype('category')
    df['Make_Model'] = df['Make_Model'].astype('category')

    # 2. Split into Features (X) and Target (y)
    X = df.drop('Price', axis=1)
    y = df['Price']

    # 3. Train/Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Initialize the XGBoost Regressor with our Hyperparameters
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        enable_categorical=True,
        random_state=42
    )

    # 5. Train the Model
    print("Training the XGBoost model...")
    model.fit(X_train, y_train)
    print("Training Complete!\n")

    # Save the model ---
    model.save_model(model_save_path)
    print(f"Model successfully saved to '{model_save_path}'\n")

    # 6. Make Predictions on the Test Set
    y_pred = model.predict(X_test)

    # 7. Calculate Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("--- RESULTS FOR YOUR REPORT ---")
    print(f"R-Squared (R2): {r2:.4f}")
    print(f"RMSE: Rs. {rmse:,.0f}")
    print(f"MAE:  Rs. {mae:,.0f}")
    print("-------------------------------\n")

    # Plot 1: Actual vs Predicted Prices
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')

    # 8. Draw a perfect prediction diagonal line
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)

    plt.xlim(0, 20000000) 
    plt.ylim(0, 20000000)

    plt.title('Actual vs. Predicted Prices')
    plt.xlabel('Actual Market Price (LKR)')
    plt.ylabel('AI Predicted Price (LKR)')
    plt.ticklabel_format(style='plain', axis='both')
    plt.grid(True, alpha=0.3)
    
    
    plt.savefig('actual_vs_predicted.png')
    print("Saved plot: 'actual_vs_predicted.png'")
    
    # 9. PLOT 2: Feature Importance
    plt.figure(figsize=(10, 6))
    # Get feature importance from the model
    importances = model.feature_importances_
    features = X.columns
    # Sort them
    indices = np.argsort(importances)

    plt.barh(range(len(indices)), importances[indices], color='teal', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.title('XGBoost Feature Importance (What drives the price?)')
    plt.xlabel('Relative Importance')
    plt.grid(True, axis='x', alpha=0.3)
    
    # Save the plot
    plt.savefig('feature_importance.png')
    print("Saved plot: 'feature_importance.png'")
    
    
    
    # 10. PLOT 3: SHAP Summary Plot
   
    print("Calculating SHAP values... (This might take a few seconds)")
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test)
    
    # Generate the SHAP Summary Plot
    plt.figure(figsize=(10, 6))
    plt.title("SHAP Summary Plot: How Features Impact Vehicle Price", fontsize=14)
    shap.summary_plot(shap_values, X_test, show=False) 
    
   
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    print("Saved plot: 'shap_summary.png'")
   

if __name__ == "__main__":
    train_and_evaluate()
