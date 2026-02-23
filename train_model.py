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
        
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    df['Location'] = df['Location'].astype('category')
    df['Fuel_Type'] = df['Fuel_Type'].astype('category')
    df['Make_Model'] = df['Make_Model'].astype('category')

    
    X = df.drop('Price', axis=1)
    y = df['Price']

   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        enable_categorical=True,
        random_state=42
    )

    
    print("Training the XGBoost model...")
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
    print("Training Complete!\n")

  
    model.save_model(model_save_path)
    print(f"Model successfully saved to '{model_save_path}'\n")

  
    y_pred = model.predict(X_test)

  
    print("="*50)
    print(" MODEL TRAINING COMPLETED")
    print("="*50)
    print(f"Training Data Size : {len(X_train)} rows")
    print(f"Testing Data Size  : {len(X_test)} rows")
    print("-" * 50)
    print(f"R-Squared (R2)     : {r2_score(y_test, y_pred):.4f}")
    print(f"Mean Abs Error (MAE): Rs. {mean_absolute_error(y_test, y_pred):,.2f}")
    print(f"Root Mean Sq (RMSE) : Rs. {np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")
    print("="*50)

    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
    plt.xlim(0, 20000000) 
    plt.ylim(0, 20000000)
    plt.title('Actual vs. Predicted Prices')
    plt.xlabel('Actual Market Price (LKR)')
    plt.ylabel('AI Predicted Price (LKR)')
    plt.ticklabel_format(style='plain', axis='both')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png')
    print("Saved plot: 'actual_vs_predicted.png'")
    plt.close()
    
   
    plt.figure(figsize=(10, 6))
    importances = model.feature_importances_
    features = X.columns
    indices = np.argsort(importances)
    plt.barh(range(len(indices)), importances[indices], color='teal', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.title('XGBoost Feature Importance (What drives the price?)')
    plt.xlabel('Relative Importance')
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Saved plot: 'feature_importance.png'")
    plt.close()

    plt.figure(figsize=(10, 6))
    residuals = y_test - y_pre
    plt.scatter(y_pred, residuals, alpha=0.5, color='royalblue')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.title('Residuals Plot (Prediction Errors)', fontsize=14)
    plt.xlabel('Predicted Price (LKR)', fontsize=12)
    plt.ylabel('Residual Error (Actual - Predicted)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('residuals_plot.png')
    print("Saved plot: 'residuals_plot.png'")
    plt.close()

   
    results = model.evals_result()
    epochs = len(results['validation_0']['rmse'])
    x_axis = range(0, epochs)

    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, results['validation_0']['rmse'], label='Train RMSE')
    plt.plot(x_axis, results['validation_1']['rmse'], label='Test RMSE')
    plt.title('XGBoost Learning Curve', fontsize=14)
    plt.xlabel('Boosting Iterations (Trees)', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('learning_curve.png')
    print("Saved plot: 'learning_curve.png'")
    plt.close()
    
   
    print("\nCalculating SHAP values... (This might take a few seconds)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    plt.figure(figsize=(10, 6))
    plt.title("SHAP Summary Plot: How Features Impact Vehicle Price", fontsize=14)
    shap.summary_plot(shap_values, X_test, show=False) 
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    print("Saved plot: 'shap_summary.png'")
    plt.close()

    
    print("\nGenerating SHAP Waterfall Plot...")
    
  
    row_to_show = 0 
    
    plt.figure(figsize=(10, 6))
    
    
    shap_object = shap.Explanation(values=shap_values[row_to_show], 
                                   base_values=explainer.expected_value, 
                                   data=X_test.iloc[row_to_show],  
                                   feature_names=X_test.columns)
    
  
    shap.waterfall_plot(shap_object, show=False)
    
    plt.title(f"SHAP Waterfall: Price Breakdown for Test Vehicle #{row_to_show}", fontsize=14)
    plt.tight_layout()
    plt.savefig('shap_waterfall.png')
    print("Saved plot: 'shap_waterfall.png'\n")
    plt.close()

if __name__ == "__main__":
    train_and_evaluate()