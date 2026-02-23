import pandas as pd
import numpy as np
import os

def preprocess_data(input_file='patpat_targeted_data.csv', output_file='patpat_ML_Ready_v2.csv'):
    if not os.path.exists(input_file):
        print(f"Error: Could not find '{input_file}'. Have you run the scraper yet?")
        return

    # 1. Load the raw dataset
    df = pd.read_csv(input_file)
    print(f"Original dataset size: {df.shape}")

    # 2. Clean Price
    df = df[~df['Price'].isin(['Negotiable', 'N/A', None])]
    df['Price'] = df['Price'].astype(str).str.replace('Rs:', '', regex=False)
    df['Price'] = df['Price'].str.replace(',', '', regex=False).str.strip()
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

    # 3. Clean Mileage
    df['Mileage'] = df['Mileage'].astype(str).str.replace(',', '', regex=False)
    df['Mileage'] = pd.to_numeric(df['Mileage'], errors='coerce')

    # 4. Clean Engine
    df['Engine'] = df['Engine'].astype(str).str.extract(r'(\d+)')[0]
    df['Engine'] = pd.to_numeric(df['Engine'], errors='coerce')

    # 5. Clean Year
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

    # 6. Feature Engineering 
    df['Make_Model'] = df['Manufacturer'].astype(str) + ' ' + df['Model'].astype(str)
    df = df.drop(columns=['Manufacturer', 'Model']) 

    # 7. Drop missing values 
    df_clean = df.dropna().copy()

    # 8. Convert to integers
    df_clean['Year'] = df_clean['Year'].astype(int)
    df_clean['Engine'] = df_clean['Engine'].astype(int)
    df_clean['Price'] = df_clean['Price'].astype(int)
    df_clean['Mileage'] = df_clean['Mileage'].astype(int)

    # 9. formally declare the text columns as categories for XGBoost
    df_clean['Location'] = df_clean['Location'].astype('category')
    df_clean['Fuel_Type'] = df_clean['Fuel_Type'].astype('category')
    df_clean['Make_Model'] = df_clean['Make_Model'].astype('category')

    print(f"Cleaned dataset size: {df_clean.shape}")

    # 10. Save the ML-Ready dataset
    df_clean.to_csv(output_file, index=False)
    print(f"Data fully cleaned, encoded, and saved as '{output_file}'!")

    print(df_clean.head())

    return df_clean

if __name__ == "__main__":
    preprocess_data()
