import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import plotly.express as px


# Page Configuration
st.set_page_config(page_title="AutoIntel Pro", page_icon="🏎️", layout="wide", initial_sidebar_state="expanded") 


# Global CSS Injection 
st.markdown("""
<style>
    /* Main Background & Fonts */
    .stApp {
        background-color: #2f3e46;
    }
    
    /* Elegant Header */
    .main-header {
        font-family: 'Inter', 'Helvetica Neue', sans-serif;
        color: #ffffff;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 5px;
        letter-spacing: -0.5px;
    }
    .sub-header {
        color: #ffffff;
        font-size: 1.15rem;
        font-weight: 400;
        margin-bottom: 30px;
        letter-spacing: 0.2px;
    }
    
    /* Top Price Card - Luxury Dark Gradient */
    .price-card {
        background-color: #0F2027;
        background-image: linear-gradient(180deg, #0F2027 0%, #203A43 100%);
        border-radius: 16px;
        padding: 50px 30px;
        color: white;
        text-align: center;
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
        margin-bottom: 35px;
        border: 1px solid rgba(255,255,255,0.1);
        position: relative;
        overflow: hidden;
    }
    /* Subtle glow effect top left */
    .price-card::before {
        content: "";
        position: absolute;
        top: -50%; left: -50%;
        width: 100%; height: 100%;
        background: radial-gradient(circle, rgba(0, 255, 209, 0.1) 0%, transparent 70%);
    }
    
    .price-label {
        text-transform: uppercase;
        letter-spacing: 3px;
        font-size: 0.95rem;
        color: #A0B2C6;
        font-weight: 600;
        margin-bottom: 15px;
    }
    .price-value {
        font-size: 4.8rem;
        font-weight: 800;
        margin: 0;
        color: #00FFD1; /* High-contrast teal for the price */
        text-shadow: 0 4px 10px rgba(0, 255, 209, 0.3);
        line-height: 1.1;
    }
    .price-desc {
        font-size: 1.25rem;
        color: #E0E6ED;
        font-weight: 400;
        margin-top: 20px;
        letter-spacing: 0.5px;
    }
    
    /* Sections */
    .section-title {
        color: #ffffff;
        font-weight: 700;
        font-size: 1.4rem;
        margin-bottom: 20px;
        padding-bottom: 8px;
        border-bottom: 3px solid #00FFD1; /* Teal accent line */
        display: inline-block;
    }
    
    /* Info Cards (Vehicle Specs) */
    .info-card {
        background-color: #0F2027;
        background-image: linear-gradient(180deg, #0F2027 0%, #203A43 100%);
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.05);
        height: 100%;
    }
    .info-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .info-item:last-child {
        border-bottom: none;
        padding-bottom: 0;
    }
    .info-item:first-child {
        padding-top: 0;
    }
    .info-label {
        color: #A0B2C6;
        font-weight: 600;
        font-size: 1rem;
    }
    .info-value {
        color: #FFFFFF;
        font-weight: 700;
        font-size: 1.05rem;
        text-align: right;
    }
    
    /* Graph Card container wrapper */
    .graph-card {
        background-color: #0F2027;
        background-image: linear-gradient(180deg, #0F2027 0%, #203A43 100%);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.05);
        width: 100%;
        display: flex;
        justify-content: center;
    }
    
    /* =========================================
       SIDEBAR OVERRIDES 
       ========================================= */
    [data-testid="stSidebar"] {
        background-color: #0F2027;
        background-image: linear-gradient(180deg, #0F2027 0%, #203A43 100%);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Force sidebar text to light */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div[data-baseweb="select"] div {
        color: #E0E6ED !important;
    }
    
    /* Highlighted Titles in Sidebar */
    [data-testid="stSidebar"] h3 {
        color: #00FFD1 !important;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    /* Input Backgrounds */
    [data-testid="stSidebar"] div[data-baseweb="select"] > div,
    [data-testid="stSidebar"] input {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 6px;
    }

    /* Primary Button Override */
    .stButton > button {
        background-color: #2a9d8f !important;
        color: #0F2027 !important;
        font-weight: 800 !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(0, 255, 209, 0.25) !important;
        transition: all 0.3s ease !important;
        border-radius: 8px !important;
    }
    .stButton > button:hover {
        background-color: #00E6BC !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 255, 209, 0.4) !important;
    }
</style>
""", unsafe_allow_html=True)




@st.cache_resource
def load_models():
    # Load XGBoost mode
    model = xgb.XGBRegressor()
    model.load_model('xgboost_car_price_model.json')
    
    # Load data to get dropdowns
    df = pd.read_csv('patpat_ML_Ready_v2.csv')
    make_models_list = sorted(df['Make_Model'].dropna().unique().tolist())
    locations = sorted(df['Location'].dropna().unique().tolist())
    fuel_types = sorted(df['Fuel_Type'].dropna().unique().tolist())
    
    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    return model, explainer, make_models_list, locations, fuel_types, df

model, explainer, make_models_list, locations, fuel_types, raw_df = load_models()


car_dictionary = {}
for item in make_models_list:
    parts = str(item).split(" ", 1)
    if len(parts) == 2:
        make, car_model = parts[0], parts[1]
        if make not in car_dictionary:
            car_dictionary[make] = []
        car_dictionary[make].append(car_model)


# SIDEBAR
st.sidebar.markdown("### Vehicle Parameters")
st.sidebar.markdown("<p style='color:#7F8C8D; font-size:0.9rem; margin-top:-10px;'>Configure to estimate value</p>", unsafe_allow_html=True)

selected_make = st.sidebar.selectbox("Manufacturer", sorted(car_dictionary.keys()))
selected_model = st.sidebar.selectbox("Model", sorted(car_dictionary[selected_make]))
year = st.sidebar.slider("Manufacture Year", min_value=1990, max_value=2026, value=2015, step=1)
engine = st.sidebar.number_input("Engine Capacity (cc)", min_value=600, max_value=5000, value=1000, step=100)
mileage = st.sidebar.number_input("Mileage (km)", min_value=0, max_value=300000, value=50000, step=5000)
selected_location = st.sidebar.selectbox("Location", locations)
selected_fuel = st.sidebar.selectbox("Fuel Type", fuel_types)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
predict_button = st.sidebar.button("Generate Valuation ➜", type="primary", use_container_width=True)

def prepare_input(loc, mil, eng, yr, fuel, make_mod):
    df = pd.DataFrame({
        'Location': [loc], 'Mileage': [mil], 'Engine': [eng], 
        'Year': [yr], 'Fuel_Type': [fuel], 'Make_Model': [make_mod]
    })
    df['Location'] = df['Location'].astype('category')
    df['Fuel_Type'] = df['Fuel_Type'].astype('category')
    df['Make_Model'] = df['Make_Model'].astype('category')
    return df


# MAIN DASHBOARD

st.markdown('<div class="main-header">Enterprise Auto Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Vehicle Valuations & Market Analytics</div>', unsafe_allow_html=True)

if predict_button:
    combined_make_model = f"{selected_make} {selected_model}"
    input_df = prepare_input(selected_location, mileage, engine, year, selected_fuel, combined_make_model)
    
    # Predict Price
    prediction = model.predict(input_df)[0]
    
  
    st.markdown(f"""
    <div class="price-card">
        <div class="price-label">Real-Time Algorithmic Valuation</div>
        <div class="price-value">Rs {prediction:,.0f}</div>
        <div class="price-desc">{year} {selected_make} {selected_model} &nbsp;•&nbsp; {engine}cc &nbsp;•&nbsp; {mileage:,} km</div>
    </div>
    """, unsafe_allow_html=True)
    
  
    col1, col2 = st.columns([1, 1.4], gap="large") 
    
    with col1:
        st.markdown('<div class="section-title">Vehicle Identity</div>', unsafe_allow_html=True)
       
        st.markdown(f"""
        <div class="info-card">
            <div class="info-item">
                <span class="info-label">Make</span>
                <span class="info-value">{selected_make}</span>
            </div>
            <div class="info-item">
                <span class="info-label">Model</span>
                <span class="info-value">{selected_model}</span>
            </div>
            <div class="info-item">
                <span class="info-label">Manufacture Year</span>
                <span class="info-value">{year}</span>
            </div>
            <div class="info-item">
                <span class="info-label">Odometer (Mileage)</span>
                <span class="info-value">{mileage:,} km</span>
            </div>
            <div class="info-item">
                <span class="info-label">Engine Capacity</span>
                <span class="info-value">{engine} cc</span>
            </div>
            <div class="info-item">
                <span class="info-label">Powertrain</span>
                <span class="info-value">{selected_fuel}</span>
            </div>
            <div class="info-item">
                <span class="info-label">Registration Geo</span>
                <span class="info-value">{selected_location}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">Valuation Drivers (SHAP)</div>', unsafe_allow_html=True)
     
    
        try:
          
            shap_obj = explainer(input_df)
            
           
            fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
            
           
            shap.plots.waterfall(shap_obj[0], show=False)
            
            
            plt.title("How Specific Technical Features Shifted The Baseline Price", fontsize=11, color="#2C3E50", pad=20, weight='bold')
            plt.tight_layout()
            
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Could not generate SHAP explanation: {e}")
            
        st.markdown('</div>', unsafe_allow_html=True)

    
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Market Analytics & Trends</div>', unsafe_allow_html=True)
    
    col3, col4 = st.columns(2, gap="large")
    
    with col3:
        
        st.markdown('<h4 style="color:#A0B2C6; font-size: 1.1rem; margin-top:0; margin-bottom:15px; font-weight:600;"><span style="color:#00FFD1;">📉</span> Projected Depreciation Curve</h4>', unsafe_allow_html=True)
        
        years_to_plot = list(range(2010, 2026))
        predicted_prices = []
        for y in years_to_plot:
            temp_df = prepare_input(selected_location, mileage, engine, y, selected_fuel, combined_make_model)
            predicted_prices.append(model.predict(temp_df)[0])
            
        chart_data1 = pd.DataFrame({'Year': years_to_plot, 'Estimated Value (LKR)': predicted_prices})
        
        fig1 = px.line(chart_data1, x='Year', y='Estimated Value (LKR)', markers=True)
        fig1.update_traces(line_color='#00FFD1', marker=dict(size=8, color='#00FFD1'))
        fig1.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#A0B2C6"),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="Manufacture Year"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="Price (LKR)"),
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        
        st.markdown('<h4 style="color:#A0B2C6; font-size: 1.1rem; margin-top:0; margin-bottom:15px; font-weight:600;"><span style="color:#00FFD1;">🛣️</span> Mileage Value Impact</h4>', unsafe_allow_html=True)
        
        mileage_range = list(range(0, 200000, 10000))
        mileage_prices = []
        for m in mileage_range:
            temp_df = prepare_input(selected_location, m, engine, year, selected_fuel, combined_make_model)
            mileage_prices.append(model.predict(temp_df)[0])
            
        chart_data2 = pd.DataFrame({'Mileage (km)': mileage_range, 'Estimated Value (LKR)': mileage_prices})
        
        fig2 = px.area(chart_data2, x='Mileage (km)', y='Estimated Value (LKR)')
        fig2.update_traces(line_color='#00FFD1', fillcolor='rgba(0, 255, 209, 0.15)')
        fig2.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#A0B2C6"),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="Mileage (km)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="Price (LKR)"),
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

   
    st.markdown('<br>', unsafe_allow_html=True)
    
    col5, col6 = st.columns(2, gap="large")
    
    with col5:
        
        st.markdown('<h4 style="color:#A0B2C6; font-size: 1.1rem; margin-top:0; margin-bottom:15px; font-weight:600;"><span style="color:#00FFD1;">📊</span> Similar Vehicles in Market</h4>', unsafe_allow_html=True)
      
        similar_cars = raw_df[raw_df['Make_Model'] == combined_make_model]
        
        if len(similar_cars) > 1:
            fig3 = px.histogram(similar_cars, x='Price', nbins=15)
            fig3.update_traces(marker_color='rgba(0, 255, 209, 0.6)', marker_line_color='#00FFD1', marker_line_width=1.5)
            
           
            fig3.add_vline(x=prediction, line_width=3, line_dash="dash", line_color="#E0E6ED", 
                           annotation_text="Your Valuation", annotation_position="top", annotation_font_color="white")
            
            fig3.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#A0B2C6"),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="Listing Price (LKR)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="Number of Vehicles"),
                margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Not enough market listings for this specific model to generate a distribution.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col6:
        
        st.markdown('<h4 style="color:#A0B2C6; font-size: 1.1rem; margin-top:0; margin-bottom:15px; font-weight:600;"><span style="color:#00FFD1;">📍</span> Price Variation by Major Cities</h4>', unsafe_allow_html=True)
        
       
        top_locs = raw_df['Location'].value_counts().head(8).index.tolist()
        loc_prices = []
        for loc in top_locs:
            temp_df = prepare_input(loc, mileage, engine, year, selected_fuel, combined_make_model)
            loc_prices.append(model.predict(temp_df)[0])
            
        chart_data4 = pd.DataFrame({'Location': top_locs, 'Estimated Value (LKR)': loc_prices})
        chart_data4 = chart_data4.sort_values('Estimated Value (LKR)', ascending=True)

        fig4 = px.bar(chart_data4, x='Estimated Value (LKR)', y='Location', orientation='h')
        fig4.update_traces(marker_color='rgba(0, 255, 209, 0.8)', hovertemplate="Location: %{y}<br>Price: Rs %{x:,.0f}<extra></extra>")
        fig4.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#A0B2C6"),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="Price (LKR)"),
            yaxis=dict(showgrid=False, title=""),
            margin=dict(l=10, r=10, t=10, b=10)
        )
        
        if selected_location in top_locs:
            colors = ['#00FFD1' if loc == selected_location else 'rgba(0, 255, 209, 0.3)' for loc in chart_data4['Location']]
            fig4.update_traces(marker_color=colors)
            
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    
    st.markdown("""
    <div style="background-color: #0F2027; background-image: linear-gradient(180deg, #0F2027 0%, #203A43 100%); border: 1px dashed rgba(255,255,255,0.2); border-radius: 12px; padding: 60px 20px; text-align: center; margin-top: 40px; box-shadow: 0 8px 20px rgba(0,0,0,0.2);">
        <h3 style="color: #00FFD1; font-weight: 600;">Awaiting Inputs</h3>
        <p style="color: #A0B2C6; font-size: 1.1rem; max-width: 500px; margin: 10px auto;">
            Adjust the vehicle specifications in the left panel and calculate the algorithmic market value instantly.
        </p>
    </div>
    """, unsafe_allow_html=True)
