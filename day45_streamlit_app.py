# day45_streamlit_app.py
import streamlit as st
import pandas as pd
import joblib

# ----------------------
# 1. Load trained model
# ----------------------
model = joblib.load("optimized_traffic_model_day43.pkl")

# ----------------------
# 2. App Title
# ----------------------
st.set_page_config(page_title="Traffic Speed Prediction", layout="centered")
st.title("🚦 Traffic Speed Prediction")
st.write("Enter the traffic details below to predict speed.")

# ----------------------
# 3. User Inputs
# ----------------------
SPEED = st.number_input("Current Speed (km/h)", min_value=0.0, max_value=200.0, value=40.0)
BUS_COUNT = st.number_input("Bus Count", min_value=0, max_value=500, value=10)
NUM_READS = st.number_input("Number of Reads", min_value=0, max_value=10000, value=500)
HOUR = st.slider("Hour of Day", 0, 23, 8)
REGION = st.selectbox("Region", ["North", "South", "East", "West"])
DAY_OF_WEEK = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
MONTH = st.selectbox("Month", ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])

rush_hour = 1 if HOUR in [7, 8, 9, 17, 18, 19] else 0

# ----------------------
# 4. Prediction Button
# ----------------------
if st.button("Predict Speed"):
    # Map categorical values to numeric
    day_mapping = {"Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4, 
                   "Friday": 5, "Saturday": 6, "Sunday": 7}
    
    month_mapping = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
                     "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
    
    region_mapping = {"North": 1, "South": 2, "East": 3, "West": 4}
    
    # The ColumnTransformer expects specific data types:
    # - Numeric columns: float64 (for scaling)
    # - Categorical columns: object/string (for encoding)
    
    # Categorical columns that need to be strings
    categorical_cols = ['TIME', 'REGION', 'DESCRIPTION', 'RECORD_ID', 'NW_LOCATION', 'SE_LOCATION', 'DAY']
    # Numeric columns that should stay as float
    numeric_cols = ['REGION_ID', 'BUS_COUNT', 'NUM_READS', 'HOUR', 'DAY_OF_WEEK', 'MONTH', 'WEST', 'EAST', 'SOUTH', 'NORTH']
    
    # Create data with proper types for each column
    input_data = {
        "TIME": str(int(HOUR)),                               # Categorical - convert to string
        "REGION_ID": float(region_mapping.get(REGION, 1)),   # Numeric
        "REGION": str(region_mapping.get(REGION, 1)),        # Categorical - convert to string
        "BUS_COUNT": float(BUS_COUNT),                       # Numeric
        "NUM_READS": float(NUM_READS),                       # Numeric
        "HOUR": float(HOUR),                                 # Numeric
        "DAY_OF_WEEK": float(day_mapping.get(DAY_OF_WEEK, 1)), # Numeric
        "MONTH": float(month_mapping.get(MONTH, 1)),         # Numeric
        "DESCRIPTION": str(0),                               # Categorical - convert to string
        "RECORD_ID": str(1),                                 # Categorical - convert to string
        "WEST": float(1 if REGION == "West" else 0),        # Numeric
        "EAST": float(1 if REGION == "East" else 0),        # Numeric
        "SOUTH": float(1 if REGION == "South" else 0),      # Numeric
        "NORTH": float(1 if REGION == "North" else 0),      # Numeric
        "NW_LOCATION": str(0),                               # Categorical - convert to string
        "SE_LOCATION": str(0),                               # Categorical - convert to string
        "DAY": str(day_mapping.get(DAY_OF_WEEK, 1))         # Categorical - convert to string
    }
    
    # Create DataFrame in the exact order expected
    expected_features = ['TIME', 'REGION_ID', 'REGION', 'BUS_COUNT', 'NUM_READS', 'HOUR', 
                        'DAY_OF_WEEK', 'MONTH', 'DESCRIPTION', 'RECORD_ID', 'WEST', 'EAST', 
                        'SOUTH', 'NORTH', 'NW_LOCATION', 'SE_LOCATION', 'DAY']
    
    input_df = pd.DataFrame([input_data])[expected_features]
    
    try:
        # The model is a pipeline with ColumnTransformer, so we need to pass a DataFrame
        # Make sure we have the exact column names and pass the DataFrame directly
        
        # Debug: Show what we're sending
        st.write("Sending DataFrame to model with columns:")
        st.write(input_df.columns.tolist())
        
        # Use the DataFrame directly (not numpy array) since it's a ColumnTransformer
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Speed: {prediction:.2f} km/h")
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        
        # Try to get more info about the ColumnTransformer
        try:
            preprocessor = model.named_steps['preprocessor']
            st.write("ColumnTransformer info:")
            
            # Check if it has transformers attribute
            if hasattr(preprocessor, 'transformers'):
                st.write("Transformers:")
                for name, transformer, columns in preprocessor.transformers:
                    st.write(f"- {name}: {type(transformer)} on columns {columns}")
                    
            # Check if it has feature_names_in_
            if hasattr(preprocessor, 'feature_names_in_'):
                st.write(f"Preprocessor expected features: {preprocessor.feature_names_in_}")
                
        except Exception as debug_error:
            st.write(f"Debug error: {debug_error}")
            
        # Final fallback: show all our debug info
        st.write("Debug - Input DataFrame:")
        st.write(input_df)
        st.write(f"DataFrame dtypes: {input_df.dtypes.to_dict()}")