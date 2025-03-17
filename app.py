
!pip install streamlit
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load assets (Ensure you have these files in your project directory)
header_image = Image.open("assets/header.jpg")  # Add your own header image
footer_text = "Made with 💜🤍"

# File paths for models and dataset
MODEL_PATH = "models/model.pkl"
MINMAX_SCALER_PATH = "models/minmaxscaler.pkl"
STANDARD_SCALER_PATH = "models/standardscaler.pkl"
DATASET_PATH = "data/updated_dataset.csv"

# Load models and scalers
@st.cache_resource
def load_models():
    model = pickle.load(open(MODEL_PATH, 'rb'))
    minmax_scaler = pickle.load(open(MINMAX_SCALER_PATH, 'rb'))
    standard_scaler = pickle.load(open(STANDARD_SCALER_PATH, 'rb'))
    return model, minmax_scaler, standard_scaler

# Load dataset for feature references
@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_PATH)

# Preprocessing function
def preprocess_input(user_input, full_dataset, minmax, standard):
    # One-hot encode season
    temp = pd.get_dummies(user_input['season'], prefix='season')
    input_df = pd.concat([user_input.drop('season', axis=1), temp], axis=1)

    # Handle missing columns
    full_cols = pd.get_dummies(full_dataset, columns=['season']).columns
    missing_cols = [col for col in full_cols if col not in input_df.columns]
    for col in missing_cols:
        input_df[col] = 0

    # Reorder columns
    input_df = input_df[full_cols]

    # Scale features
    scaled = minmax.transform(input_df)
    scaled = standard.transform(scaled)

    return scaled

# Prediction function
def predict_crop(model, features):
    return model.predict(features.reshape(1, -1))[0]

# Crop mapping (update according to your model's output)
crop_mapping = {
    0: 'Rice', 1: 'Maize', 2: 'Jute', 3: 'Cotton', 4: 'Coconut',
    5: 'Papaya', 6: 'Orange', 7: 'Apple', 8: 'Muskmelon',
    9: 'Watermelon', 10: 'Grapes', 11: 'Mango', 12: 'Banana',
    13: 'Pomegranate', 14: 'Lentil', 15: 'Blackgram',
    16: 'Mungbean', 17: 'Mothbeans', 18: 'Pigeonpeas',
    19: 'Kidneybeans', 20: 'Chickpea', 21: 'Coffee'
}

# Main app
def main():
    # Header section
    st.image(header_image, use_column_width=True)
    st.title("🌾 Crop Recommendation System")
    #st.markdown("
    #Welcome to the **Crop Recommendation System**! This tool predicts the best crop to grow based on agricultural factors such as soil nutrients, climate, and season.
    #Use the sliders below to input values and get recommendations dynamically.")

    # Load components
    model, minmax, standard = load_models()
    full_data = load_dataset()

    # Input container
    with st.container():
        st.header("Inputs")
        col1, col2 = st.columns(2)

        with col1:
            N = st.slider("Nitrogen (N)", 0, 150, 50)
            P = st.slider("Phosphorus (P)", 0, 150, 50)
            K = st.slider("Potassium (K)", 0, 200, 50)
            temp = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)
            humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0)

        with col2:
            ph = st.slider("pH Level", 0.0, 14.0, 7.0)
            rainfall = st.slider("Rainfall (mm)", 0.0, 300.0, 100.0)
            season = st.selectbox("Season", ["Kharif", "Rabi", "Zaid"])

        user_input = pd.DataFrame({
            'N': [N],
            'P': [P],
            'K': [K],
            'temperature': [temp],
            'humidity': [humidity],
            'ph': [ph],
            'rainfall': [rainfall],
            'season': [season]
        })

    # Processing and prediction
    processed = preprocess_input(user_input, full_data, minmax, standard)
    prediction = predict_crop(model, processed)
    predicted_crop = crop_mapping[prediction]

    # Results display
    st.markdown("---")
    st.subheader(f"🌱 Predicted Crop: **{predicted_crop}**")

    # Feature importance visualization
    st.markdown("### Feature Importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    features = model.feature_importances_
    feature_names = pd.get_dummies(full_data, columns=['season']).columns
    sns.barplot(x=features, y=feature_names, palette='viridis', ax=ax)
    ax.set_title("Feature Importance", fontsize=14)
    ax.set_xlabel("Importance Score")
    st.pyplot(fig)

    # Input vs dataset distribution
    st.markdown("### Input Distribution Comparison")
    input_df = user_input.drop('season', axis=1)
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.kdeplot(full_data[input_df.columns], fill=True, alpha=0.3, ax=ax2)
    sns.kdeplot(input_df, fill=True, color='red', ax=ax2)
    ax2.set_title("Input Values vs Dataset Distribution")
    st.pyplot(fig2)

    # Footer
    st.markdown("---")
    st.markdown(f"<div style='text-align: center; font-size: 12px;'>{footer_text}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
