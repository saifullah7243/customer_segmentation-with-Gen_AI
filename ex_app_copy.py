from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle
import streamlit.components.v1 as components

import plotly.express as px
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# Load the saved model, scaler, label encoder, and historical data
with open("random_forest_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open("label_encoder.pkl", "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Load historical customer data (for clustering visualization)
historical_data = pd.read_csv("final_customer_data_V1.csv")

# Preprocess historical data for visualization
features = historical_data[['total_spent', 'n_transaction', 'day_since_last_visit', 'median_days']]
scaled_features = scaler.transform(features)

# Reduce data to 2D using PCA for visualization
pca = PCA(n_components=2)
historical_data_2d = pca.fit_transform(scaled_features)
historical_data['pca_x'] = historical_data_2d[:, 0]
historical_data['pca_y'] = historical_data_2d[:, 1]
historical_data['segment_code'] = label_encoder.transform(historical_data['segment'])

# Streamlit UI
st.title("Customer Segmentation Prediction and Visualization")
st.markdown(
    """
    This application predicts the customer segment based on transaction data and visualizes 
    the clusters with the new input highlighted.
    """
)

# Organizing input fields in columns
col1, col2 = st.columns(2)
with col1:
    total_spent = st.number_input("Total Spent ($)", min_value=0.0, value=0.0, step=0.1)
    day_since_last_visit = st.number_input("Days Since Last Visit", min_value=0, value=0, step=1)

with col2:
    n_transaction = st.number_input("Number of Transactions", min_value=0, value=0, step=1)
    median_days = st.number_input("Median Days Between Visits", min_value=0.0, value=0.0, step=0.1)

# Predict button
if st.button("Predict Segment"):
    if total_spent == 0 and n_transaction == 0 and day_since_last_visit == 0 and median_days == 0:
        st.warning("Please enter valid values for prediction.")
    else:
        # Scale the input features
        input_features = np.array([[total_spent, n_transaction, day_since_last_visit, median_days]])
        scaled_features = scaler.transform(input_features)
        
        # Predict the class
        prediction = model.predict(scaled_features)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        st.success(f"Predicted Segment: **{predicted_label}** 🎉")
        
        # Project new input to 2D PCA space
        input_2d = pca.transform(scaled_features)
        
        # Create a DataFrame for the new input
        new_data = pd.DataFrame({
            'pca_x': [input_2d[0, 0]],
            'pca_y': [input_2d[0, 1]],
            'segment': [predicted_label]
        })
        
        # Combine with historical data
        combined_data = pd.concat([historical_data, new_data], ignore_index=True)

        # Plot clusters using Plotly (Pie chart)
        segment_counts = combined_data['segment'].value_counts()
        
        # Define the color based on predicted segment
        colors = {segment: 'blue' for segment in segment_counts.index}
        colors[predicted_label] = 'red'  # Highlight the predicted segment
        
        # Create the pie chart with updated colors
        fig = px.pie(
            names=segment_counts.index,
            values=segment_counts.values,
            title="Customer Segmentation Distribution",
            color=segment_counts.index,
            color_discrete_map=colors
        )
        
        # Highlight the new input segment by adding an annotation
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text="New Customer",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
            font=dict(size=15, color="red"),
            align="center"
        )

        # Display the pie chart
        st.plotly_chart(fig, use_container_width=True)

# Divider for navigation
st.markdown("---")

# Navigation Links
st.markdown(
    """
    ### Navigate to Further Analysis:
    - [Customer Analysis](https://appapp-hm6yjdrpj7sahnumqfmjif.streamlit.app/)  
    - [Business Analysis](https://appapp-hm6yjdrpj7sahnumqfmjif.streamlit.app/)
    """,
    unsafe_allow_html=True,
)

# Divider for technologies
st.markdown("---")
st.markdown("### Technologies Used")

# Technologies in horizontal box format
technologies = [
    "Pandas", "Numpy", "Matplotlib", "Plotly", 
    "Scikit Learn", "Standard Scaler", 
    "Label Encoder", "One Hot Encoding", "Streamlit", "Markdown"
]

# Custom CSS to style the horizontal boxes
st.markdown(
    """
    <style>
    .tech-box {
        display: inline-block;
        padding: 10px 20px;
        margin: 5px;
        border-radius: 8px;
        background-color: #f3f3f3;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        font-size: 16px;
        font-weight: bold;
        color: #333;
        text-align: center;
        white-space: nowrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display technologies in horizontal boxes
tech_html = ''.join([f'<div class="tech-box">{tech}</div>' for tech in technologies])
st.markdown(tech_html, unsafe_allow_html=True)
