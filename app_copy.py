import os
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import pickle
import plotly.express as px
import warnings
import requests
warnings.filterwarnings("ignore")
load_dotenv()

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

# Segment descriptions and treatments
segment_descriptions = {
    "Champions": "Recent purchase, frequent transactions, high spending.",
    "Loyal Customers": "Often spend good money buying your products. Responsive to promotions.",
    "Potential Loyalist": "Recent customers but spent a good amount and bought more than once.",
    "Recent Customers": "Bought most recently, but not often.",
    "Promising": "Recent shoppers but haven’t spent much.",
    "Customers Needing Attention": "Above-average recency, frequency, and monetary values. They may not have bought very recently though.",
    "About to Sleep": "Below average recency, frequency, and monetary values. Will lose them if not reactivated.",
    "At Risk": "They spent big money and purchased often. But the last purchase was a long time ago.",
    "Can’t Lose Them": "Often made the biggest purchases but they haven’t returned for a long time.",
    "Hibernating": "The last purchase was long ago. Low spenders with a low number of orders.",
    "Lost": "Lowest recency, frequency, and monetary scores."
}

segment_treatments = {
    "Champions": "Offer exclusive rewards to keep them engaged. Reward frequent purchases with loyalty programs.",
    "Loyal Customers": "Engage with personalized promotions and discounts. Keep them excited with new product launches.",
    "Potential Loyalist": "Encourage repeat purchases with targeted email campaigns or discounts on their next purchase.",
    "Recent Customers": "Reach out with engaging offers and reminders to encourage more frequent purchases.",
    "Promising": "Provide incentives to increase spending through discounts or limited-time offers.",
    "Customers Needing Attention": "Use re-engagement campaigns, such as emails, special promotions, or loyalty incentives.",
    "About to Sleep": "Run reactivation campaigns with special offers or loyalty rewards to bring them back.",
    "At Risk": "Offer a discount or special offer to regain their attention before they become inactive.",
    "Can’t Lose Them": "Target with urgent re-engagement offers and loyalty rewards to bring them back.",
    "Hibernating": "Send re-engagement emails with promotions to reawaken their interest.",
    "Lost": "Focus on win-back campaigns, offering special deals or highly targeted promotions."
}

# Function to get Groq AI recommendations
def get_groq_recommendation(segment, action_plan):
    groq_api_url = "https://api.groq.com/openai/v1/chat/completions"
    api_key = os.getenv("GROQ_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "Gemma2-9b-it",
        "messages": [
            {"role": "system", "content": "You are an expert in customer segmentation and retention strategies."},
            {"role": "user", "content": f"The customer segment is '{segment}'. The action plan for this segment is:\n\n{action_plan}\n\nProvide additional suggestions to improve customer retention and increase revenue."}
        ]
    }
    try:
        response = requests.post(groq_api_url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        response_data = response.json()
        if "choices" in response_data and response_data["choices"]:
            return response_data["choices"][0]["message"]["content"].strip()
        else:
            return "No valid recommendation received from Groq API."
    except Exception as e:
        return f"Error while fetching Groq recommendations: {e}"

# Sidebar Navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio(
    "Select a Page",
    ("Prediction & AI Recommendations", "Further Analysis", "About")
)

if option == "Prediction & AI Recommendations":
    st.title("Customer Segmentation Prediction and Visualization")
    st.markdown("This application predicts customer segments based on input data.")

    col1, col2 = st.columns(2)
    with col1:
        total_spent = st.number_input("Total Spent ($)", min_value=0.0, value=0.0, step=0.1)
        day_since_last_visit = st.number_input("Days Since Last Visit", min_value=0, value=0, step=1)

    with col2:
        n_transaction = st.number_input("Number of Transactions", min_value=0, value=0, step=1)
        median_days = st.number_input("Median Days Between Visits", min_value=0.0, value=0.0, step=0.1)

    if st.button("Predict Segment"):
        input_features = np.array([[total_spent, n_transaction, day_since_last_visit, median_days]])
        scaled_features = scaler.transform(input_features)
        prediction = model.predict(scaled_features)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        st.success(f"Predicted Segment: **{predicted_label}** 🎉")

        description = segment_descriptions.get(predicted_label, "Description unavailable.")
        action_plan = segment_treatments.get(predicted_label, "No suggested action available.")

        st.markdown(f"**Segment Description**: {description}")
        st.markdown(f"**Suggested Action**: {action_plan}")

        with st.spinner("Generating Groq AI recommendations..."):
            ai_recommendation = get_groq_recommendation(predicted_label, action_plan)
            st.markdown(f"**Groq AI Recommendation**: {ai_recommendation}")

elif option == "Further Analysis":
    st.title("Further Analysis")
    st.markdown( """
            ### Navigate to Further Analysis:
            - [Customer Analysis](https://appapp-hm6yjdrpj7sahnumqfmjif.streamlit.app/)  
            - [Business Analysis](https://appapp-hm6yjdrpj7sahnumqfmjif.streamlit.app/)
            """)

elif option == "About":
    st.title("About This Application")
    st.markdown(
        """
         **Core Technologies:**
        - **LangChain**: For seamless integration with AI models.
        - **Python-dotenv**: For managing environment variables.
        - **Ipykernel**: Supporting Jupyter notebooks.

        **Data Processing and Visualization:**
        - **Pandas**: Data manipulation and analysis.
        - **Numpy**: Numerical computing.
        - **Matplotlib**: Basic plotting and visualization.
        - **Plotly Express**: Interactive visualizations.
        - **Seaborn**: Advanced statistical plotting.

        **Machine Learning:**
        - **Scikit-learn**: Building and evaluating machine learning models.
        - **Randforest**: For advanced Bagging techniques.

        **Other Technologies:**
        - **Openpyxl**: For working with Excel files.
        - **LangChain Community & Core**: Enhanced capabilities for working with AI models.
        - **Streamlit**: Interactive web application framework.
        """
    )
