
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from data_ingestion import load_data
from data_cleaning import clean_training_data, clean_testing_data
from model import train_model, predict, calculate_metrics
import os
from datetime import datetime, timedelta

# Set up Streamlit layout
st.set_page_config(layout="wide")

# Session state to manage toggles and configurations
if 'cron_job_active' not in st.session_state:
    st.session_state['cron_job_active'] = False

if 'last_updated' not in st.session_state:
    st.session_state['last_updated'] = datetime.now()

# Function to monitor a folder for new CSV files
def fetch_latest_file(folder_path):
    try:
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
        if not files:
            return None  # No files in the folder
        latest_file = max(files, key=os.path.getctime)
        return latest_file
    except Exception as e:
        st.error(f"Error fetching latest file: {e}")
        return None

# Tabs in Streamlit
tabs = st.tabs(["Model Training", "Testing & Continuous Inference"])

# Tab 1: Model Training
with tabs[0]:
    st.header("Model Training")
    uploaded_file = st.file_uploader("Upload training dataset (CSV or Excel):", type=['csv', 'xlsx'])
    target_column = st.text_input("Enter the target column name:")

    if uploaded_file:
        # Data ingestion
        raw_data = load_data(uploaded_file)

        # Data cleaning
        cleaned_data = clean_training_data(raw_data)

        # Display cleaned data
        st.subheader("Cleaned Data")
        st.write(cleaned_data)

        # Train Model
        if st.button("Train Model"):
            model, auc_score, fpr, tpr, thresholds, y_true, y_pred_prob = train_model(cleaned_data, target_column)
            st.session_state['model'] = model
            st.session_state['training_data'] = cleaned_data
            st.success(f"Model trained successfully! AUC-ROC: {auc_score:.2f}")

# Tab 2: Testing and Continuous Inference
with tabs[1]:
    st.header("Testing and Continuous Inference")

    # Toggle for continuous testing
    cron_job_toggle = st.checkbox("Enable Continuous Testing (Cron Jobs)", value=st.session_state['cron_job_active'])
    st.session_state['cron_job_active'] = cron_job_toggle

    # User input for folder path when cron jobs are active
    folder_path = st.text_input("Specify folder path for continuous data (if Cron Jobs enabled):")
    new_test_file = None

    # Continuous testing mode
    if cron_job_toggle and folder_path:
        st.info("Cron Jobs enabled. Monitoring folder for new data...")
        new_test_file = fetch_latest_file(folder_path)

        if new_test_file:
            st.success(f"Found new file: {os.path.basename(new_test_file)}")
            test_data = pd.read_csv(new_test_file)

            # Data cleaning for test data
            st.subheader("Cleaned Test Data")
            cleaned_test_data = clean_testing_data(test_data)
            st.write(cleaned_test_data)
            # Predict results
            model = st.session_state.get('model')
            if model:
                predictions, probabilities = predict(cleaned_test_data)
                st.write("Predictions:")
                st.write(predictions)

                # Save predictions to a CSV
                output_file = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                test_data['predictions'] = predictions
                test_data.to_csv(output_file, index=False)
                st.success(f"Predictions saved to {output_file}")
        else:
            st.warning("No new file found in the specified folder.")

    # Manual testing mode
    elif not cron_job_toggle:
        uploaded_test_file = st.file_uploader("Upload test dataset (CSV or Excel):", type=['csv', 'xlsx'])
        if uploaded_test_file:
            test_data = load_data(uploaded_test_file)
            st.write("Test Data:")
            st.write(test_data)
            # Data cleaning for test data
            st.subheader("Cleaned Test Data")
            cleaned_test_data = clean_testing_data(test_data)
            st.write(cleaned_test_data)

            # Predict results
            model = st.session_state.get('model')
            if model:
                predictions, probabilities = predict(cleaned_test_data)
                st.write("Predictions:")
                st.write(predictions)

                # Save predictions to a CSV
                output_file = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                test_data['predictions'] = predictions
                test_data.to_csv(output_file, index=False)
                st.success(f"Predictions saved to {output_file}")
            else:
                st.error("No trained model found. Please train the model first.")
