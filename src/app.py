# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve
# from data_ingestion import load_data
# from data_cleaning import clean_data
# from model import train_model, calculate_metrics

# # Title
# st.title("End-to-End Data Pipeline with AUC-ROC and Metrics Visualization")

# # File Upload
# uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel):", type=['csv', 'xlsx'])

# # Initialize variables
# if 'model_data' not in st.session_state:
#     st.session_state['model_data'] = None

# if uploaded_file:
#     # Data Ingestion
#     raw_data = load_data(uploaded_file)

#     # Display Raw Data
#     st.subheader("Raw Data")
#     st.write(raw_data)

#     # Data Cleaning
#     cleaned_data = clean_data(raw_data)

#     # Display Cleaned Data
#     st.subheader("Cleaned Data")
#     st.write(cleaned_data)

#     # Train Model
#     target_column = st.text_input("Enter the target column name:")
#     if st.button("Train Model"):
#         #st.write("DEBUG: Stored model data in session state:", st.session_state['model_data'])

#         if target_column in cleaned_data.columns:
#             # Train the model and retrieve data
#             model, auc_score, fpr, tpr, thresholds, y_true, y_pred_prob = train_model(cleaned_data, target_column)

#             # Store data in session state
#             st.session_state['model_data'] = {
#                 'auc_score': auc_score,
#                 'fpr': fpr,
#                 'tpr': tpr,
#                 'thresholds': thresholds,
#                 'y_true': y_true,  # Store ground truth
#                 'y_pred_prob': y_pred_prob,  # Store predicted probabilities
#             }
#             #st.write("DEBUG: Model data stored in session state:", st.session_state['model_data'])
#             st.success(f"Model trained successfully! AUC-ROC: {auc_score:.2f}")
#         else:
#             st.error("Target column not found in dataset.")

#     #st.write("DEBUG: Accessing y_true from session state:", st.session_state['model_data']['y_true'])

#     # AUC-ROC Visualization and Metrics
#     if st.session_state['model_data']:
#         st.subheader("AUC-ROC Curve and Metrics")
#         #st.write("DEBUG: Full session state model data:", st.session_state['model_data'])

#         # Retrieve AUC-ROC metrics and predictions
#         auc_data = st.session_state['model_data']
#         #auc_data = st.session_state['model_data']
#         # if 'y_true' not in auc_data:
#         #     st.error("ERROR: 'y_true' is missing in session state!")
#         # else:
#         #     y_true = auc_data['y_true']
#         #     y_pred_prob = auc_data['y_pred_prob']
#         #     st.write("DEBUG: Retrieved y_true from session state:", y_true[:5])
#         fpr = auc_data['fpr']
#         tpr = auc_data['tpr']
#         thresholds = auc_data['thresholds']
#         y_true = auc_data['y_true']  # Ground truth labels
#         y_pred_prob = auc_data['y_pred_prob']  # Predicted probabilities

#         # Debugging: Print session state data for troubleshooting
#         #st.write("DEBUG: Stored data in session state:", st.session_state['model_data'])

#         # Add slider for threshold
#         threshold = st.slider("Set the threshold for predictions:", 0.0, 1.0, 0.5, 0.01)

#         # Calculate metrics
#         precision, recall, accuracy, f1 = calculate_metrics(y_true, y_pred_prob, threshold)

#         # Display metrics
#         st.write(f"**Threshold:** {threshold:.2f}")
#         st.write(f"**Precision:** {precision:.2f}")
#         st.write(f"**Recall:** {recall:.2f}")
#         st.write(f"**Accuracy:** {accuracy:.2f}")
#         st.write(f"**F1-Score:** {f1:.2f}")

#         # Plot AUC-ROC Curve dynamically
#         closest_idx = (thresholds >= threshold).argmax()
#         plt.figure(figsize=(8, 6))
#         plt.plot(fpr, tpr, label=f"AUC = {auc_data['auc_score']:.2f}")
#         plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Guess")
#         plt.scatter(fpr[closest_idx], tpr[closest_idx], color='red', label=f"Threshold = {threshold:.2f}")
#         plt.xlabel("False Positive Rate")
#         plt.ylabel("True Positive Rate")
#         plt.title("AUC-ROC Curve with Dynamic Threshold")
#         plt.legend(loc="lower right")
#         st.pyplot(plt)

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