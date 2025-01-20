import pandas as pd
import io

def load_data(uploaded_file):
    """
    Load data from a Streamlit UploadedFile object.
    """
    # Check if the uploaded file is a CSV
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    # Check if the uploaded file is an Excel file
    elif uploaded_file.name.endswith('.xlsx'):
        data = pd.read_excel(uploaded_file, engine='openpyxl')
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")

    return data