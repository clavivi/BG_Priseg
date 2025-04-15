import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import tempfile
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Donor RFM Analysis Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #1E88E5;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #4CAF50;
    }
    .warning-box {
        background-color: #FFF8E1;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #FFC107;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>Donor RFM Analysis Tool</h1>", unsafe_allow_html=True)
st.markdown("<div class='info-box'>Upload donor gift data, configure parameters, and generate a complete RFM analysis with segmentation and insights.</div>", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload & Configure", "Data Profile", "RFM Analysis", "Segmentation", "Visualizations", "Export Data"])

# Helper functions

# Include the updated preprocessing function
exec(open("preprocessor-update.py").read())

# Include the updated RFM calculation function
exec(open("rfm-update.py").read())

# Include the segmentation function
exec(open("segmentation-update.py").read())

def create_download_link(df, filename, link_text):
    """Generate a download link for a DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def generate_data_profile(df):
    """Generate a data profile for the dataframe"""
    profile_data = []
    
    # Loop through all columns
    for col in df.columns:
        col_type = df[col].dtype
        total_values = len(df)
        null_count = df[col].isna().sum()
        empty_str_count = 0
        
        # For string columns, count empty strings
        if col_type == 'object':
            empty_str_count = (df[