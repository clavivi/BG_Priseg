import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import io
import base64

st.set_page_config(page_title="Donor Segmentation Tool", layout="wide")

# Define CSS styles
st.markdown("""
<style>
    .main-header {
        font-size: 28px;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 22px;
        font-weight: bold;
        color: #2563EB;
        margin-top: 30px;
        margin-bottom: 10px;
    }
    .subsection-header {
        font-size: 18px;
        font-weight: bold;
        color: #3B82F6;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .info-box {
        background-color: #EFF6FF;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .success-box {
        background-color: #ECFDF5;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .warning-box {
        background-color: #FFFBEB;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<div class="main-header">Donor Segmentation and Mailing List Tool</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
This application helps you analyze donor data, segment donors, and prepare customized mailing lists with targeted ask amounts.
<br><br>
<b>Workflow:</b>
<ol>
    <li>Upload your donor gift records (CSV or Excel)</li>
    <li>Map your data columns to standard fields</li>
    <li>Review data quality and summary statistics</li>
    <li>Configure segmentation settings</li>
    <li>Generate and download the final segmented donor list</li>
</ol>
</div>
""", unsafe_allow_html=True)

# Standard column mapping
STANDARD_COLUMNS = {
    'donor_id': {'required': True, 'type': 'STRING', 'description': 'Unique identifier for the donor'},
    'donor_first_name': {'required': False, 'type': 'STRING', 'description': "Donor's first name"},
    'donor_last_name': {'required': False, 'type': 'STRING', 'description': "Donor's last name"},
    'donor_email': {'required': False, 'type': 'STRING', 'description': "Donor's email address"},
    'postal_code': {'required': True, 'type': 'STRING', 'description': "Donor's last known postal code"},
    'mobile_number': {'required': False, 'type': 'STRING', 'description': "Donor's mobile phone number"},
    'gift_date': {'required': True, 'type': 'DATE', 'description': 'Date the gift was made'},
    'gift_amount': {'required': True, 'type': 'NUMERIC', 'description': 'Amount of the donation'},
    'gift_currency': {'required': False, 'type': 'STRING', 'description': 'Currency of the donation (e.g., CAD, USD)'},
    'campaign_name': {'required': False, 'type': 'STRING', 'description': 'Name of the campaign the gift was attributed to'},
    'campaign_name2': {'required': False, 'type': 'STRING', 'description': 'Any other Identifier of campaign and attribution'},
    'campaign_code': {'required': False, 'type': 'STRING', 'description': 'A non-verbose campaign Identifier'},
    'solicitation_code': {'required': False, 'type': 'STRING', 'description': 'Code used to track the solicitation source'},
    'payment_method': {'required': False, 'type': 'STRING', 'description': 'How the donor paid (e.g., Credit Card, EFT, Cheque)'},
    'payment_channel': {'required': False, 'type': 'STRING', 'description': 'Channel through which the donation was received'},
    'gift_type': {'required': False, 'type': 'STRING', 'description': 'Type of gift (e.g., One-Time, Monthly, Pledge)'},
    'do_not_mail_flag': {'required': False, 'type': 'BOOLEAN', 'description': 'Whether the donor opted out of physical mail'},
    'do_not_contact_flag': {'required': False, 'type': 'BOOLEAN', 'description': 'Whether the donor opted out of any contact'},
    'do_not_solicit': {'required': False, 'type': 'BOOLEAN', 'description': 'Whether the donor opted out of solicitation'},
    'do_not_email_flag': {'required': False, 'type': 'BOOLEAN', 'description': 'Whether the donor opted out of email communication'}
}

# Define RFM Scoring Functions
def recency_score(days):
    if days <= 60:
        return 5
    elif days <= 365:
        return 4
    elif days <= 540:
        return 3
    elif days <= 906:
        return 2
    elif days <= 1088:
        return 1
    else:
        return 0

def frequency_score(freq):
    if freq >= 8:
        return 5
    elif freq >= 6:
        return 4
    elif freq >= 4:
        return 3
    elif freq >= 2:
        return 2
    else:
        return 1

def monetary_score(amount):
    if amount >= 5000:
        return 5
    elif amount >= 1000:
        return 4
    elif amount >= 500:
        return 3
    elif amount >= 100:
        return 2
    else:
        return 1

# Function to generate a download link for dataframe
def get_download_link(df, filename, button_text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="btn">{button_text}</a>'
    return href

# Initialize session state
if 'data_uploaded' not in st.session_state:
    st.session_state.data_uploaded = False
if 'column_mapping_done' not in st.session_state:
    st.session_state.column_mapping_done = False
if 'preprocessing_done' not in st.session_state:
    st.session_state.preprocessing_done = False
if 'segmentation_done' not in st.session_state:
    st.session_state.segmentation_done = False
if 'donor_summary' not in st.session_state:
    st.session_state.donor_summary = None
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'column_mapping' not in st.session_state:
    st.session_state.column_mapping = {}
if 'same_day_gifts_aggregated' not in st.session_state:
    st.session_state.same_day_gifts_aggregated = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1

# Create tabs for the workflow
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Upload Data", 
    "2. Column Mapping", 
    "3. Data Quality", 
    "4. Segmentation", 
    "5. Final Export"
])

# Step 1: Upload Data
with tab1:
    st.markdown('<div class="section-header">Upload Donor Gift Records</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            # Determine file type and read accordingly
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Display raw data preview
            st.markdown('<div class="subsection-header">Data Preview</div>', unsafe_allow_html=True)
            st.dataframe(df.head())
            
            # Display data shape
            st.markdown(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")
            
            # Check for required columns
            st.markdown('<div class="subsection-header">Column Detection</div>', unsafe_allow_html=True)
            st.markdown("<p>In the next step, you'll map your columns to our standard format.</p>", unsafe_allow_html=True)
            
            # Store the raw data in session state
            st.session_state.raw_data = df
            st.session_state.data_uploaded = True
            
            # Move to the next step
            if st.button("Continue to Column Mapping"):
                st.session_state.current_step = 2
                st.experimental_rerun()
                
        except Exception as e:
            st.error(f"Error loading file: {e}")
    else:
        st.info("Please upload a CSV or Excel file containing donor gift records.")

# Step 2: Column Mapping
with tab2:
    if st.session_state.data_uploaded:
        st.markdown('<div class="section-header">Map Your Columns to Standard Fields</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        Map each of your data columns to our standard fields. Required fields are marked with *.
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize column mapping if not already done
        if not st.session_state.column_mapping:
            # Guess mappings based on common patterns
            guessed_mapping = {}
            df_columns = st.session_state.raw_data.columns
            
            # Simple pattern matching for common column names
            for std_col, info in STANDARD_COLUMNS.items():
                matched = False
                # Check for exact match
                if std_col in df_columns:
                    guessed_mapping[std_col] = std_col
                    matched = True
                else:
                    # Check for partial matches
                    for col in df_columns:
                        col_lower = col.lower()
                        std_col_parts = std_col.split('_')
                        
                        # Match donor ID variations
                        if std_col == 'donor_id' and any(x in col_lower for x in ['id', 'no', 'num', 'client', 'donor']):
                            guessed_mapping[std_col] = col
                            matched = True
                            break
                            
                        # Match name fields
                        elif std_col == 'donor_first_name' and any(x in col_lower for x in ['first', 'prénom', 'prenom']):
                            guessed_mapping[std_col] = col
                            matched = True
                            break
                            
                        elif std_col == 'donor_last_name' and any(x in col_lower for x in ['last', 'nom', 'family']):
                            guessed_mapping[std_col] = col
                            matched = True
                            break
                            
                        # Match date fields
                        elif std_col == 'gift_date' and any(x in col_lower for x in ['date', 'dt']):
                            guessed_mapping[std_col] = col
                            matched = True
                            break
                            
                        # Match amount fields
                        elif std_col == 'gift_amount' and any(x in col_lower for x in ['amount', 'montant', 'sum', 'donation']):
                            guessed_mapping[std_col] = col
                            matched = True
                            break
                            
                        # Match postal code fields
                        elif std_col == 'postal_code' and any(x in col_lower for x in ['postal', 'code p', 'zip']):
                            guessed_mapping[std_col] = col
                            matched = True
                            break
                            
                        # Match campaign fields
                        elif std_col == 'campaign_code' and any(x in col_lower for x in ['campaign', 'code', 'act']):
                            guessed_mapping[std_col] = col
                            matched = True
                            break
                
                # If no match found for required field, set to empty
                if not matched and info['required']:
                    guessed_mapping[std_col] = ""
            
            st.session_state.column_mapping = guessed_mapping
        
        # Create form for column mapping
        with st.form("column_mapping_form"):
            col1, col2 = st.columns(2)
            
            # Display dropdown for each standard column
            for i, (std_col, info) in enumerate(STANDARD_COLUMNS.items()):
                # Alternate between columns
                current_col = col1 if i % 2 == 0 else col2
                
                # Create a label with required indicator
                label = f"{std_col} *" if info['required'] else std_col
                
                # Get current mapping if exists
                current_value = st.session_state.column_mapping.get(std_col, "")
                
                # Create dropdown with None option and all available columns
                options = [""] + list(st.session_state.raw_data.columns)
                
                # Select box for column mapping
                selected = current_col.selectbox(
                    label,
                    options=options,
                    index=options.index(current_value) if current_value in options else 0,
                    help=info['description'],
                    key=f"mapping_{std_col}"
                )
                
                # Update mapping
                st.session_state.column_mapping[std_col] = selected
            
            # Submit button
            submit_button = st.form_submit_button("Save Column Mapping")
            
            if submit_button:
                # Validate required fields
                missing_required = []
                for std_col, info in STANDARD_COLUMNS.items():
                    if info['required'] and not st.session_state.column_mapping.get(std_col):
                        missing_required.append(std_col)
                
                if missing_required:
                    st.error(f"Please map the following required fields: {', '.join(missing_required)}")
                else:
                    st.session_state.column_mapping_done = True
                    st.success("Column mapping completed! Continue to data preprocessing.")
                    
                    # Store the mapping for future reference
                    mapping_df = pd.DataFrame([
                        {"Standard Field": k, "Your Column": v, "Required": STANDARD_COLUMNS[k]['required']}
                        for k, v in st.session_state.column_mapping.items()
                        if v  # Only include mapped fields
                    ])
                    st.session_state.mapping_df = mapping_df
                    
                    # Move to the next step
                    st.session_state.current_step = 3
                    st.experimental_rerun()
    else:
        st.warning("Please upload data in the previous step first.")
        if st.button("Go to Upload Data"):
            st.session_state.current_step = 1
            st.experimental_rerun()

# Step 3: Data Quality and Preprocessing
with tab3:
    if st.session_state.column_mapping_done:
        st.markdown('<div class="section-header">Data Quality and Preprocessing</div>', unsafe_allow_html=True)
        
        # Get the mapping
        mapping = st.session_state.column_mapping
        
        # Show mapping summary
        st.markdown('<div class="subsection-header">Column Mapping Summary</div>', unsafe_allow_html=True)
        st.dataframe(st.session_state.mapping_df)
        
        # Process the data
        if st.button("Process Data") or st.session_state.preprocessing_done:
            with st.spinner("Processing data..."):
                try:
                    # Get the raw data
                    df = st.session_state.raw_data.copy()
                    
                    # Create a new dataframe with standard column names
                    processed_df = pd.DataFrame()
                    
                    # Apply column mapping
                    for std_col, source_col in mapping.items():
                        if source_col:  # Only process mapped columns
                            processed_df[std_col] = df[source_col]
                    
                    # Ensure donor_id is string
                    processed_df['donor_id'] = processed_df['donor_id'].astype(str).str.strip()
                    
                    # Convert gift_date to datetime
                    try:
                        processed_df['gift_date'] = pd.to_datetime(processed_df['gift_date'])
                    except Exception as e:
                        st.warning(f"Error converting gift_date to datetime. Using original format. Error: {e}")
                    
                    # Clean gift amount
                    if 'gift_amount' in processed_df.columns:
                        # Handle currency symbols and commas
                        if processed_df['gift_amount'].dtype == 'object':
                            processed_df['gift_amount'] = processed_df['gift_amount'].astype(str)
                            processed_df['gift_amount'] = processed_df['gift_amount'].str.replace('$', '', regex=False)
                            processed_df['gift_amount'] = processed_df['gift_amount'].str.replace(' ', '', regex=False)
                            processed_df['gift_amount'] = processed_df['gift_amount'].str.replace(',', '.', regex=False)
                        processed_df['gift_amount'] = pd.to_numeric(processed_df['gift_amount'], errors='coerce')
                    
                    # Store the processed data
                    st.session_state.processed_data = processed_df
                    
                    # Calculate current date for Recency
                    current_date = datetime.now()
                    
                    # Aggregate gifts from the same donor on the same day
                    if 'gift_date' in processed_df.columns and 'donor_id' in processed_df.columns:
                        same_day_agg = processed_df.groupby(['donor_id', pd.Grouper(key='gift_date', freq='D')]).agg({
                            'gift_amount': 'sum'
                        }).reset_index()
                        
                        # Store the same-day aggregated data
                        st.session_state.same_day_gifts_aggregated = same_day_agg
                        
                        # Create donor level summary
                        donor_summary = same_day_agg.groupby('donor_id').agg({
                            'gift_date': [
                                ('Recency', lambda x: (current_date - x.max()).days),
                                ('Last_Gift_Date', 'max')
                            ],
                            'gift_amount': [
                                ('Frequency', 'count'),
                                ('Monetary', 'sum'),
                                ('Average_Donation', 'mean'),
                                ('Min_Gift_Amount', 'min'),
                                ('Max_Gift_Amount', 'max')
                            ]
                        })
                        
                        # Flatten column names
                        donor_summary.columns = ['Recency', 'Last_Gift_Date', 'Frequency', 'Monetary', 'Average_Donation', 'Min_Gift_Amount', 'Max_Gift_Amount']
                        
                        # Reset index to make donor_id a column
                        donor_summary = donor_summary.reset_index()
                        
                        # Calculate RFM scores
                        donor_summary['R_Score'] = donor_summary['Recency'].apply(recency_score)
                        donor_summary['F_Score'] = donor_summary['Frequency'].apply(frequency_score)
                        donor_summary['M_Score'] = donor_summary['Monetary'].apply(monetary_score)
                        donor_summary['RFM_Sum'] = donor_summary['R_Score'] + donor_summary['F_Score'] + donor_summary['M_Score']
                        donor_summary['Engagement_Score'] = donor_summary['R_Score'] + donor_summary['F_Score']
                        
                        # Add campaign flag if campaign_code exists
                        if 'campaign_code' in processed_df.columns:
                            # Get unique donor IDs who donated to specific campaigns
                            if 'campaign_name' in processed_df.columns:
                                campaign_names = processed_df['campaign_name'].dropna().unique()
                                
                                for campaign in campaign_names:
                                    campaign_donors = set(processed_df[processed_df['campaign_name'] == campaign]['donor_id'].unique())
                                    donor_summary[f'{campaign}_Flag'] = donor_summary['donor_id'].apply(lambda x: 1 if x in campaign_donors else 0)
                            
                            # Add year analysis
                            current_year = datetime.now().year
                            for year in range(current_year-2, current_year+1):
                                # Create year filters
                                year_start = pd.Timestamp(f"{year}-01-01")
                                year_end = pd.Timestamp(f"{year}-12-31")
                                
                                # Filter gifts by year
                                year_gifts = same_day_agg[(same_day_agg['gift_date'] >= year_start) & 
                                                         (same_day_agg['gift_date'] <= year_end)]
                                
                                # Group by donor for the year
                                year_summary = year_gifts.groupby('donor_id').agg({
                                    'gift_amount': ['count', 'sum']
                                })
                                
                                # Flatten columns
                                year_summary.columns = [f'F{year}', f'M{year}']
                                
                                # Merge with donor summary
                                donor_summary = donor_summary.merge(year_summary, on='donor_id', how='left')
                            
                            # Fill NAs with 0 for frequency and monetary year columns
                            for year in range(current_year-2, current_year+1):
                                donor_summary[f'F{year}'] = donor_summary[f'F{year}'].fillna(0)
                                donor_summary[f'M{year}'] = donor_summary[f'M{year}'].fillna(0)
                        
                        # Add monthly donor flag based on frequency pattern
                        # A donor is considered monthly if they have at least 3 months with donations in the last year
                        recent_gifts = processed_df[processed_df['gift_date'] >= (current_date - pd.Timedelta(days=365))]
                        monthly_pattern = recent_gifts.groupby(['donor_id', pd.Grouper(key='gift_date', freq='M')]).size().reset_index(name='gifts_per_month')
                        monthly_counts = monthly_pattern.groupby('donor_id')['gifts_per_month'].agg(lambda x: sum(x > 0)).reset_index(name='months_with_gifts')
                        monthly_donors = set(monthly_counts[monthly_counts['months_with_gifts'] >= 3]['donor_id'])
                        
                        donor_summary['Monthly_Donor_Flag'] = donor_summary['donor_id'].apply(lambda x: 1 if x in monthly_donors else 0)
                        
                        # Add contact information
                        # Get the latest contact info for each donor
                        contact_cols = ['donor_id']
                        for col in ['donor_first_name', 'donor_last_name', 'donor_email', 'postal_code', 'mobile_number']:
                            if col in processed_df.columns:
                                contact_cols.append(col)
                        
                        if len(contact_cols) > 1:  # If we have contact info beyond just donor_id
                            contact_info = processed_df[contact_cols].drop_duplicates(subset=['donor_id'], keep='last')
                            donor_summary = donor_summary.merge(contact_info, on='donor_id', how='left')
                        
                        # Add do not contact flags
                        for flag in ['do_not_mail_flag', 'do_not_contact_flag', 'do_not_solicit', 'do_not_email_flag']:
                            if flag in processed_df.columns:
                                flag_info = processed_df[['donor_id', flag]].drop_duplicates(subset=['donor_id'], keep='last')
                                donor_summary = donor_summary.merge(flag_info, on='donor_id', how='left')
                                donor_summary[flag] = donor_summary[flag].fillna(False)
                        
                        # Store the donor summary
                        st.session_state.donor_summary = donor_summary
                        st.session_state.preprocessing_done = True
                        
                    else:
                        st.error("Missing required columns for aggregation. Please check your column mapping.")
                
                except Exception as e:
                    st.error(f"Error during data processing: {e}")
                    import traceback
                    st.error(traceback.format_exc())
            
            # Display data quality insights
            if st.session_state.preprocessing_done:
                st.markdown('<div class="success-box">Data processing completed successfully!</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="subsection-header">Data Quality Report</div>', unsafe_allow_html=True)
                
                # Create tabs for different data quality aspects
                dq_tab1, dq_tab2, dq_tab3 = st.tabs(["Summary Statistics", "Missing Values", "Distribution Analysis"])
                
                with dq_tab1:
                    # Summary statistics
                    donor_summary = st.session_state.donor_summary
                    st.markdown("##### Donor Summary Statistics")
                    
                    # Basic counts
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Donors", f"{len(donor_summary):,}")
                    
                    # Monetary statistics
                    col2.metric("Total Donations", f"${donor_summary['Monetary'].sum():,.2f}")
                    col3.metric("Average Gift Size", f"${donor_summary['Average_Donation'].mean():,.2f}")
                    col4.metric("Median Gift Size", f"${donor_summary['Average_Donation'].median():,.2f}")
                    
                    # RFM statistics
                    st.markdown("##### RFM Score Distribution")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    col1.markdown(f"**Recency (R) Scores**")
                    col1.dataframe(donor_summary['R_Score'].value_counts().sort_index().reset_index().rename(columns={'index': 'Score', 'R_Score': 'Count'}))
                    
                    col2.markdown(f"**Frequency (F) Scores**")
                    col2.dataframe(donor_summary['F_Score'].value_counts().sort_index().reset_index().rename(columns={'index': 'Score', 'F_Score': 'Count'}))
                    
                    col3.markdown(f"**Monetary (M) Scores**")
                    col3.dataframe(donor_summary['M_Score'].value_counts().sort_index().reset_index().rename(columns={'index': 'Score', 'M_Score': 'Count'}))
                    
                    col4.markdown(f"**RFM Sum Distribution**")
                    col4.dataframe(donor_summary['RFM_Sum'].value_counts().sort_index().reset_index().rename(columns={'index': 'Score', 'RFM_Sum': 'Count'}))
                    
                    # Monthly donors
                    st.markdown("##### Monthly Donor Analysis")
                    col1, col2 = st.columns(2)
                    monthly_count = donor_summary['Monthly_Donor_Flag'].sum()
                    col1.metric("Monthly Donors", f"{monthly_count:,}")
                    col2.metric("Percentage of Donors", f"{(monthly_count / len(donor_summary) * 100):.1f}%")
                
                with dq_tab2:
                    # Missing values analysis
                    st.markdown("##### Missing Values Analysis")
                    
                    processed_df = st.session_state.processed_data
                    missing_data = pd.DataFrame({
                        'Column': processed_df.columns,
                        'Missing Values': processed_df.isnull().sum().values,
                        'Percentage': (processed_df.isnull().sum().values / len(processed_df) * 100)
                    }).sort_values('Missing Values', ascending=False)
                    
                    st.dataframe(missing_data)
                    
                    # Data type issues
                    st.markdown("##### Data Type Issues")
                    
                    # Check for numeric fields with non-numeric values
                    if 'gift_amount' in processed_df.columns:
                        non_numeric = processed_df['gift_amount'].isna().sum()
                        if non_numeric > 0:
                            st.warning(f"Found {non_numeric:,} rows with non-numeric gift amounts.")
                    
                    # Check for date fields with invalid dates
                    if 'gift_date' in processed_df.columns:
                        invalid_dates = processed_df['gift_date'].isna().sum()
                        if invalid_dates > 0:
                            st.warning(f"Found {invalid_dates:,} rows with invalid gift dates.")
                    
                    # Quality checks on donor_id
                    if 'donor_id' in processed_df.columns:
                        duplicate_ids = processed_df['donor_id'].duplicated().sum()
                        if duplicate_ids > 0:
                            st.info(f"Found {duplicate_ids:,} duplicate donor_id entries (normal for multiple gifts).")
                
                with dq_tab3:
                    # Distribution analysis
                    st.markdown("##### Gift Amount Distribution")
                    
                    # Create a histogram of gift amounts
                    fig = px.histogram(
                        st.session_state.processed_data, 
                        x='gift_amount',
                        nbins=50,
                        title='Distribution of Gift Amounts',
                        labels={'gift_amount': 'Gift Amount'},
                        log_y=True  # Use log scale for better visibility
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # RFM heatmaps
                    st.markdown("##### RFM Analysis")
                    
                    # Create RF Matrix (Recency-Frequency)
                    rf_counts = donor_summary.groupby(['R_Score', 'F_Score']).size().reset_index(name='Count')
                    rf_pivot = rf_counts.pivot(index='R_Score', columns='F_Score', values='Count').fillna(0)
                    
                    # Plot RF Matrix
                    fig = px.imshow(
                        rf_pivot,
                        labels=dict(x="Frequency Score", y="Recency Score", color="Donor Count"),
                        x=rf_pivot.columns,
                        y=rf_pivot.index,
                        color_continuous_scale='Blues',
                        title='Recency-Frequency Matrix',
                        text_auto=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create RM Matrix (Recency-Monetary)
                    rm_counts = donor_summary.groupby(['R_Score', 'M_Score']).size().reset_index(name='Count')
                    rm_pivot = rm_counts.pivot(index='R_Score', columns='M_Score', values='Count').fillna(0)
                    
                    # Plot RM Matrix
                    fig = px.imshow(
                        rm_pivot,
                        labels=dict(x="Monetary Score", y="Recency Score", color="Donor Count"),
                        x=rm_pivot.columns,
                        y=rm_pivot.index,
                        color_continuous_scale='Greens',
                        title='Recency-Monetary Matrix',
                        text_auto=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create a box plot of Average Donation by RFM Sum
                    fig = px.box(
                        donor_summary, 
                        x='RFM_Sum', 
                        y='Average_Donation',
                        title='Average Donation by RFM Sum',
                        labels={'RFM_Sum': 'RFM Sum', 'Average_Donation': 'Average Donation Amount'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Gift date timeline
                    if 'gift_date' in st.session_state.processed_data.columns:
                        # Create a time series of gift counts
                        time_series = st.session_state.processed_data.groupby(pd.Grouper(key='gift_date', freq='M')).size().reset_index(name='gift_count')
                        
                        # Plot time series
                        fig = px.line(
                            time_series,
                            x='gift_date',
                            y='gift_count',
                            title='Monthly Gift Count Timeline',
                            labels={'gift_date': 'Date', 'gift_count': 'Number of Gifts'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                