import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import io
import base64

st.set_page_config(page_title="Donor Segmentation Tool", layout="wide")
# Add this function to your data quality tab or in a dedicated analysis tab

def create_rf_distribution_chart(donor_data):
    """
    Creates an RF distribution chart similar to the example provided.
    Shows donor counts by RF score (R+F) and RF Sum (R+F+M)
    
    Parameters:
    -----------
    donor_data : pandas.DataFrame
        The donor summary data with R_Score, F_Score, and M_Score columns
    """
    # Make sure we have the required columns
    required_cols = ['R_Score', 'F_Score', 'M_Score']
    if not all(col in donor_data.columns for col in required_cols):
        st.error("Missing required columns for RF distribution analysis")
        return
    
    # Calculate RF Score (R+F) for each donor
    donor_data['RF_Score'] = donor_data['R_Score'] + donor_data['F_Score']
    
    # Calculate RF Sum (R+F+M) for each donor if not already calculated
    if 'RFM_Sum' not in donor_data.columns:
        donor_data['RFM_Sum'] = donor_data['R_Score'] + donor_data['F_Score'] + donor_data['M_Score']
    
    # Create a new column with labels for the x-axis that combine RF Score and their components
    # Format: "01" for R=0, F=1 or "55" for R=5, F=5
    donor_data['RF_Label'] = donor_data.apply(
        lambda x: f"{int(x['R_Score'])}{int(x['F_Score'])}", axis=1
    )
    
    # Create a combined identifier for sorting by RF Sum and then RF Score
    donor_data['sort_key'] = donor_data.apply(
        lambda x: f"{int(x['RFM_Sum']):02d}_{int(x['RF_Score']):02d}_{x['RF_Label']}", axis=1
    )
    
    # Group by RF Label and count donors
    rf_counts = donor_data.groupby(['sort_key', 'RF_Label', 'RF_Score', 'RFM_Sum']).size().reset_index(name='Count')
    
    # Sort by RF Sum and then RF Score
    rf_counts = rf_counts.sort_values('sort_key')
    
    # Create a color mapper based on RFM_Sum
    color_scale = px.colors.sequential.Viridis
    max_rfm_sum = donor_data['RFM_Sum'].max()
    min_rfm_sum = donor_data['RFM_Sum'].min()
    
    # Map RFM_Sum values to colors
    def get_color(rfm_sum):
        # Normalize to 0-1 scale
        normalized = (rfm_sum - min_rfm_sum) / (max_rfm_sum - min_rfm_sum) if max_rfm_sum > min_rfm_sum else 0.5
        # Get color index (using a subset of the color scale to avoid too light colors)
        color_idx = int(normalized * (len(color_scale) - 1))
        return color_scale[color_idx]
    
    # Add color to the dataframe
    rf_counts['Color'] = rf_counts['RFM_Sum'].apply(get_color)
    
    # Create the bar chart
    fig = go.Figure()
    
    # Add traces for each unique RFM Sum value (to get different colors)
    for rfm_sum in sorted(rf_counts['RFM_Sum'].unique()):
        subset = rf_counts[rf_counts['RFM_Sum'] == rfm_sum]
        
        fig.add_trace(go.Bar(
            x=subset['RF_Label'],
            y=subset['Count'],
            name=f'Sum {rfm_sum}',
            marker_color=get_color(rfm_sum),
            hovertemplate='<b>RF Score:</b> %{x}<br><b>Count:</b> %{y:,.0f}<br><b>RFM Sum:</b> ' + str(rfm_sum),
            text=subset['Count'],
            textposition='auto'
        ))
    
    # Customize the layout
    fig.update_layout(
        title='Distribution of RF Scores - (Ordered by RF Sum)',
        xaxis_title='RF Score (R+F)',
        yaxis_title='Number of Donors',
        barmode='group',
        legend_title='RF Sum Value',
        height=600,
        xaxis={'categoryorder': 'array', 'categoryarray': subset['RF_Label'].tolist()},
        plot_bgcolor='rgba(240, 240, 245, 0.8)',
        legend={'orientation': 'h', 'y': 1.1}
    )
    
    # Add grid lines
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.2)')
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.2)')
    
    # Make y-axis start at 0
    fig.update_yaxes(rangemode='tozero')
    
    # Add a second x-axis title explaining the labels
    fig.add_annotation(
        x=0.5,
        y=-0.15,
        xref='paper',
        yref='paper',
        text='RF Score (R+F)',
        showarrow=False,
        font=dict(size=14)
    )
    
    return fig


# Alternative version using a heatmap visualization:
def create_rf_heatmap(donor_data):
    """
    Creates an RF heatmap showing the distribution of donors across R and F scores.
    
    Parameters:
    -----------
    donor_data : pandas.DataFrame
        The donor summary data with R_Score and F_Score columns
    """
    # Make sure we have the required columns
    if 'R_Score' not in donor_data.columns or 'F_Score' not in donor_data.columns:
        st.error("Missing required columns for RF heatmap analysis")
        return
    
    # Create a pivot table of R_Score vs F_Score counts
    rf_matrix = pd.crosstab(
        donor_data['R_Score'], 
        donor_data['F_Score'],
        margins=True,
        margins_name='Total'
    )
    
    # Create the heatmap
    fig = px.imshow(
        rf_matrix,
        labels=dict(x="Frequency Score", y="Recency Score", color="Donor Count"),
        x=rf_matrix.columns,
        y=rf_matrix.index,
        color_continuous_scale="Viridis",
        text_auto=True
    )
    
    fig.update_layout(
        title='RF Heatmap - Donor Distribution',
        height=600,
        width=800
    )
    
    return fig



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
# Standard column mapping
STANDARD_COLUMNS = {
    'donor_id': {'required': True, 'type': 'STRING', 'description': 'Unique identifier for the donor'},
    'donor_first_name': {'required': False, 'type': 'STRING', 'description': "Donor's first name"},
    'donor_last_name': {'required': False, 'type': 'STRING', 'description': "Donor's last name"},
    'donor_email': {'required': False, 'type': 'STRING', 'description': "Donor's email address"},
    'donor_address': {'required': True, 'type': 'STRING', 'description': "Donor's street address"},
    'donor_address2': {'required': False, 'type': 'STRING', 'description': "Donor's address line 2 (apt, suite, etc.)"},
    'city': {'required': True, 'type': 'STRING', 'description': "Donor's city"},
    'state': {'required': True, 'type': 'STRING', 'description': "Donor's state/province"},
    'postal_code': {'required': True, 'type': 'STRING', 'description': "Donor's postal/zip code"},
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
        
        # Standard column mapping
        STANDARD_COLUMNS = {
            'donor_id': {'required': True, 'type': 'STRING', 'description': 'Unique identifier for the donor'},
            'donor_first_name': {'required': False, 'type': 'STRING', 'description': "Donor's first name"},
            'donor_last_name': {'required': False, 'type': 'STRING', 'description': "Donor's last name"},
            'donor_email': {'required': False, 'type': 'STRING', 'description': "Donor's email address"},
            'donor_address': {'required': True, 'type': 'STRING', 'description': "Donor's street address"},
            'donor_address2': {'required': False, 'type': 'STRING', 'description': "Donor's address line 2 (apt, suite, etc.)"},
            'city': {'required': True, 'type': 'STRING', 'description': "Donor's city"},
            'state': {'required': True, 'type': 'STRING', 'description': "Donor's state/province"},
            'postal_code': {'required': True, 'type': 'STRING', 'description': "Donor's postal/zip code"},
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
        
        # Initialize column mapping if not already done
        if not st.session_state.get('column_mapping'):
            # Initialize empty mapping
            st.session_state.column_mapping = {}
            
            # Get dataframe columns
            df_columns = st.session_state.raw_data.columns
            df = st.session_state.raw_data
            
            # Define column variations for common fields
            column_variations = {
                'donor_id': ['id', 'no', 'num', 'client', 'donor', 'constituent', 'supporter'],
                'donor_first_name': ['first', 'prénom', 'prenom', 'fname', 'firstname', 'name1'],
                'donor_last_name': ['last', 'nom', 'family', 'lname', 'lastname', 'name2', 'surname'],
                'donor_email': ['email', 'courriel', 'e-mail', 'emailaddress'],
                'donor_address': ['address', 'addr', 'street', 'adresse', 'address1', 'address_line_1'],
                'donor_address2': ['address2', 'addr2', 'suite', 'apt', 'address_line_2'],
                'city': ['city', 'ville', 'town', 'municipality'],
                'state': ['state', 'province', 'region', 'prov', 'état', 'etat'],
                'postal_code': ['postal', 'code p', 'zip', 'postalcode', 'codepostal', 'zipcode'],
                'mobile_number': ['mobile', 'cell', 'phone', 'telephone', 'tel', 'tél'],
                'gift_date': ['date', 'dt', 'giftdate', 'donation date', 'transaction date'],
                'gift_amount': ['amount', 'montant', 'sum', 'donation', 'gift', 'transaction'],
                'campaign_name': ['campaign', 'campagne', 'appeal'],
                'campaign_code': ['code', 'appeal code', 'campaign id', 'act', 'act code'],
                'payment_method': ['method', 'payment', 'pmt', 'pay method'],
                'payment_channel': ['channel', 'source', 'medium'],
                'gift_type': ['type', 'donation type', 'gift type', 'recurrence'],
            }
            
            # First pass: exact matches and pattern-based matching
            for std_col, info in STANDARD_COLUMNS.items():
                # Check for exact match
                if std_col in df_columns:
                    st.session_state.column_mapping[std_col] = std_col
                    continue
                    
                # Check for pattern matches
                matched = False
                if std_col in column_variations:
                    for col in df_columns:
                        col_lower = col.lower()
                        if any(var in col_lower for var in column_variations[std_col]):
                            st.session_state.column_mapping[std_col] = col
                            matched = True
                            break
                
                # If no match found but required, leave empty
                if not matched and info['required']:
                    st.session_state.column_mapping[std_col] = ""
            
            # Second pass: special cases and positional logic
            
            # Special case for address fields
            if 'donor_address' not in st.session_state.column_mapping or not st.session_state.column_mapping['donor_address']:
                for col in df_columns:
                    col_lower = col.lower()
                    if 'address' in col_lower and not any(x in col_lower for x in ['2', 'line2', 'line 2']):
                        st.session_state.column_mapping['donor_address'] = col
                        break
            
            # Special case for address2
            if 'donor_address2' not in st.session_state.column_mapping or not st.session_state.column_mapping['donor_address2']:
                for col in df_columns:
                    col_lower = col.lower()
                    if 'address' in col_lower and any(x in col_lower for x in ['2', 'line2', 'line 2']):
                        st.session_state.column_mapping['donor_address2'] = col
                        break
            
            # City often follows address
            if 'city' not in st.session_state.column_mapping or not st.session_state.column_mapping['city']:
                address_col = st.session_state.column_mapping.get('donor_address', '')
                if address_col:
                    address_index = list(df_columns).index(address_col)
                    if address_index + 1 < len(df_columns):
                        next_col = df_columns[address_index + 1]
                        if not any(x in next_col.lower() for x in ['address', 'apt', 'suite']):
                            st.session_state.column_mapping['city'] = next_col
            
            # State often follows city
            if 'state' not in st.session_state.column_mapping or not st.session_state.column_mapping['state']:
                city_col = st.session_state.column_mapping.get('city', '')
                if city_col:
                    city_index = list(df_columns).index(city_col)
                    if city_index + 1 < len(df_columns):
                        next_col = df_columns[city_index + 1]
                        if len(next_col) < 15:  # States/provinces tend to have short column names
                            st.session_state.column_mapping['state'] = next_col
            
            # Third pass: data type-based matching
            
            # Find date columns for gift_date
            if 'gift_date' not in st.session_state.column_mapping or not st.session_state.column_mapping['gift_date']:
                for col in df_columns:
                    col_lower = col.lower()
                    try:
                        if pd.api.types.is_datetime64_dtype(df[col]) or 'date' in col_lower:
                            st.session_state.column_mapping['gift_date'] = col
                            break
                    except:
                        pass
            
            # Find numeric columns for gift_amount
            if 'gift_amount' not in st.session_state.column_mapping or not st.session_state.column_mapping['gift_amount']:
                for col in df_columns:
                    try:
                        if pd.api.types.is_numeric_dtype(df[col].dtype) and 'amount' in col.lower():
                            st.session_state.column_mapping['gift_amount'] = col
                            break
                    except:
                        pass
                
                # If still not found, use first numeric column
                if 'gift_amount' not in st.session_state.column_mapping or not st.session_state.column_mapping['gift_amount']:
                    for col in df_columns:
                        try:
                            if pd.api.types.is_numeric_dtype(df[col].dtype):
                                st.session_state.column_mapping['gift_amount'] = col
                                break
                        except:
                            pass
                            
        # Show quick preview of the data
        st.markdown("### Data Preview")
        st.dataframe(st.session_state.raw_data.head(3))
        
        # Group columns by category for better organization
        column_groups = {
            "Donor Information": ["donor_id", "donor_first_name", "donor_last_name", "donor_email", "mobile_number"],
            "Address Information": ["donor_address", "donor_address2", "city", "state", "postal_code"],
            "Gift Details": ["gift_date", "gift_amount", "gift_currency", "gift_type", "payment_method", "payment_channel"],
            "Campaign Information": ["campaign_name", "campaign_name2", "campaign_code", "solicitation_code"],
            "Preferences": ["do_not_mail_flag", "do_not_contact_flag", "do_not_solicit", "do_not_email_flag"]
        }
        
        # Create the form for column mapping
        with st.form("column_mapping_form"):
            st.markdown("### Map Your Columns")
            
            # Create tabs for each column group
            column_tabs = st.tabs(list(column_groups.keys()))
            
            # Loop through each group and create mapping UI
            for i, (group_name, cols) in enumerate(column_groups.items()):
                with column_tabs[i]:
                    for std_col in cols:
                        info = STANDARD_COLUMNS[std_col]
                        
                        # Create a label with required indicator
                        label = f"{std_col} {'*' if info['required'] else ''}"
                        
                        # Get current mapping if exists
                        current_value = st.session_state.column_mapping.get(std_col, "")
                        
                        # Add helpful context for this field
                        st.markdown(f"**{label}** - {info['description']}")
                        
                        # Create dropdown with empty option and all available columns
                        options = [""] + list(st.session_state.raw_data.columns)
                        
                        # Show sample values for the selected column if one is selected
                        if current_value:
                            sample_values = st.session_state.raw_data[current_value].head(2).tolist()
                            sample_text = f"Sample values: {', '.join(str(x) for x in sample_values)}"
                        else:
                            sample_text = "No column selected"
                            
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            # Select box for column mapping
                            selected = st.selectbox(
                                f"Map to your column",
                                options=options,
                                index=options.index(current_value) if current_value in options else 0,
                                key=f"mapping_{std_col}"
                            )
                        
                        with col2:
                            st.markdown("&nbsp;")  # For vertical alignment
                            if info['required']:
                                st.markdown("**Required**")
                            else:
                                st.markdown("Optional")
                        
                        # Update mapping
                        st.session_state.column_mapping[std_col] = selected
                        
                        # Show sample values
                        st.caption(sample_text)
                        st.markdown("---")
            
            # Buttons at the bottom of the form
            col1, col2 = st.columns(2)
            
            with col1:
                advanced_options = st.checkbox("Show Advanced Options")
            
            # Advanced options for data type overrides
            if advanced_options:
                st.markdown("### Advanced Options")
                st.markdown("Override data type conversion if needed")
                
                advanced_cols = ["gift_date", "gift_amount"]
                
                for col in advanced_cols:
                    if st.session_state.column_mapping.get(col):
                        if col == "gift_date":
                            date_format = st.text_input(
                                "Date format (if needed)", 
                                value="",
                                help="e.g., %Y-%m-%d for YYYY-MM-DD format"
                            )
                            if date_format:
                                st.session_state['date_format'] = date_format
                        
                        if col == "gift_amount":
                            amount_prefix = st.text_input(
                                "Currency symbol to remove", 
                                value="$",
                                help="Symbol to remove from amount values"
                            )
                            if amount_prefix:
                                st.session_state['amount_prefix'] = amount_prefix
            
            # Submit button
            submit_button = st.form_submit_button("Save and Continue")
            
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
                    
                    # Store the mapping for future reference
                    mapping_df = pd.DataFrame([
                        {"Standard Field": k, "Your Column": v, "Required": STANDARD_COLUMNS[k]['required']}
                        for k, v in st.session_state.column_mapping.items()
                        if v  # Only include mapped fields
                    ])
                    st.session_state.mapping_df = mapping_df
                    
                    # Success message
                    st.success("Column mapping completed successfully! Continue to data preprocessing.")
                    
                    # Move to the next step
                    st.session_state.current_step = 3
                    st.rerun()
        
        # If mapping is already done, show a summary
        if st.session_state.column_mapping_done:
            st.markdown('<div class="success-box">Column mapping completed!</div>', unsafe_allow_html=True)
            
            # Show mapping summary
            st.markdown("### Column Mapping Summary")
            
            if 'mapping_df' in st.session_state:
                st.dataframe(st.session_state.mapping_df)
            
            # Button to continue
            if st.button("Continue to Data Quality"):
                st.session_state.current_step = 3
                st.rerun()
            
            # Option to edit the mapping
            if st.button("Edit Mapping"):
                st.session_state.column_mapping_done = False
                st.rerun()
    
    else:
        st.warning("Please upload data in the previous step first.")
        if st.button("Go to Upload Data"):
            st.session_state.current_step = 1
            st.rerun()

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
                
                # Add this to your data quality tab or segmentation tab to integrate the RF Distribution chart

# In your data quality tab section, find the Distribution Analysis tab and add this:

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
                    
                    # Add this code to your data quality tab (dq_tab3) section
# Replace the existing RF Distribution Analysis section with this code

                    # Add the RF Distribution Analysis
                    st.markdown("##### RF Distribution Analysis")

                    donor_summary = st.session_state.donor_summary

                    # Check if we have the R and F scores
                    if all(col in donor_summary.columns for col in ['R_Score', 'F_Score']):
                        try:
                            # Choose which visualization to show
                            viz_type = st.radio(
                                "Distribution Visualization Type:",
                                ["Simple RF Distribution", "Grouped by RFM Sum"],
                                horizontal=True
                            )
                            
                            if viz_type == "Simple RF Distribution":
                                # Show simple RF distribution
                                rf_dist_fig = create_rf_distribution_chart(donor_summary)
                                st.plotly_chart(rf_dist_fig, use_container_width=True)
                                
                                # Show the RF score breakdown
                                st.markdown("##### RF Score Breakdown")
                                
                                # Calculate RF score if not already present
                                if 'RF_Score' not in donor_summary.columns:
                                    donor_summary['RF_Score'] = donor_summary['R_Score'] + donor_summary['F_Score']
                                
                                # Get count by RF score
                                rf_counts = donor_summary['RF_Score'].value_counts().sort_index().reset_index()
                                rf_counts.columns = ['RF Score', 'Count']
                                
                                # Calculate percentage and cumulative percentage
                                total_donors = rf_counts['Count'].sum()
                                rf_counts['Percentage'] = (rf_counts['Count'] / total_donors * 100).round(1)
                                rf_counts['Cumulative %'] = (rf_counts['Count'].cumsum() / total_donors * 100).round(1)
                                
                                # Display the table
                                st.dataframe(rf_counts)
                                
                            else:
                                # Show RFM grouped distribution
                                if 'M_Score' in donor_summary.columns:
                                    rfm_dist_fig = create_rfm_grouped_distribution(donor_summary)
                                    st.plotly_chart(rfm_dist_fig, use_container_width=True)
                                    
                                    # Show distribution by RFM Sum
                                    st.markdown("##### RFM Sum Distribution")
                                    
                                    # Calculate RFM Sum if not already present
                                    if 'RFM_Sum' not in donor_summary.columns:
                                        donor_summary['RFM_Sum'] = donor_summary['R_Score'] + donor_summary['F_Score'] + donor_summary['M_Score']
                                    
                                    # Get count by RFM Sum
                                    rfm_counts = donor_summary['RFM_Sum'].value_counts().sort_index().reset_index()
                                    rfm_counts.columns = ['RFM Sum', 'Count']
                                    
                                    # Calculate percentage 
                                    rfm_counts['Percentage'] = (rfm_counts['Count'] / rfm_counts['Count'].sum() * 100).round(1)
                                    
                                    # Display the table
                                    st.dataframe(rfm_counts)
                                else:
                                    st.warning("Missing M_Score column for RFM grouped visualization. Please ensure preprocessing calculates M_Score.")
                                    
                            # Add RF Heatmap option
                            with st.expander("View as RF Heatmap", expanded=False):
                                rf_heatmap = create_rf_heatmap(donor_summary)
                                st.plotly_chart(rf_heatmap, use_container_width=True)
                            
                            # Add download button
                            if st.button("Download Chart as PNG"):
                                try:
                                    buffer = io.BytesIO()
                                    if viz_type == "Simple RF Distribution":
                                        rf_dist_fig.write_image(buffer, format="png", width=1200, height=800)
                                    else:
                                        rfm_dist_fig.write_image(buffer, format="png", width=1200, height=800)
                                    
                                    buffer.seek(0)
                                    b64 = base64.b64encode(buffer.read()).decode()
                                    download_filename = "rf_distribution.png" if viz_type == "Simple RF Distribution" else "rfm_distribution.png"
                                    href = f'<a href="data:image/png;base64,{b64}" download="{download_filename}" class="btn">Download Image</a>'
                                    st.markdown(href, unsafe_allow_html=True)
                                except Exception as e:
                                    st.error(f"Error generating download: {e}")
                        
                        except Exception as e:
                            st.error(f"Error creating RF distribution visualization: {e}")
                            st.error(f"Details: {str(e)}")
                            import traceback
                            st.error(traceback.format_exc())
                    else:
                        st.warning("RF Scores not available. Complete data preprocessing to see this visualization.")

                    # Add a download button for the visualization (optional)
                    if 'donor_summary' in st.session_state and all(col in st.session_state.donor_summary.columns for col in ['R_Score', 'F_Score', 'M_Score']):
                        download_container = st.container()
                        with download_container:
                            if st.button("Download RF Distribution Chart"):
                                try:
                                    buffer = io.BytesIO()
                                    rf_dist_fig.write_image(buffer, format="png", width=1200, height=800)
                                    buffer.seek(0)
                                    b64 = base64.b64encode(buffer.read()).decode()
                                    href = f'<a href="data:image/png;base64,{b64}" download="rf_distribution.png">Download Image</a>'
                                    st.markdown(href, unsafe_allow_html=True)
                                except Exception as e:
                                    st.error(f"Error generating download: {e}")

                    
# SIMPLE FIX: Add this code directly to your seg-app.py file
# Replace the existing "Step 4: Segmentation" section with this code


# DUAL SEGMENTATION: Replace your Step 4 with this code

# Step 4: Segmentation
with tab4:
    if st.session_state.preprocessing_done:
        st.markdown('<div class="section-header">Donor Segmentation System</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        This system lets you create two types of segments:
        <br><br>
        <b>Priority Segments:</b> Determine which donors to include in your mailing campaign
        <br>
        <b>Variable Segments:</b> Determine which content/version each included donor receives
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs for Priority and Variable segments
        priority_tab, variable_tab, ask_tab = st.tabs(["Priority Segments", "Variable Segments", "Ask Amounts"])
        
        # Get donor data and available metrics
        donor_summary = st.session_state.donor_summary
        available_metrics = donor_summary.columns.tolist()
        
        # Create lists of different metric types
        score_metrics = [col for col in available_metrics if col.endswith('_Score') or col.endswith('_Sum')]
        time_metrics = [col for col in available_metrics if col.startswith('F20') or col.startswith('M20')]
        flag_metrics = [col for col in available_metrics if col.endswith('_Flag')]
        value_metrics = ['Recency', 'Frequency', 'Monetary', 'Average_Donation', 'Min_Gift_Amount', 'Max_Gift_Amount']
        value_metrics = [m for m in value_metrics if m in available_metrics]
        date_metrics = [col for col in available_metrics if 'Date' in col or 'date' in col]
        
        # Available Donor Metrics - display in an expander
        with st.expander("Available Donor Metrics for Segmentation", expanded=False):
            st.markdown("""
            ### Available Donor Metrics
            * **RFM Scores**: `R_Score`, `F_Score`, `M_Score` (individual scores on 0-5 scale)
            * **Combined Scores**: `RFM_Sum` (Sum of all three scores), `Engagement_Score` (R + F scores)
            * **Recency Metrics**: `Recency` (days since last donation), `Last_Gift_Date` (date of last gift)
            * **Frequency Metrics**: `Frequency` (total number of donations), `F2024`, `F2023`, `F2022` (donations per year)
            * **Monetary Metrics**: `Monetary` (total donation amount), `M2024`, `M2023`, `M2022` (donation amounts per year)
            * **Gift Metrics**: `Average_Donation`, `Min_Gift_Amount`, `Max_Gift_Amount`, `Last_Donation_Amount`
            * **Special Flags**: `Monthly_Donor_Flag`, Campaign flags (e.g., `Compassion_Donor_Flag`)
            
            **Note**: The actual available metrics depend on your data. Year-specific metrics (e.g., F2024) 
            will only be available if they were calculated during data processing.
            """)
        
        # Function to create segment rules - reused for both segment types
        def create_segment_rules(prefix, num_rules=1):
            # Initialize empty rules list
            rules = []
            
            # Available metrics for rules
            rule_metrics = score_metrics + time_metrics + flag_metrics + value_metrics + date_metrics
            
            # Add additional metrics if available
            additional_metrics = ['Engagement_Score', 'Monthly_Donor_Flag']
            for metric in additional_metrics:
                if metric in available_metrics and metric not in rule_metrics:
                    rule_metrics.append(metric)
                    
            # Add donor_id for "catch-all" rules
            if 'donor_id' not in rule_metrics:
                rule_metrics.append('donor_id')
            
            for i in range(num_rules):
                st.markdown(f"##### Condition {i+1}")
                
                col1, col2, col3 = st.columns([2, 1, 2])
                
                with col1:
                    metric = st.selectbox(f"Metric {i+1}", rule_metrics, key=f"{prefix}_metric_{i}")
                
                with col2:
                    # Different operators based on metric type
                    if metric in flag_metrics or metric.endswith('_Flag'):
                        operators = ['==', '!=']
                    elif metric in date_metrics:
                        operators = ['==', '!=', '>', '<', '>=', '<=']
                    else:
                        operators = ['==', '!=', '>', '<', '>=', '<=']
                    
                    operator = st.selectbox(f"Operator {i+1}", operators, key=f"{prefix}_operator_{i}")
                
                with col3:
                    # Different value inputs based on metric type
                    if metric in flag_metrics or metric.endswith('_Flag'):
                        value = st.selectbox(f"Value {i+1}", [0, 1], key=f"{prefix}_value_{i}")
                    elif metric in date_metrics:
                        value = st.date_input(f"Value {i+1}", key=f"{prefix}_value_{i}")
                    elif metric == 'donor_id':
                        value = st.text_input(f"Value {i+1}", value="", key=f"{prefix}_value_{i}")
                    else:
                        value = st.number_input(f"Value {i+1}", key=f"{prefix}_value_{i}")
                
                # Connector (AND/OR) if not the last rule
                connector = "AND"
                if i < num_rules - 1:
                    connector = st.selectbox(f"Connect with", ["AND", "OR"], key=f"{prefix}_connector_{i}")
                
                # Add rule to list
                rules.append({
                    'metric': metric,
                    'operator': operator,
                    'value': value,
                    'connector': connector
                })
            
            return rules
        
        # Function to manage segments (add/edit/delete) - reused for both segment types
        def manage_segments(segment_type):
            # Initialize session state for segment type if not exists
            if f'{segment_type}_definitions' not in st.session_state:
                if segment_type == 'priority':
                    st.session_state.priority_definitions = [
                        {
                            'name': 'Mid-High Value', 
                            'rules': [{'metric': 'M_Score', 'operator': '>=', 'value': 4, 'connector': 'AND'},
                                    {'metric': 'R_Score', 'operator': '>=', 'value': 3, 'connector': 'AND'}]
                        },
                        {
                            'name': 'Recent Donor', 
                            'rules': [{'metric': 'R_Score', 'operator': '>=', 'value': 4, 'connector': 'AND'}]
                        },
                        {
                            'name': 'Lapsed High Value', 
                            'rules': [{'metric': 'M_Score', 'operator': '>=', 'value': 4, 'connector': 'AND'},
                                    {'metric': 'R_Score', 'operator': '<', 'value': 3, 'connector': 'AND'}]
                        },
                        {
                            'name': 'Other',
                            'rules': [{'metric': 'donor_id', 'operator': '!=', 'value': '', 'connector': 'AND'}]
                        }
                    ]
                else:  # variable segments
                    st.session_state.variable_definitions = [
                        {
                            'name': 'High Value', 
                            'rules': [{'metric': 'M_Score', 'operator': '>=', 'value': 4, 'connector': 'AND'}]
                        },
                        {
                            'name': 'Active', 
                            'rules': [{'metric': 'R_Score', 'operator': '>=', 'value': 3, 'connector': 'AND'}]
                        },
                        {
                            'name': 'New Donor', 
                            'rules': [{'metric': 'Frequency', 'operator': '==', 'value': 1, 'connector': 'AND'},
                                     {'metric': 'R_Score', 'operator': '>=', 'value': 3, 'connector': 'AND'}]
                        },
                        {
                            'name': 'Default',
                            'rules': [{'metric': 'donor_id', 'operator': '!=', 'value': '', 'connector': 'AND'}]
                        }
                    ]
            
            definitions_key = f'{segment_type}_definitions'
            
            st.markdown(f"### {segment_type.title()} Segment Management")
            
            # Choose action: Add new segment, edit existing, or delete
            action = st.radio(f"Select action for {segment_type} segments:", 
                             ["Add new segment", "Edit existing segment", "Delete segment", "Reorder segments"], 
                             horizontal=True,
                             key=f"{segment_type}_action")
            
            if action == "Add new segment":
                # Get segment name
                segment_name = st.text_input(f"Enter {segment_type} segment name", 
                                           value=f"New {segment_type.title()} Segment", 
                                           key=f"{segment_type}_new_name")
                
                # Number of rules
                num_rules = st.number_input(f"Number of conditions for this {segment_type} segment", 
                                          min_value=1, max_value=10, value=1,
                                          key=f"{segment_type}_num_rules")
                
                # Create rules for the segment
                rules = create_segment_rules(f"{segment_type}_new", num_rules)
                
                # Add button
                if st.button(f"Add {segment_type.title()} Segment", key=f"add_{segment_type}"):
                    getattr(st.session_state, definitions_key).append({
                        'name': segment_name,
                        'rules': rules
                    })
                    st.success(f"{segment_type.title()} segment '{segment_name}' added!")
                    st.rerun()
                    
            elif action == "Edit existing segment":
                # Select segment to edit
                segment_names = [seg['name'] for seg in getattr(st.session_state, definitions_key)]
                selected_segment = st.selectbox(f"Select {segment_type} segment to edit", 
                                              segment_names,
                                              key=f"{segment_type}_edit_select")
                
                # Find the selected segment
                segment_index = segment_names.index(selected_segment)
                current_segment = getattr(st.session_state, definitions_key)[segment_index]
                
                # Edit segment name
                new_name = st.text_input(f"{segment_type.title()} segment name", 
                                       value=current_segment['name'],
                                       key=f"{segment_type}_edit_name")
                
                # Number of rules
                num_rules = st.number_input(f"Number of conditions for this {segment_type} segment", 
                                          min_value=1, max_value=10, value=len(current_segment['rules']),
                                          key=f"{segment_type}_edit_num_rules")
                
                # Edit rules
                rules = create_segment_rules(f"{segment_type}_edit", num_rules)
                
                # Update button
                if st.button(f"Update {segment_type.title()} Segment", key=f"update_{segment_type}"):
                    getattr(st.session_state, definitions_key)[segment_index] = {
                        'name': new_name,
                        'rules': rules
                    }
                    st.success(f"{segment_type.title()} segment '{new_name}' updated!")
                    st.rerun()
                    
            elif action == "Delete segment":
                # Select segment to delete
                segment_names = [seg['name'] for seg in getattr(st.session_state, definitions_key)]
                selected_segment = st.selectbox(f"Select {segment_type} segment to delete",
                                              segment_names,
                                              key=f"{segment_type}_delete_select")
                
                # Delete button
                if st.button(f"Delete {segment_type.title()} Segment", key=f"delete_{segment_type}"):
                    segment_index = segment_names.index(selected_segment)
                    getattr(st.session_state, definitions_key).pop(segment_index)
                    st.success(f"{segment_type.title()} segment '{selected_segment}' deleted!")
                    st.rerun()
                    
            elif action == "Reorder segments":
                st.markdown(f"Drag to reorder {segment_type} segments (higher = higher priority):")
                
                segment_list = getattr(st.session_state, definitions_key)
                segment_names = [seg['name'] for seg in segment_list]
                
                # Create a numbered list with up/down buttons
                for i, name in enumerate(segment_names):
                    col1, col2, col3 = st.columns([1, 4, 1])
                    
                    with col1:
                        st.write(f"**{i+1}.**")
                    
                    with col2:
                        st.write(f"**{name}**")
                    
                    with col3:
                        up_disabled = (i == 0)
                        down_disabled = (i == len(segment_names) - 1)
                        
                        if not up_disabled and st.button("↑", key=f"{segment_type}_up_{i}"):
                            # Swap with previous segment
                            segment_list[i], segment_list[i-1] = segment_list[i-1], segment_list[i]
                            st.rerun()
                            
                        if not down_disabled and st.button("↓", key=f"{segment_type}_down_{i}"):
                            # Swap with next segment
                            segment_list[i], segment_list[i+1] = segment_list[i+1], segment_list[i]
                            st.rerun()
        
        # PRIORITY SEGMENTS TAB
        with priority_tab:
            st.markdown('<div class="subsection-header">Priority Segments</div>', unsafe_allow_html=True)
            st.markdown("""
            Priority segments determine **which donors to include** in your mailing campaign. 
            Donors who match a priority segment other than "Other" or "Exclude" will be included in the final export.
            """)
            
            # Priority segment management
            with st.expander("Manage Priority Segments", expanded=True):
                manage_segments('priority')
            
            # Preview of current priority segment definitions
            st.markdown('<div class="subsection-header">Current Priority Segment Definitions</div>', unsafe_allow_html=True)
            
            if 'priority_definitions' in st.session_state:
                for i, segment in enumerate(st.session_state.priority_definitions):
                    with st.expander(f"{i+1}. {segment['name']}", expanded=True):
                        rules_text = ""
                        for j, rule in enumerate(segment['rules']):
                            rules_text += f"**{rule['metric']}** {rule['operator']} **{rule['value']}**"
                            if j < len(segment['rules']) - 1:
                                rules_text += f" *{rule['connector']}* "
                        st.markdown(rules_text)
                
                # Option to mark certain segments as "exclude"
                st.markdown("### Segments to Exclude from Mailing")
                st.markdown("Select any segments that should be **excluded** from the mailing:")
                
                if 'exclude_segments' not in st.session_state:
                    st.session_state.exclude_segments = []
                
                segment_names = [seg['name'] for seg in st.session_state.priority_definitions]
                exclude_segments = st.multiselect(
                    "Segments to exclude",
                    options=segment_names,
                    default=st.session_state.exclude_segments,
                    key="priority_exclude"
                )
                st.session_state.exclude_segments = exclude_segments
        
        # VARIABLE SEGMENTS TAB
        with variable_tab:
            st.markdown('<div class="subsection-header">Variable Segments</div>', unsafe_allow_html=True)
            st.markdown("""
            Variable segments determine **which content version** each included donor receives. 
            These segments are applied only to donors who are already included based on their priority segment.
            """)
            
            # Variable segment management
            with st.expander("Manage Variable Segments", expanded=True):
                manage_segments('variable')
            
            # Preview of current variable segment definitions
            st.markdown('<div class="subsection-header">Current Variable Segment Definitions</div>', unsafe_allow_html=True)
            
            if 'variable_definitions' in st.session_state:
                for i, segment in enumerate(st.session_state.variable_definitions):
                    with st.expander(f"{i+1}. {segment['name']}", expanded=True):
                        rules_text = ""
                        for j, rule in enumerate(segment['rules']):
                            rules_text += f"**{rule['metric']}** {rule['operator']} **{rule['value']}**"
                            if j < len(segment['rules']) - 1:
                                rules_text += f" *{rule['connector']}* "
                        st.markdown(rules_text)
        
        # ASK AMOUNTS TAB
        with ask_tab:
            st.markdown('<div class="subsection-header">Ask Amount Configuration</div>', unsafe_allow_html=True)
            
            ask_method = st.radio(
                "Ask Amount Calculation Method",
                options=["Multiplier of Average Donation", "Fixed Tiers", "Need-Based", "Fixed by Segment"],
                index=0
            )
            
            if ask_method == "Multiplier of Average Donation":
                ask1_multiplier = st.number_input("Ask 1 Multiplier", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
                ask2_multiplier = st.number_input("Ask 2 Multiplier", min_value=0.1, max_value=5.0, value=1.5, step=0.1)
                ask3_multiplier = st.number_input("Ask 3 Multiplier", min_value=0.1, max_value=5.0, value=2.0, step=0.1)
                ask4_multiplier = st.number_input("Ask 4 Multiplier", min_value=0.1, max_value=5.0, value=2.5, step=0.1)
                round_to = st.number_input("Round to nearest", min_value=1, max_value=100, value=5)
                min_ask = st.number_input("Minimum Ask Amount", min_value=1, max_value=1000, value=20)
                max_ask_avg = st.number_input("Maximum Average Donation to Calculate Asks For", min_value=50, max_value=10000, value=1000)
            
            elif ask_method == "Fixed Tiers":
                tier1 = st.number_input("Tier 1 (Low)", min_value=1, max_value=1000, value=25)
                tier2 = st.number_input("Tier 2 (Medium)", min_value=1, max_value=1000, value=50)
                tier3 = st.number_input("Tier 3 (High)", min_value=1, max_value=5000, value=100)
                tier4 = st.number_input("Tier 4 (Premium)", min_value=1, max_value=10000, value=250)
            
            elif ask_method == "Need-Based":
                base_unit = st.number_input("Base Unit Cost (e.g., $80 feeds one family)", min_value=1, max_value=1000, value=80)
                base_unit_desc = st.text_input("Base Unit Description", value="feeds a family")
            
            elif ask_method == "Fixed by Segment":
                st.markdown("##### Set Fixed Ask Amounts by Variable Segment")
                
                segment_asks = {}
                if 'variable_definitions' in st.session_state:
                    for segment in st.session_state.variable_definitions:
                        st.markdown(f"**{segment['name']}**")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            ask1 = st.number_input(f"Ask 1 for {segment['name']}", min_value=0, max_value=10000, value=25, step=5, key=f"ask1_{segment['name']}")
                        with col2:
                            ask2 = st.number_input(f"Ask 2 for {segment['name']}", min_value=0, max_value=10000, value=50, step=5, key=f"ask2_{segment['name']}")
                        with col3:
                            ask3 = st.number_input(f"Ask 3 for {segment['name']}", min_value=0, max_value=10000, value=100, step=5, key=f"ask3_{segment['name']}")
                        with col4:
                            ask4 = st.number_input(f"Ask 4 for {segment['name']}", min_value=0, max_value=10000, value=250, step=5, key=f"ask4_{segment['name']}")
                        
                        segment_asks[segment['name']] = [ask1, ask2, ask3, ask4]
                    
                    st.session_state.segment_asks = segment_asks
            
            # Language customization
            st.markdown("##### Ask String Customization")
            
            english_template = st.text_input(
                "English Ask Template", 
                value="${amount} to provide care and community to seniors like Hector."
            )
            
            french_template = st.text_input(
                "French Ask Template", 
                value="{amount} $ pour offrir des soins et une communauté aux aînés comme Hector."
            )
        
        # Apply segmentation button - outside the tabs to always be visible
        # Replace the "Apply segmentation button" section in your code with this fixed version

# Apply segmentation button - outside the tabs to always be visible
st.markdown('<div class="subsection-header">Apply Segmentation</div>', unsafe_allow_html=True)

if st.button("Apply Segmentation and Calculate Ask Amounts", key="apply_all_segments"):
    try:
        with st.spinner("Applying segmentation and calculating ask amounts..."):
            # Get the donor summary
            donor_data = donor_summary.copy()
            
            # Helper function to evaluate rules
            def evaluate_rule(row, rule):
                try:
                    metric = rule['metric']
                    operator = rule['operator']
                    value = rule['value']
                    
                    # Skip if metric doesn't exist in the row
                    if metric not in row:
                        return False
                    
                    # Handle date fields differently
                    if isinstance(value, (pd.Timestamp, datetime)) or (isinstance(value, str) and any(x in metric.lower() for x in ['date', 'time'])):
                        try:
                            row_value = pd.to_datetime(row[metric]) if pd.notna(row[metric]) else None
                            value = pd.to_datetime(value) if not isinstance(value, (pd.Timestamp, datetime)) else value
                            
                            if operator == '==':
                                return row_value == value
                            elif operator == '!=':
                                return row_value != value
                            elif operator == '>':
                                return row_value > value if row_value else False
                            elif operator == '<':
                                return row_value < value if row_value else False
                            elif operator == '>=':
                                return row_value >= value if row_value else False
                            elif operator == '<=':
                                return row_value <= value if row_value else False
                        except:
                            # If date conversion fails, just compare as strings
                            row_value = str(row[metric]) if pd.notna(row[metric]) else ""
                            value = str(value)
                            
                            if operator == '==':
                                return row_value == value
                            elif operator == '!=':
                                return row_value != value
                            else:
                                return False
                    else:
                        # Handle null values
                        if pd.isna(row[metric]):
                            if operator == '==':
                                return pd.isna(value)
                            elif operator == '!=':
                                return not pd.isna(value)
                            else:
                                return False
                        
                        # Convert values to appropriate types
                        try:
                            row_value = float(row[metric]) if isinstance(row[metric], (int, float, str)) else row[metric]
                            value = float(value) if isinstance(value, (int, float, str)) and not isinstance(row_value, str) else value
                        except:
                            row_value = str(row[metric])
                            value = str(value)
                        
                        # Standard comparisons for non-date fields
                        if operator == '==':
                            return row_value == value
                        elif operator == '!=':
                            return row_value != value
                        elif operator == '>':
                            return row_value > value
                        elif operator == '<':
                            return row_value < value
                        elif operator == '>=':
                            return row_value >= value
                        elif operator == '<=':
                            return row_value <= value
                    
                    return False
                except Exception as e:
                    st.warning(f"Error evaluating rule {rule} on row: {e}")
                    return False
            
            def evaluate_segment_rules(row, rules):
                try:
                    if not rules:
                        return False
                        
                    # Start with True for AND chains, False for OR chains
                    first_connector = rules[0]['connector'] if len(rules) > 1 else 'AND'
                    if first_connector == 'AND':
                        result = True
                    else:
                        result = False
                        
                    # First rule
                    current_result = evaluate_rule(row, rules[0])
                    
                    # Process subsequent rules
                    for i in range(1, len(rules)):
                        prev_connector = rules[i-1]['connector']
                        if prev_connector == 'AND':
                            result = result and current_result
                        else:  # OR
                            result = result or current_result
                            
                        # Evaluate the current rule for the next iteration
                        current_result = evaluate_rule(row, rules[i])
                        
                    # Handle the last rule result
                    if len(rules) == 1:
                        return current_result
                    else:
                        if first_connector == 'AND':
                            return result and current_result
                        else:  # OR
                            return result or current_result
                except Exception as e:
                    st.warning(f"Error evaluating segment rules: {e}")
                    return False
            
            # APPLY PRIORITY SEGMENTS
            if 'priority_definitions' in st.session_state:
                # Create a new column for priority segments
                donor_data['Priority_Segment'] = 'Uncategorized'
                
                # Apply each priority segment definition in order
                for segment in st.session_state.priority_definitions:
                    # Find donors who haven't been assigned yet and match this segment's rules
                    mask = (donor_data['Priority_Segment'] == 'Uncategorized')
                    if mask.any() and segment['rules']:
                        # For each unassigned donor, check if they match this segment's rules
                        for idx in donor_data[mask].index:
                            row = donor_data.loc[idx]
                            if evaluate_segment_rules(row, segment['rules']):
                                donor_data.at[idx, 'Priority_Segment'] = segment['name']
            
            # Mark excluded segments
            donor_data['Include_In_Mailing'] = 1  # Default to include
            
            if 'exclude_segments' in st.session_state:
                for segment_name in st.session_state.exclude_segments:
                    donor_data.loc[donor_data['Priority_Segment'] == segment_name, 'Include_In_Mailing'] = 0
                
                # Also exclude Uncategorized by default
                donor_data.loc[donor_data['Priority_Segment'] == 'Uncategorized', 'Include_In_Mailing'] = 0
            else:
                # Default - exclude 'Other' or 'Uncategorized'
                donor_data.loc[donor_data['Priority_Segment'].isin(['Other', 'Uncategorized']), 'Include_In_Mailing'] = 0
            
            # APPLY VARIABLE SEGMENTS
            if 'variable_definitions' in st.session_state:
                # Create a new column for variable segments
                donor_data['Variable_Segment'] = 'Uncategorized'
                
                # Only apply to donors included in the mailing
                included_mask = donor_data['Include_In_Mailing'] == 1
                
                # Apply each variable segment definition in order
                for segment in st.session_state.variable_definitions:
                    # Find included donors who haven't been assigned yet and match this segment's rules
                    mask = (donor_data['Variable_Segment'] == 'Uncategorized') & (donor_data['Include_In_Mailing'] == 1)
                    if mask.any() and segment['rules']:
                        # For each unassigned donor, check if they match this segment's rules
                        for idx in donor_data[mask].index:
                            row = donor_data.loc[idx]
                            if evaluate_segment_rules(row, segment['rules']):
                                donor_data.at[idx, 'Variable_Segment'] = segment['name']
            
            # CALCULATE ASK AMOUNTS
            if ask_method == "Multiplier of Average Donation":
                def calculate_ask_values(avg_donation):
                    if pd.isna(avg_donation) or avg_donation > max_ask_avg:
                        return None, None, None, None
                    
                    ask1 = max(min_ask, round(avg_donation * ask1_multiplier / round_to) * round_to)
                    ask2 = max(min_ask, round(avg_donation * ask2_multiplier / round_to) * round_to)
                    ask3 = max(min_ask, round(avg_donation * ask3_multiplier / round_to) * round_to)
                    ask4 = max(min_ask, round(avg_donation * ask4_multiplier / round_to) * round_to)
                    
                    return ask1, ask2, ask3, ask4
                
                # Apply to all included donors
                included_mask = donor_data['Include_In_Mailing'] == 1
                
                # Initialize ask columns with NaN
                donor_data['Ask_1'] = float('nan')
                donor_data['Ask_2'] = float('nan') 
                donor_data['Ask_3'] = float('nan')
                donor_data['Ask_4'] = float('nan')
                
                # Calculate for included donors
                for idx in donor_data[included_mask].index:
                    avg_donation = donor_data.at[idx, 'Average_Donation']
                    ask1, ask2, ask3, ask4 = calculate_ask_values(avg_donation)
                    
                    donor_data.at[idx, 'Ask_1'] = ask1
                    donor_data.at[idx, 'Ask_2'] = ask2
                    donor_data.at[idx, 'Ask_3'] = ask3
                    donor_data.at[idx, 'Ask_4'] = ask4
                
            elif ask_method == "Fixed Tiers":
                # Same fixed tiers for all included donors
                included_mask = donor_data['Include_In_Mailing'] == 1
                donor_data.loc[included_mask, 'Ask_1'] = tier1
                donor_data.loc[included_mask, 'Ask_2'] = tier2
                donor_data.loc[included_mask, 'Ask_3'] = tier3
                donor_data.loc[included_mask, 'Ask_4'] = tier4
                
            elif ask_method == "Need-Based":
                # Based on the unit cost
                included_mask = donor_data['Include_In_Mailing'] == 1
                donor_data.loc[included_mask, 'Ask_1'] = base_unit
                donor_data.loc[included_mask, 'Ask_2'] = base_unit * 2
                donor_data.loc[included_mask, 'Ask_3'] = base_unit * 3
                donor_data.loc[included_mask, 'Ask_4'] = base_unit * 5
                
            elif ask_method == "Fixed by Segment":
                # Initialize with NaN
                donor_data['Ask_1'] = float('nan')
                donor_data['Ask_2'] = float('nan')
                donor_data['Ask_3'] = float('nan')
                donor_data['Ask_4'] = float('nan')
                
                # Apply segment-specific ask amounts to variable segments
                if 'segment_asks' in st.session_state:
                    for segment_name, asks in st.session_state.segment_asks.items():
                        mask = (donor_data['Variable_Segment'] == segment_name) & (donor_data['Include_In_Mailing'] == 1)
                        if asks[0] > 0: donor_data.loc[mask, 'Ask_1'] = asks[0]
                        if asks[1] > 0: donor_data.loc[mask, 'Ask_2'] = asks[1]
                        if asks[2] > 0: donor_data.loc[mask, 'Ask_3'] = asks[2]
                        if asks[3] > 0: donor_data.loc[mask, 'Ask_4'] = asks[3]
            
            # Generate ask strings
            def generate_ask_string(amount, language):
                if pd.isna(amount):
                    return ""
                try:
                    amount = int(amount)
                    if language == "Francais" or language == "French":
                        return french_template.replace("{amount}", str(amount))
                    return english_template.replace("{amount}", str(amount))
                except:
                    return ""
            
            # Apply language check function that handles various language column formats
            def is_french(language_value):
                if pd.isna(language_value):
                    return False
                lang_str = str(language_value).lower()
                return 'fr' in lang_str or 'francais' in lang_str or 'french' in lang_str
            
            # Generate ask strings based on language
            language_col = None
            potential_lang_cols = ['Langue_desc', 'langue', 'language', 'lang', 'donor_language']
            
            for col in potential_lang_cols:
                if col in donor_data.columns:
                    language_col = col
                    break
            
            # Only generate ask strings for included donors
            included_mask = donor_data['Include_In_Mailing'] == 1
            
            # Initialize ask string columns
            donor_data['Ask_1_String'] = ""
            donor_data['Ask_2_String'] = ""
            donor_data['Ask_3_String'] = ""
            donor_data['Ask_4_String'] = ""
            
            if language_col:
                # Create is_french column only for included donors
                donor_data['is_french'] = False
                donor_data.loc[included_mask, 'is_french'] = donor_data.loc[included_mask, language_col].apply(is_french)
                
                # Generate ask strings for each included donor
                for idx in donor_data[included_mask].index:
                    is_fr = donor_data.at[idx, 'is_french']
                    lang = 'Francais' if is_fr else 'English'
                    
                    donor_data.at[idx, 'Ask_1_String'] = generate_ask_string(donor_data.at[idx, 'Ask_1'], lang)
                    donor_data.at[idx, 'Ask_2_String'] = generate_ask_string(donor_data.at[idx, 'Ask_2'], lang)
                    donor_data.at[idx, 'Ask_3_String'] = generate_ask_string(donor_data.at[idx, 'Ask_3'], lang)
                    donor_data.at[idx, 'Ask_4_String'] = generate_ask_string(donor_data.at[idx, 'Ask_4'], lang)
            else:
                # Default to English if no language column is found
                for idx in donor_data[included_mask].index:
                    donor_data.at[idx, 'Ask_1_String'] = generate_ask_string(donor_data.at[idx, 'Ask_1'], 'English')
                    donor_data.at[idx, 'Ask_2_String'] = generate_ask_string(donor_data.at[idx, 'Ask_2'], 'English')
                    donor_data.at[idx, 'Ask_3_String'] = generate_ask_string(donor_data.at[idx, 'Ask_3'], 'English')
                    donor_data.at[idx, 'Ask_4_String'] = generate_ask_string(donor_data.at[idx, 'Ask_4'], 'English')
            
            # Store the results - use all data, not just filtered
            st.session_state.donor_summary_filtered = donor_data
            st.session_state.segmentation_done = True
            
            # Success message
            st.success("Segmentation and ask amounts calculated successfully!")
            
            # Show summary statistics
            st.markdown("### Segmentation Results")
            
            # Display priority segment stats
            st.markdown("#### Priority Segments")
            
            # Count by priority segment
            priority_counts = donor_data['Priority_Segment'].value_counts().reset_index()
            priority_counts.columns = ['Segment', 'Count']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Priority segments pie chart
                fig1 = px.pie(
                    priority_counts, 
                    values='Count', 
                    names='Segment',
                    title='Priority Segment Distribution'
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Inclusion breakdown
                included_donors = donor_data[donor_data['Include_In_Mailing'] == 1]
                excluded_donors = donor_data[donor_data['Include_In_Mailing'] == 0]
                
                inclusion_data = pd.DataFrame([
                    {'Status': 'Included in Mailing', 'Count': len(included_donors)},
                    {'Status': 'Excluded from Mailing', 'Count': len(excluded_donors)}
                ])
                
                fig2 = px.pie(
                    inclusion_data, 
                    values='Count', 
                    names='Status',
                    title='Mailing Inclusion Status',
                    color_discrete_sequence=['#28a745', '#dc3545']
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Display variable segment stats
            st.markdown("#### Variable Segments")
            
            # Only show for included donors
            var_counts = donor_data[donor_data['Include_In_Mailing'] == 1]['Variable_Segment'].value_counts().reset_index()
            var_counts.columns = ['Segment', 'Count']
            
            # Variable segments bar chart
            fig3 = px.bar(
                var_counts, 
                x='Segment', 
                y='Count',
                title='Variable Segment Distribution (Included Donors Only)',
                color='Segment'
            )
            st.plotly_chart(fig3, use_container_width=True)
            
            # Display segment count table
            st.markdown("#### Segment Counts")
            
            # Priority segment counts
            st.markdown("**Priority Segments**")
            st.dataframe(priority_counts.set_index('Segment'))
            
            # Variable segment counts (included donors only)
            st.markdown("**Variable Segments (Included Donors Only)**")
            st.dataframe(var_counts.set_index('Segment'))
            
            # Ask amount preview
            if 'Ask_1' in donor_data.columns:
                st.markdown("#### Ask Amount Preview")
                
                # Get average ask by variable segment for included donors
                ask_preview = donor_data[donor_data['Include_In_Mailing'] == 1].groupby('Variable_Segment')[
                    ['Ask_1', 'Ask_2', 'Ask_3', 'Ask_4']
                ].mean().reset_index()
                
                # Format for display
                for col in ['Ask_1', 'Ask_2', 'Ask_3', 'Ask_4']:
                    ask_preview[col] = ask_preview[col].apply(lambda x: f"${x:.2f}" if not pd.isna(x) else "N/A")
                
                st.dataframe(ask_preview.set_index('Variable_Segment'))
            
            # Button to go to export page
            if st.button("Go to Final Export"):
                st.session_state.current_step = 5
                st.rerun()
    
    except Exception as e:
        st.error(f"Error during segmentation: {e}")
        import traceback
        st.error(traceback.format_exc())

# Step 5: Final Export - Updated for Dual Segmentation System
with tab5:
    if 'segmentation_done' in st.session_state and st.session_state.segmentation_done and 'donor_summary_filtered' in st.session_state:
        st.markdown('<div class="section-header">Final Export</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        Review the final segmented donor list and download it for your mailing campaign.
        Only donors with <b>Include_In_Mailing = 1</b> will be included in the export by default.
        </div>
        """, unsafe_allow_html=True)
        
        # Get the final donor data
        donor_data = st.session_state.donor_summary_filtered
        
        # Display summary statistics
        st.markdown('<div class="subsection-header">Export Summary</div>', unsafe_allow_html=True)
        
        # Count donors included in mailing
        included_donors = donor_data[donor_data['Include_In_Mailing'] == 1]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Donors", f"{len(donor_data):,}")
        col2.metric("Included in Mailing", f"{len(included_donors):,}")
        col3.metric("Priority Segments", f"{donor_data['Priority_Segment'].nunique()}")
        col4.metric("Variable Segments", f"{donor_data['Variable_Segment'].nunique()}")
        
        # Summary by segment
        st.markdown("##### Donor Count by Priority Segment")
        segment_summary = donor_data.groupby('Priority_Segment').agg({
            'donor_id': 'count',
            'Include_In_Mailing': 'sum',
            'Monetary': 'sum',
            'Average_Donation': 'mean'
        }).reset_index()
        
        segment_summary.columns = ['Priority Segment', 'Total Donors', 'Included in Mailing', 'Total Donations', 'Average Donation']
        segment_summary['Total Donations'] = segment_summary['Total Donations'].apply(lambda x: f"${x:,.2f}")
        segment_summary['Average Donation'] = segment_summary['Average Donation'].apply(lambda x: f"${x:,.2f}")
        
        st.dataframe(segment_summary)
        
        # Summary by variable segment (included donors only)
        st.markdown("##### Donor Count by Variable Segment (Included Donors Only)")
        var_segment_summary = included_donors.groupby('Variable_Segment').agg({
            'donor_id': 'count',
            'Monetary': 'sum',
            'Average_Donation': 'mean'
        }).reset_index()
        
        var_segment_summary.columns = ['Variable Segment', 'Donor Count', 'Total Donations', 'Average Donation']
        var_segment_summary['Total Donations'] = var_segment_summary['Total Donations'].apply(lambda x: f"${x:,.2f}")
        var_segment_summary['Average Donation'] = var_segment_summary['Average Donation'].apply(lambda x: f"${x:,.2f}")
        
        st.dataframe(var_segment_summary)
        
        # Ask amount preview
        ask_cols = [col for col in donor_data.columns if col.startswith('Ask_') and not col.endswith('_String')]
        if ask_cols:
            st.markdown("##### Ask Amount Preview by Variable Segment")
            ask_preview = included_donors.groupby('Variable_Segment')[ask_cols].mean().reset_index()
            
            # Format currency
            for col in ask_cols:
                if col in ask_preview.columns:
                    ask_preview[col] = ask_preview[col].apply(lambda x: f"${x:,.2f}" if not pd.isna(x) else "N/A")
            
            st.dataframe(ask_preview)
        
        # Preview of the export data
        st.markdown('<div class="subsection-header">Export Data Preview</div>', unsafe_allow_html=True)
        
        # Option to include all donors or only those marked for inclusion
        include_all = st.checkbox("Include ALL donors in export (not just those marked for mailing)", value=False)
        
        # Filter data based on inclusion setting
        if include_all:
            export_base_data = donor_data.copy()
        else:
            export_base_data = included_donors.copy()
        
        # Let user select columns to include in export
        available_columns = export_base_data.columns.tolist()
        
        # Define column categories for easier selection
        contact_cols = [col for col in available_columns if any(x in col.lower() for x in ['donor', 'postal', 'email', 'mobile', 'phone', 'address'])]
        segment_cols = [col for col in available_columns if any(x in col.lower() for x in ['segment', 'flag', 'include'])]
        analytics_cols = [col for col in available_columns if any(x in col.lower() for x in ['score', 'rfm', 'recency', 'frequency', 'monetary'])]
        ask_cols = [col for col in available_columns if 'ask' in col.lower()]
        
        # Create column selection with expandable sections
        with st.expander("Select Columns for Export", expanded=True):
            st.markdown("##### Required Columns")
            st.info("donor_id is always included")
            
            st.markdown("##### Contact Information")
            selected_contact = st.multiselect("Select contact columns", options=contact_cols, default=contact_cols)
            
            st.markdown("##### Segmentation")
            selected_segment = st.multiselect("Select segmentation columns", 
                                            options=segment_cols, 
                                            default=['Priority_Segment', 'Variable_Segment', 'Include_In_Mailing'])
            
            st.markdown("##### Analytics")
            selected_analytics = st.multiselect("Select analytics columns", options=analytics_cols, default=[])
            
            st.markdown("##### Ask Amounts")
            selected_ask = st.multiselect("Select ask amount columns", options=ask_cols, default=[col for col in ask_cols if 'String' in col])
            
            # Combine all selected columns
            selected_columns = ['donor_id'] + selected_contact + selected_segment + selected_analytics + selected_ask
            
            # Remove duplicates while preserving order
            seen = set()
            selected_columns = [x for x in selected_columns if not (x in seen or seen.add(x))]
        
        # Make sure selected columns exist in the dataframe
        selected_columns = [col for col in selected_columns if col in export_base_data.columns]
        
        # Preview of export data
        export_data = export_base_data[selected_columns].copy()
        st.dataframe(export_data.head(10))
        
        # Total rows in export
        st.info(f"Total donors in export: {len(export_data):,}")
        
        # Button to download data
        col1, col2 = st.columns(2)
        
        # Download options
        file_format = col1.radio("File Format", options=["CSV", "Excel"], horizontal=True)
        col2.markdown("&nbsp;")  # Spacing
        col2.markdown("&nbsp;")  # Spacing
        include_stats = col2.checkbox("Include segment statistics", value=True)
        
        # Generate downloadable file
        if st.button("Generate Export File"):
            try:
                with st.spinner("Generating export file..."):
                    # Create a BytesIO buffer
                    buffer = io.BytesIO()
                    
                    if file_format == "Excel":
                        # Create Excel file
                        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                            export_data.to_excel(writer, sheet_name='Donor Data', index=False)
                            
                            # Add segment statistics if requested
                            if include_stats:
                                # Priority Segment Statistics
                                segment_stats = donor_data.groupby('Priority_Segment').agg({
                                    'donor_id': 'count',
                                    'Include_In_Mailing': 'sum',
                                    'Monetary': 'sum',
                                    'Average_Donation': 'mean',
                                    'Frequency': 'mean',
                                    'Recency': 'mean',
                                    'R_Score': 'mean',
                                    'F_Score': 'mean',
                                    'M_Score': 'mean',
                                    'RFM_Sum': 'mean'
                                }).reset_index()
                                
                                segment_stats.columns = [
                                    'Priority Segment', 'Total Donors', 'Included in Mailing', 
                                    'Total Donations', 'Avg Donation', 'Avg Frequency', 
                                    'Avg Recency Days', 'Avg R', 'Avg F', 'Avg M', 'Avg RFM'
                                ]
                                
                                segment_stats.to_excel(writer, sheet_name='Priority Segments', index=False)
                                
                                # Variable Segment Statistics (for included donors)
                                var_segment_stats = included_donors.groupby('Variable_Segment').agg({
                                    'donor_id': 'count',
                                    'Monetary': 'sum',
                                    'Average_Donation': 'mean',
                                    'R_Score': 'mean',
                                    'F_Score': 'mean',
                                    'M_Score': 'mean',
                                    'RFM_Sum': 'mean'
                                }).reset_index()
                                
                                var_segment_stats.columns = [
                                    'Variable Segment', 'Donor Count', 'Total Donations', 
                                    'Avg Donation', 'Avg R', 'Avg F', 'Avg M', 'Avg RFM'
                                ]
                                
                                var_segment_stats.to_excel(writer, sheet_name='Variable Segments', index=False)
                                
                                # Ask Amount Statistics
                                if 'Ask_1' in included_donors.columns:
                                    ask_stats = included_donors.groupby('Variable_Segment')[
                                        ['Ask_1', 'Ask_2', 'Ask_3', 'Ask_4']
                                    ].mean().reset_index()
                                    
                                    ask_stats.to_excel(writer, sheet_name='Ask Amounts', index=False)
                                
                                # Cross-tabulation of Priority and Variable segments
                                if len(included_donors) > 0:
                                    cross_tab = pd.crosstab(
                                        included_donors['Priority_Segment'], 
                                        included_donors['Variable_Segment'],
                                        margins=True
                                    )
                                    
                                    cross_tab.to_excel(writer, sheet_name='Segment Cross-Tab')
                        
                        # Get the Excel data
                        buffer.seek(0)
                        excel_data = buffer.getvalue()
                        
                        # Create a download link
                        b64 = base64.b64encode(excel_data).decode()
                        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="donor_segmentation.xlsx" class="btn">Download Excel File</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        
                    else:  # CSV format
                        # Convert to CSV
                        csv = export_data.to_csv(index=False)
                        
                        # Create a download link
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="donor_segmentation.csv" class="btn">Download CSV File</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        
                        # If stats were requested, create a separate stats CSV
                        if include_stats:
                            # Priority Segment Statistics
                            segment_stats = donor_data.groupby('Priority_Segment').agg({
                                'donor_id': 'count',
                                'Include_In_Mailing': 'sum',
                                'Monetary': 'sum',
                                'Average_Donation': 'mean',
                                'Frequency': 'mean',
                                'Recency': 'mean'
                            }).reset_index()
                            
                            segment_stats.columns = [
                                'Priority Segment', 'Total Donors', 'Included in Mailing', 
                                'Total Donations', 'Avg Donation', 'Avg Frequency', 'Avg Recency Days'
                            ]
                            
                            # Convert to CSV
                            stats_csv = segment_stats.to_csv(index=False)
                            
                            # Create a download link
                            b64_stats = base64.b64encode(stats_csv.encode()).decode()
                            href_stats = f'<a href="data:file/csv;base64,{b64_stats}" download="segment_statistics.csv" class="btn">Download Segment Statistics CSV</a>'
                            st.markdown(href_stats, unsafe_allow_html=True)
                    
                    st.success("Export file generated successfully! Click the link above to download.")
            
            except Exception as e:
                st.error(f"Error generating export file: {e}")
                import traceback
                st.error(traceback.format_exc())
        
        # Documentation section
        with st.expander("Export Documentation", expanded=False):
            st.markdown("""
            ## Export Documentation
            
            ### File Contents
            
            The export file contains donors segmented according to your defined rules:
            
            - **Priority Segments**: Determine which donors are included in the mailing
            - **Variable Segments**: Determine which content version each donor receives
            - **Include_In_Mailing**: Flag (1=include, 0=exclude) indicating whether the donor should be included
            
            ### Usage Notes
            
            1. **Mail House Integration**: The exported CSV/Excel file can be provided to your mail house with instructions to:
               - Mail only to donors with Include_In_Mailing = 1
               - Use Variable_Segment to determine which content version to send
               - Use Ask_X_String fields for the specific donation ask amounts
            
            2. **Exclusion Logic**: By default, donors in segments marked for exclusion or in the 'Other' segment are not included in the mailing (Include_In_Mailing = 0).
            
            3. **Statistical Tabs**: When selecting Excel format with statistics, the file includes these additional tabs:
               - Priority Segments: Statistics for each priority segment
               - Variable Segments: Statistics for each variable segment (included donors only)
               - Ask Amounts: Average ask amounts by segment
               - Segment Cross-Tab: Cross-tabulation of Priority vs Variable segments
            """)
    
    else:
        st.warning("Please complete segmentation in the previous step first.")
        if st.button("Go to Segmentation"):
            st.session_state.current_step = 4
            st.rerun()

# Footer
st.markdown("""
---
<div style="text-align: center; color: #666;">
Donor Segmentation Tool | Developed for Mailing List Prioritization
</div>
""", unsafe_allow_html=True)
                