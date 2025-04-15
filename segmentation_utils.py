# segmentation_utils.py
# Import this file in your main app

import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import io
import base64
import streamlit as st


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


def apply_segmentation(donor_summary, segment_config):
    """
    Applies segmentation logic to donor data
    
    Parameters:
    -----------
    donor_summary : pandas.DataFrame
        The donor summary dataframe
    segment_config : dict
        Configuration for segmentation
        
    Returns:
    --------
    pandas.DataFrame
        Donor summary with segmentation applied
    """
    try:
        # Create a copy to avoid modifying the original
        df = donor_summary.copy()
        
        # Extract config parameters
        mm_monetary_threshold = segment_config.get('mm_monetary_threshold', 4)
        mm_recency_threshold = segment_config.get('mm_recency_threshold', 2)
        new_freq_threshold = segment_config.get('new_freq_threshold', 1)
        new_recency_threshold = segment_config.get('new_recency_threshold', 3)
        active_recency_threshold = segment_config.get('active_recency_threshold', 3)
        active_rfm_threshold = segment_config.get('active_rfm_threshold', 7)
        lapsed_rfm_threshold = segment_config.get('lapsed_rfm_threshold', 7)
        selected_campaigns = segment_config.get('selected_campaigns', [])
        top_n_midmajor = segment_config.get('top_n_midmajor', 4000)
        
        # Define priority segment function
        def assign_priority_segment(row):
            # Check for campaign-based segments if campaigns were selected
            campaign_match = False
            for campaign in selected_campaigns:
                campaign_col = f'{campaign}_Flag'
                if campaign_col in row and row[campaign_col] == 1:
                    campaign_match = True
            
            # Apply segmentation logic
            if row['M_Score'] >= mm_monetary_threshold and row['R_Score'] >= mm_recency_threshold:
                return 'Mid Major'
            elif row['F_Score'] <= new_freq_threshold and row['R_Score'] >= new_recency_threshold:
                return 'New OT'
            elif row['R_Score'] >= active_recency_threshold and campaign_match:
                return 'Active Campaign'
            elif row['R_Score'] >= active_recency_threshold and row['RFM_Sum'] >= active_rfm_threshold:
                return 'Active - High RFM'
            elif row['R_Score'] < active_recency_threshold and campaign_match:
                return 'Lapsed Campaign'
            elif row['R_Score'] < active_recency_threshold and row['RFM_Sum'] >= lapsed_rfm_threshold:
                return 'Lapsed'
            else:
                return 'Other'
        
        # Apply priority segmentation
        df['Priority_Segment'] = df.apply(assign_priority_segment, axis=1)
        
        # Select top donors by average donation for Mid Major variable segment
        top_donors = df[df['M_Score'] >= mm_monetary_threshold].nlargest(top_n_midmajor, 'Average_Donation')['donor_id'].tolist()
        
        # Define variable segment function
        def assign_variable_segment(row):
            if row['Priority_Segment'] == 'Other':
                return None
            elif row['donor_id'] in top_donors:
                return 'Mid Major'
            elif row['F_Score'] <= new_freq_threshold and row['R_Score'] >= new_recency_threshold:
                return 'New Donor'
            elif row['R_Score'] >= active_recency_threshold:
                return 'Active'
            else:
                return 'Lapsed'
        
        # Apply variable segmentation
        df['Variable_Segment'] = df.apply(assign_variable_segment, axis=1)
        
        return df
        
    except Exception as e:
        st.error(f"Error during segmentation: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return donor_summary


def calculate_ask_amounts(donor_data, ask_config):
    """
    Calculates ask amounts for donors
    
    Parameters:
    -----------
    donor_data : pandas.DataFrame
        Donor data with segmentation applied
    ask_config : dict
        Configuration for ask amounts
        
    Returns:
    --------
    pandas.DataFrame
        Donor data with ask amounts and strings
    """
    try:
        # Create a copy to avoid modifying the original
        df = donor_data.copy()
        
        # Filter to only include donors with priority segments other than 'Other'
        donor_summary_filtered = df[df['Priority_Segment'] != 'Other'].copy()
        
        # Get ask method and parameters
        ask_method = ask_config.get('ask_method', 'Multiplier of Average Donation')
        
        # Define calculation function based on method
        if ask_method == "Multiplier of Average Donation":
            ask1_multiplier = ask_config.get('ask1_multiplier', 1.0)
            ask2_multiplier = ask_config.get('ask2_multiplier', 1.5)
            ask3_multiplier = ask_config.get('ask3_multiplier', 2.0)
            ask4_multiplier = ask_config.get('ask4_multiplier', 2.5)
            round_to = ask_config.get('round_to', 5)
            min_ask = ask_config.get('min_ask', 20)
            max_ask_avg = ask_config.get('max_ask_avg', 1000)
            
            def calculate_ask_values(avg_donation):
                if pd.isna(avg_donation) or avg_donation > max_ask_avg:
                    return None, None, None, None
                
                ask1 = max(min_ask, round(avg_donation * ask1_multiplier / round_to) * round_to)
                ask2 = max(min_ask, round(avg_donation * ask2_multiplier / round_to) * round_to)
                ask3 = max(min_ask, round(avg_donation * ask3_multiplier / round_to) * round_to)
                ask4 = max(min_ask, round(avg_donation * ask4_multiplier / round_to) * round_to)
                
                return ask1, ask2, ask3, ask4
                
        elif ask_method == "Fixed Tiers":
            tier1 = ask_config.get('tier1', 25)
            tier2 = ask_config.get('tier2', 50)
            tier3 = ask_config.get('tier3', 100)
            tier4 = ask_config.get('tier4', 250)
            
            def calculate_ask_values(avg_donation):
                return tier1, tier2, tier3, tier4
                
        elif ask_method == "Need-Based":
            base_unit = ask_config.get('base_unit', 80)
            
            def calculate_ask_values(avg_donation):
                ask1 = base_unit
                ask2 = base_unit * 2
                ask3 = base_unit * 3
                ask4 = base_unit * 5
                return ask1, ask2, ask3, ask4
        
        # Apply ask amount calculations
        donor_summary_filtered[['Ask_1', 'Ask_2', 'Ask_3', 'Ask_4']] = donor_summary_filtered['Average_Donation'].apply(
            lambda x: pd.Series(calculate_ask_values(x))
        )
        
        # Generate ask strings
        english_template = ask_config.get('english_template', "${amount} to provide care and community to seniors like Hector.")
        french_template = ask_config.get('french_template', "{amount} $ pour offrir des soins et une communauté aux aînés comme Hector.")
        open_ask_english = ask_config.get('open_ask_english', "$______ to help as many seniors like Hector as possible!")
        open_ask_french = ask_config.get('open_ask_french', "$______ pour aider autant d'aînés comme Hector que possible!")
        monthly_ask_english = ask_config.get('monthly_ask_english', "I'd like to give $_____ every month to provide care and community to seniors in Montreal.")
        monthly_ask_french = ask_config.get('monthly_ask_french', "Je veux donner $_____ chaque mois pour offrir des soins et une communauté aux aînés à Montréal.")
        
        def generate_ask_string(amount, language):
            if pd.isna(amount):
                return ""
            amount = int(amount)
            if language == "Francais" or language == "French":
                return french_template.replace("{amount}", str(amount))
            return english_template.replace("{amount}", str(amount))
        
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
            if col in donor_summary_filtered.columns:
                language_col = col
                break
        
        if language_col:
            donor_summary_filtered['is_french'] = donor_summary_filtered[language_col].apply(is_french)
            
            donor_summary_filtered['Ask_1_String'] = donor_summary_filtered.apply(
                lambda x: generate_ask_string(x['Ask_1'], 'Francais' if x['is_french'] else 'English'), axis=1
            )
            donor_summary_filtered['Ask_2_String'] = donor_summary_filtered.apply(
                lambda x: generate_ask_string(x['Ask_2'], 'Francais' if x['is_french'] else 'English'), axis=1
            )
            donor_summary_filtered['Ask_3_String'] = donor_summary_filtered.apply(
                lambda x: generate_ask_string(x['Ask_3'], 'Francais' if x['is_french'] else 'English'), axis=1
            )
            donor_summary_filtered['Ask_4_String'] = donor_summary_filtered.apply(
                lambda x: generate_ask_string(x['Ask_4'], 'Francais' if x['is_french'] else 'English'), axis=1
            )
            
            # Add open ask
            donor_summary_filtered['Ask_5'] = ""
            donor_summary_filtered['Ask_5_String'] = donor_summary_filtered['is_french'].apply(
                lambda x: open_ask_french if x else open_ask_english
            )
            
            # Add monthly ask
            donor_summary_filtered['Ask_6'] = ""
            donor_summary_filtered['Ask_6_String'] = donor_summary_filtered['is_french'].apply(
                lambda x: monthly_ask_french if x else monthly_ask_english
            )
        else:
            # Default to English if no language column is found
            for ask_num in range(1, 5):
                donor_summary_filtered[f'Ask_{ask_num}_String'] = donor_summary_filtered[f'Ask_{ask_num}'].apply(
                    lambda x: generate_ask_string(x, 'English')
                )
            
            # Add open ask and monthly ask
            donor_summary_filtered['Ask_5'] = ""
            donor_summary_filtered['Ask_5_String'] = open_ask_english
            
            donor_summary_filtered['Ask_6'] = ""
            donor_summary_filtered['Ask_6_String'] = monthly_ask_english
        
        return donor_summary_filtered
        
    except Exception as e:
        st.error(f"Error during ask amount calculation: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return donor_data


def generate_export_file(donor_data, selected_columns, file_format="Excel", include_stats=True):
    """
    Generates a downloadable export file
    
    Parameters:
    -----------
    donor_data : pandas.DataFrame
        Donor data with segmentation and ask amounts
    selected_columns : list
        List of columns to include in export
    file_format : str
        "Excel" or "CSV"
    include_stats : bool
        Whether to include segment statistics
        
    Returns:
    --------
    tuple
        (file_data, download_link, file_extension)
    """
    try:
        # Filter columns
        export_data = donor_data[selected_columns].copy()
        
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
                        'Monetary': 'sum',
                        'Average_Donation': 'mean',
                        'Frequency': 'mean',
                        'Recency': 'mean',
                        'R_Score': 'mean',
                        'F_Score': 'mean',
                        'M_Score': 'mean',
                        'RFM_Sum': 'mean'
                    }).reset_index()
                    
                    segment_stats.to_excel(writer, sheet_name='Segment Statistics', index=False)
                    
                    # Variable Segment Statistics
                    var_segment_stats = donor_data.groupby('Variable_Segment').agg({
                        'donor_id': 'count',
                        'Monetary': 'sum',
                        'Average_Donation': 'mean'
                    }).reset_index()
                    
                    var_segment_stats.to_excel(writer, sheet_name='Variable Segments', index=False)
                    
                    # Ask Amount Statistics
                    if all(col in donor_data.columns for col in ['Ask_1', 'Ask_2', 'Ask_3', 'Ask_4']):
                        ask_stats = donor_data.groupby('Priority_Segment')[['Ask_1', 'Ask_2', 'Ask_3', 'Ask_4']].mean()
                        ask_stats.to_excel(writer, sheet_name='Ask Amounts', index=True)
            
            # Get the Excel data
            buffer.seek(0)
            file_data = buffer.read()
            
            # Create a download link
            b64 = base64.b64encode(file_data).decode()
            download_link = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="donor_segmentation.xlsx" class="btn">Download Excel File</a>'
            file_extension = "xlsx"
            
        else:  # CSV format
            # Convert to CSV
            csv = export_data.to_csv(index=False)
            file_data = csv.encode()
            
            # Create a download link
            b64 = base64.b64encode(file_data).decode()
            download_link = f'<a href="data:file/csv;base64,{b64}" download="donor_segmentation.csv" class="btn">Download CSV File</a>'
            file_extension = "csv"
            
            # If stats were requested, also generate a stats CSV
            if include_stats:
                stats_buffer = io.BytesIO()
                segment_stats = donor_data.groupby('Priority_Segment').agg({
                    'donor_id': 'count',
                    'Monetary': 'sum',
                    'Average_Donation': 'mean',
                    'Frequency': 'mean',
                    'Recency': 'mean',
                    'R_Score': 'mean',
                    'F_Score': 'mean',
                    'M_Score': 'mean',
                    'RFM_Sum': 'mean'
                }).reset_index()
                
                stats_csv = segment_stats.to_csv(index=False)
                stats_b64 = base64.b64encode(stats_csv.encode()).decode()
                
                # Append stats download link
                download_link += f'<br><br><a href="data:file/csv;base64,{stats_b64}" download="segment_statistics.csv" class="btn">Download Segment Statistics CSV</a>'
        
        return file_data, download_link, file_extension
        
    except Exception as e:
        st.error(f"Error generating export file: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, f"Error: {str(e)}", None


def display_segment_visualizations(donor_data):
    """
    Displays visualizations for donor segmentation
    
    Parameters:
    -----------
    donor_data : pandas.DataFrame
        Donor data with segmentation applied
    """
    try:
        # Priority segment distribution
        st.markdown("##### Priority Segment Distribution")
        
        priority_dist = donor_data['Priority_Segment'].value_counts().reset_index()
        priority_dist.columns = ['Segment', 'Count']
        
        fig = px.pie(
            priority_dist, 
            values='Count', 
            names='Segment',
            title='Priority Segment Distribution',
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Variable segment distribution if exists
        if 'Variable_Segment' in donor_data.columns:
            st.markdown("##### Variable Segment Distribution")
            
            variable_dist = donor_data['Variable_Segment'].value_counts().reset_index()
            variable_dist.columns = ['Segment', 'Count']
            
            fig = px.bar(
                variable_dist, 
                x='Segment', 
                y='Count',
                title='Variable Segment Distribution',
                color='Segment',
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Segment monetary analysis
        st.markdown("##### Segment Monetary Analysis")
        
        segment_monetary = donor_data.groupby('Priority_Segment').agg({
            'donor_id': 'count',
            'Monetary': 'sum',
            'Average_Donation': 'mean'
        }).reset_index()
        
        segment_monetary.columns = ['Segment', 'Donor Count', 'Total Donations', 'Average Donation']
        
        fig = px.bar(
            segment_monetary,
            x='Segment',
            y='Total Donations',
            title='Total Donations by Priority Segment',
            color='Segment',
            text='Donor Count'
        )
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        # RFM Distribution by Segment
        st.markdown("##### RFM Distribution by Segment")
        
        fig = px.box(
            donor_data,
            x='Priority_Segment',
            y='RFM_Sum',
            title='RFM Sum Distribution by Priority Segment',
            color='Priority_Segment'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # RF Heatmap by Segment
        if len(donor_data['Priority_Segment'].unique()) <= 6:  # Only show for a reasonable number of segments
            st.markdown("##### Recency-Frequency Matrix by Segment")
            
            for segment in donor_data['Priority_Segment'].unique():
                if segment == 'Other':
                    continue
                    
                segment_data = donor_data[donor_data['Priority_Segment'] == segment]
                
                rf_counts = segment_data.groupby(['R_Score', 'F_Score']).size().reset_index(name='Count')
                
                # Check if we have enough data for a meaningful heatmap
                if len(rf_counts) > 3:
                    rf_pivot = rf_counts.pivot(index='R_Score', columns='F_Score', values='Count').fillna(0)
                    
                    fig = px.imshow(
                        rf_pivot,
                        labels=dict(x="Frequency Score", y="Recency Score", color="Donor Count"),
                        x=rf_pivot.columns,
                        y=rf_pivot.index,
                        color_continuous_scale='Blues',
                        title=f'Recency-Frequency Matrix: {segment}',
                        text_auto=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error displaying segment visualizations: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

# Helper function to get segment metrics
def get_segment_metrics(donor_data):
    """
    Calculates key metrics for each segment
    
    Parameters:
    -----------
    donor_data : pandas.DataFrame
        Donor data with segmentation applied
        
    Returns:
    --------
    pandas.DataFrame
        Metrics by segment
    """
    try:
        metrics = donor_data.groupby('Priority_Segment').agg({
            'donor_id': 'count',
            'Monetary': 'sum',
            'Average_Donation': 'mean',
            'Frequency': 'mean',
            'Recency': 'mean',
            'R_Score': 'mean',
            'F_Score': 'mean',
            'M_Score': 'mean',
            'RFM_Sum': 'mean'
        }).reset_index()
        
        metrics.columns = [
            'Segment', 'Donor Count', 'Total Donations', 'Avg Donation', 
            'Avg Frequency', 'Avg Recency Days', 'Avg R', 'Avg F', 
            'Avg M', 'Avg RFM'
        ]
        
        # Format metrics
        metrics['Total Donations'] = metrics['Total Donations'].map('${:,.2f}'.format)
        metrics['Avg Donation'] = metrics['Avg Donation'].map('${:,.2f}'.format)
        metrics['Avg Frequency'] = metrics['Avg Frequency'].map('{:,.1f}'.format)
        metrics['Avg Recency Days'] = metrics['Avg Recency Days'].map('{:,.0f}'.format)
        metrics['Avg R'] = metrics['Avg R'].map('{:,.1f}'.format)
        metrics['Avg F'] = metrics['Avg F'].map('{:,.1f}'.format)
        metrics['Avg M'] = metrics['Avg M'].map('{:,.1f}'.format)
        metrics['Avg RFM'] = metrics['Avg RFM'].map('{:,.1f}'.format)
        
        return metrics
        
    except Exception as e:
        st.error(f"Error calculating segment metrics: {str(e)}")
        return pd.DataFrame()