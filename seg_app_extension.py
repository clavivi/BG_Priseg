# seg_app_extension.py
# Import this after your main app to add segmentation and export functionality

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import io
import base64
from segmentation_utils import (
    apply_segmentation,
    calculate_ask_amounts,
    generate_export_file,
    display_segment_visualizations,
    get_segment_metrics
)

def add_segmentation_tab():
    """
    Adds the segmentation tab functionality to your app
    """
    # Check if we have donor summary data to work with
    if 'donor_summary' not in st.session_state or st.session_state.donor_summary is None:
        st.warning("Please complete data preprocessing in the previous step first.")
        return
    
    st.markdown('<div class="section-header">Donor Segmentation</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Configure segmentation criteria to categorize donors into priority and variable segments. 
    These segments will be used for targeted mailing and ask amounts.
    </div>
    """, unsafe_allow_html=True)
    
    # Segmentation form
    with st.form("segmentation_form"):
        st.markdown('<div class="subsection-header">Priority Segments Configuration</div>', unsafe_allow_html=True)
        
        # Define segment criteria
        st.markdown("##### Mid Major Segment")
        mm_monetary_threshold = st.slider("Monetary Score Threshold for Mid Major", 1, 5, 4)
        mm_recency_threshold = st.slider("Minimum Recency Score for Mid Major", 1, 5, 2)
        
        st.markdown("##### New Donors")
        new_freq_threshold = st.slider("Frequency Score for New Donors", 1, 3, 1)
        new_recency_threshold = st.slider("Minimum Recency Score for New Donors", 1, 5, 3)
        
        st.markdown("##### Active Segments")
        active_recency_threshold = st.slider("Minimum Recency Score for Active Donors", 1, 5, 3)
        active_rfm_threshold = st.slider("Minimum RFM Sum for Active-High RFM", 3, 15, 7)
        
        st.markdown("##### Lapsed Segments")
        lapsed_rfm_threshold = st.slider("Minimum RFM Sum for Lapsed Donors", 3, 15, 7)
        
        # Campaign based segmentation
        st.markdown("##### Campaign-based Segmentation")
        if 'processed_data' in st.session_state and 'campaign_name' in st.session_state.processed_data.columns:
            campaign_names = st.session_state.processed_data['campaign_name'].dropna().unique()
            selected_campaigns = st.multiselect(
                "Select campaigns for specific segments", 
                options=campaign_names,
                default=campaign_names[:1] if len(campaign_names) > 0 else []
            )
        else:
            st.info("No campaign names found in data. Using generic segmentation only.")
            selected_campaigns = []
        
        # Variable segment configuration
        st.markdown('<div class="subsection-header">Variable Segments Configuration</div>', unsafe_allow_html=True)
        
        # Number of top donors for Mid Major variable segment
        top_n_midmajor = st.number_input("Number of top donors (by Average Donation) for Mid Major variable segment", min_value=100, max_value=10000, value=4000, step=100)
        
        # Ask amount configuration
        st.markdown('<div class="subsection-header">Ask Amount Configuration</div>', unsafe_allow_html=True)
        
        ask_method = st.radio(
            "Ask Amount Calculation Method",
            options=["Multiplier of Average Donation", "Fixed Tiers", "Need-Based"],
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
        
        open_ask_english = st.text_input(
            "Open Ask (English)",
            value="$______ to help as many seniors like Hector as possible!"
        )
        
        open_ask_french = st.text_input(
            "Open Ask (French)",
            value="$______ pour aider autant d'aînés comme Hector que possible!"
        )
        
        monthly_ask_english = st.text_input(
            "Monthly Ask (English)",
            value="I'd like to give $_____ every month to provide care and community to seniors in Montreal."
        )
        
        monthly_ask_french = st.text_input(
            "Monthly Ask (French)",
            value="Je veux donner $_____ chaque mois pour offrir des soins et une communauté aux aînés à Montréal."
        )
        
        # Submit button
        submit_button = st.form_submit_button("Apply Segmentation")
        
        if submit_button:
            try:
                with st.spinner("Applying segmentation..."):
                    # Create segmentation config
                    segment_config = {
                        'mm_monetary_threshold': mm_monetary_threshold,
                        'mm_recency_threshold': mm_recency_threshold,
                        'new_freq_threshold': new_freq_threshold,
                        'new_recency_threshold': new_recency_threshold,
                        'active_recency_threshold': active_recency_threshold,
                        'active_rfm_threshold': active_rfm_threshold,
                        'lapsed_rfm_threshold': lapsed_rfm_threshold,
                        'selected_campaigns': selected_campaigns,
                        'top_n_midmajor': top_n_midmajor
                    }
                    
                    # Create ask config
                    ask_config = {
                        'ask_method': ask_method,
                        'english_template': english_template,
                        'french_template': french_template,
                        'open_ask_english': open_ask_english,
                        'open_ask_french': open_ask_french,
                        'monthly_ask_english': monthly_ask_english,
                        'monthly_ask_french': monthly_ask_french
                    }
                    
                    # Add method-specific parameters
                    if ask_method == "Multiplier of Average Donation":
                        ask_config.update({
                            'ask1_multiplier': ask1_multiplier,
                            'ask2_multiplier': ask2_multiplier,
                            'ask3_multiplier': ask3_multiplier,
                            'ask4_multiplier': ask4_multiplier,
                            'round_to': round_to,
                            'min_ask': min_ask,
                            'max_ask_avg': max_ask_avg
                        })
                    elif ask_method == "Fixed Tiers":
                        ask_config.update({
                            'tier1': tier1,
                            'tier2': tier2,
                            'tier3': tier3,
                            'tier4': tier4
                        })
                    elif ask_method == "Need-Based":
                        ask_config.update({
                            'base_unit': base_unit,
                            'base_unit_desc': base_unit_desc
                        })
                    
                    # Apply segmentation
                    segmented_data = apply_segmentation(st.session_state.donor_summary, segment_config)
                    
                    # Calculate ask amounts
                    donor_data_with_asks = calculate_ask_amounts(segmented_data, ask_config)
                    
                    # Store the results
                    st.session_state.donor_summary_filtered = donor_data_with_asks
                    st.session_state.segmentation_done = True
                    
                    st.success("Segmentation and ask amounts calculated successfully!")
            
            except Exception as e:
                st.error(f"Error during segmentation: {e}")
                import traceback
                st.error(traceback.format_exc())
    
    # Display results if segmentation is done
    if 'segmentation_done' in st.session_state and st.session_state.segmentation_done:
        st.markdown('<div class="success-box">Segmentation completed successfully!</div>', unsafe_allow_html=True)
        
        # Button to go to export page
        if st.button("Go to Final Export"):
            st.session_state.current_step = 5
            st.rerun()
        
        # Display segmentation results
        if 'donor_summary_filtered' in st.session_state:
            donor_summary_filtered = st.session_state.donor_summary_filtered
            
            st.markdown('<div class="subsection-header">Segmentation Results</div>', unsafe_allow_html=True)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Donors in Segments", f"{len(donor_summary_filtered):,}")
            col2.metric("Priority Segments", f"{donor_summary_filtered['Priority_Segment'].nunique()}")
            
            if 'Variable_Segment' in donor_summary_filtered.columns:
                col3.metric("Variable Segments", f"{donor_summary_filtered['Variable_Segment'].dropna().nunique()}")
            
            # Display segment metrics
            st.markdown("##### Segment Metrics")
            segment_metrics = get_segment_metrics(donor_summary_filtered)
            st.dataframe(segment_metrics)
            
            # Display visualizations
            display_segment_visualizations(donor_summary_filtered)
            
            # Ask amount preview if available
            if 'Ask_1' in donor_summary_filtered.columns:
                st.markdown("##### Ask Amount Preview")
                
                # Sample of ask amounts
                ask_sample = donor_summary_filtered[['Priority_Segment', 'Average_Donation', 'Ask_1', 'Ask_2', 'Ask_3', 'Ask_4']].head(10)
                st.dataframe(ask_sample)
                
                # Average ask by segment
                ask_avg = donor_summary_filtered.groupby('Priority_Segment')[['Ask_1', 'Ask_2', 'Ask_3', 'Ask_4']].mean().reset_index()
                
                # Format for display
                for col in ['Ask_1', 'Ask_2', 'Ask_3', 'Ask_4']:
                    ask_avg[col] = ask_avg[col].apply(lambda x: f"${x:.2f}")
                
                st.markdown("##### Average Ask by Segment")
                st.dataframe(ask_avg)


def add_export_tab():
    """
    Adds the export tab functionality to your app
    """
    # Check if segmentation is done
    if 'segmentation_done' not in st.session_state or not st.session_state.segmentation_done:
        st.warning("Please complete segmentation in the previous step first.")
        return
    
    st.markdown('<div class="section-header">Final Export</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Review the final segmented donor list and download it for your mailing campaign.
    </div>
    """, unsafe_allow_html=True)
    
    # Get the final donor data
    if 'donor_summary_filtered' not in st.session_state:
        st.error("Donor data not found. Please return to the segmentation step.")
        return
        
    donor_data = st.session_state.donor_summary_filtered
    
    # Display summary statistics
    st.markdown('<div class="subsection-header">Export Summary</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Donors", f"{len(donor_data):,}")
    col2.metric("Priority Segments", f"{donor_data['Priority_Segment'].nunique()}")
    
    if 'Variable_Segment' in donor_data.columns:
        col3.metric("Variable Segments", f"{donor_data['Variable_Segment'].dropna().nunique()}")
    
    # Summary by segment
    st.markdown("##### Donor Count by Segment")
    segment_summary = donor_data.groupby('Priority_Segment').agg({
        'donor_id': 'count',
        'Monetary': 'sum',
        'Average_Donation': 'mean'
    }).reset_index()
    
    segment_summary.columns = ['Priority Segment', 'Donor Count', 'Total Donations', 'Average Donation']
    segment_summary['Total Donations'] = segment_summary['Total Donations'].apply(lambda x: f"${x:,.2f}")
    segment_summary['Average Donation'] = segment_summary['Average_Donation'].apply(lambda x: f"${x:,.2f}")
    
    st.dataframe(segment_summary)
    
    # Ask amount preview
    ask_cols = [col for col in donor_data.columns if col.startswith('Ask_') and not col.endswith('_String')]
    if ask_cols:
        st.markdown("##### Ask Amount Preview")
        ask_preview = donor_data.groupby('Priority_Segment')[ask_cols].mean().reset_index()
        
        # Format currency
        for col in ask_cols:
            if col in ask_preview.columns:
                ask_preview[col] = ask_preview[col].apply(lambda x: f"${x:,.2f}" if not pd.isna(x) else "N/A")
        
        st.dataframe(ask_preview)
    
    # Preview of the export data
    st.markdown('<div class="subsection-header">Export Data Preview</div>', unsafe_allow_html=True)
    
    # Let user select columns to include in export
    available_columns = donor_data.columns.tolist()
    
    # Define column categories for easier selection
    contact_cols = [col for col in available_columns if any(x in col.lower() for x in ['donor', 'postal', 'email', 'mobile', 'phone', 'address'])]
    segment_cols = [col for col in available_columns if any(x in col.lower() for x in ['segment', 'flag'])]
    analytics_cols = [col for col in available_columns if any(x in col.lower() for x in ['score', 'rfm', 'recency', 'frequency', 'monetary'])]
    ask_cols = [col for col in available_columns if 'ask' in col.lower()]
    
    # Create column selection with expandable sections
    with st.expander("Select Columns for Export", expanded=True):
        st.markdown("##### Required Columns")
        st.info("donor_id is always included")
        
        st.markdown("##### Contact Information")
        selected_contact = st.multiselect("Select contact columns", options=contact_cols, default=contact_cols)
        
        st.markdown("##### Segmentation")
        selected_segment = st.multiselect("Select segmentation columns", options=segment_cols, default=['Priority_Segment', 'Variable_Segment'])
        
        st.markdown("##### Analytics")
        selected_analytics = st.multiselect("Select analytics columns", options=analytics_cols, default=[])
        
        st.markdown("##### Ask Amounts")
        selected_ask = st.multiselect("Select ask amount columns", options=ask_cols, default=[col for col in ask_cols if 'String' in col])
        
        # Combine all selected columns
        selected_columns = ['donor_id'] + selected_contact + selected_segment + selected_analytics + selected_ask
        
        # Remove duplicates while preserving order
        seen = set()
        selected_columns = [x for x in selected_columns if not (x in seen or seen.add(x))]
    
    # Preview of export data
    export_data = donor_data[selected_columns].copy()
    st.dataframe(export_data.head(10))
    
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
                file_data, download_link, file_extension = generate_export_file(
                    donor_data=donor_data,
                    selected_columns=selected_columns,
                    file_format=file_format,
                    include_stats=include_stats
                )
                
                if file_data is not None:
                    st.markdown(download_link, unsafe_allow_html=True)
                    st.success(f"Export file generated successfully! Click the link above to download.")
        
        except Exception as e:
            st.error(f"Error generating export file: {e}")
            import traceback
            st.error(traceback.format_exc())


# Function to integrate the extension with the main app
def integrate_extension():
    """
    Integrates the extension functions with the main app
    """
    # Create a placeholder for tab contents
    if st.session_state.current_step == 4:
        add_segmentation_tab()
    elif st.session_state.current_step == 5:
        add_export_tab()


# Call this function at the end of your main app
if __name__ == "__main__":
    integrate_extension()