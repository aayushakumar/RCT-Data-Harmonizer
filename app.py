

import streamlit as st
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
import matplotlib.pyplot as plt
import altair as alt
import os
import io
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="RCT Data Harmonizer",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_data(uploaded_files) -> Dict[str, pd.DataFrame]:
    """
    Load multiple CSV files into a dictionary of DataFrames.
    
    Args:
        uploaded_files: List of uploaded file objects from Streamlit
    
    Returns:
        Dict mapping filename to DataFrame
    """
    dataframes = {}
    for file in uploaded_files:
        try:
            df = pd.read_csv(file)
            dataframes[file.name] = df
            st.success(f"✅ Loaded {file.name}: {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            st.error(f"❌ Error loading {file.name}: {str(e)}")
    return dataframes

def get_canonical_variables() -> Dict[str, List[str]]:
    """
    Define canonical variable names and their common variations.
    
    Returns:
        Dictionary mapping canonical names to lists of variations
    """
    return {
        "participant_id": ["id", "subject_id", "participant", "subj_id", "patient_id", "pid"],
        "age": ["age", "age_years", "participant_age", "subject_age"],
        "gender": ["gender", "sex", "male_female", "m_f", "gender_mf"],
        "baseline_weight": ["weight", "baseline_weight", "weight_kg", "wt", "body_weight"],
        "follow_up_weight": ["follow_weight", "followup_weight", "weight_followup", "final_weight"],
        "baseline_bmi": ["bmi", "baseline_bmi", "bmi_baseline", "body_mass_index"],
        "follow_up_bmi": ["follow_bmi", "followup_bmi", "bmi_followup", "final_bmi"],
        "physical_activity_change": ["pa_change", "activity_change", "pa_minutes", "phys_act", "activity_mins"],
        "diet_adherence": ["diet_score", "adherence", "diet_compliance", "nutrition_score"],
        "systolic_bp": ["sbp", "systolic", "sys_bp", "blood_pressure_sys"],
        "diastolic_bp": ["dbp", "diastolic", "dia_bp", "blood_pressure_dia"],
        "treatment_group": ["group", "treatment", "arm", "intervention", "condition"],
        "dropout": ["dropout", "withdrawn", "completed", "status", "drop_out"]
    }

def match_columns(dataframes: Dict[str, pd.DataFrame], canonical_vars: Dict[str, List[str]], 
                 threshold: float = 70.0) -> Dict[str, Dict[str, Tuple[str, float]]]:
    """
    Fuzzy match column names to canonical variables across all DataFrames.
    
    Args:
        dataframes: Dictionary of DataFrames
        canonical_vars: Dictionary of canonical variable mappings
        threshold: Minimum match score threshold
    
    Returns:
        Nested dictionary: {filename: {original_col: (canonical_name, score)}}
    """
    all_matches = {}
    
    for filename, df in dataframes.items():
        matches = {}
        
        for col in df.columns:
            best_match = None
            best_score = 0
            best_canonical = None
            
            # Try to match against all canonical variables
            for canonical_name, variations in canonical_vars.items():
                # Check exact matches first
                if col.lower() in [v.lower() for v in variations]:
                    best_canonical = canonical_name
                    best_score = 100.0
                    break
                
                # Fuzzy match against variations
                for variation in variations:
                    score = fuzz.ratio(col.lower(), variation.lower())
                    if score > best_score:
                        best_score = score
                        best_canonical = canonical_name
            
            # Only include matches above threshold
            if best_score >= threshold:
                matches[col] = (best_canonical, best_score)
            else:
                matches[col] = ("unmapped", best_score)
        
        all_matches[filename] = matches
    
    return all_matches

def create_mapping_interface(dataframes: Dict[str, pd.DataFrame], 
                           matches: Dict[str, Dict[str, Tuple[str, float]]],
                           canonical_vars: Dict[str, List[str]]) -> Dict[str, Dict[str, str]]:
    """
    Create an interactive interface for reviewing and editing column mappings.
    
    Args:
        dataframes: Dictionary of DataFrames
        matches: Initial fuzzy matches
        canonical_vars: Available canonical variables
    
    Returns:
        Final mapping dictionary: {filename: {original_col: canonical_name}}
    """
    st.subheader("📋 Review and Edit Column Mappings")
    
    canonical_options = ["unmapped"] + list(canonical_vars.keys())
    final_mappings = {}
    
    for filename, df in dataframes.items():
        st.markdown(f"**{filename}**")
        
        # Create editable mapping table
        mapping_data = []
        file_matches = matches.get(filename, {})
        
        for col in df.columns:
            original_name = col
            matched_canonical, score = file_matches.get(col, ("unmapped", 0))
            
            mapping_data.append({
                "Original Name": original_name,
                "Matched Canonical": matched_canonical,
                "Match Score": f"{score:.1f}%"
            })
        
        # Display as DataFrame for review
        mapping_df = pd.DataFrame(mapping_data)
        st.dataframe(mapping_df, use_container_width=True)
        
        # Create override interface
        file_mapping = {}
        st.markdown("**Override Mappings (if needed):**")
        
        cols = st.columns(min(3, len(df.columns)))
        for i, col in enumerate(df.columns):
            with cols[i % 3]:
                current_match = file_matches.get(col, ("unmapped", 0))[0]
                override = st.selectbox(
                    f"{col}",
                    options=canonical_options,
                    index=canonical_options.index(current_match) if current_match in canonical_options else 0,
                    key=f"mapping_{filename}_{col}"
                )
                file_mapping[col] = override
        
        final_mappings[filename] = file_mapping
        st.divider()
    
    return final_mappings

def harmonize_dataframes(dataframes: Dict[str, pd.DataFrame], 
                        mappings: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """
    Apply harmonization mappings and merge DataFrames.
    
    Args:
        dataframes: Original DataFrames
        mappings: Column mapping dictionaries
    
    Returns:
        Merged and harmonized DataFrame
    """
    harmonized_dfs = []
    
    for filename, df in dataframes.items():
        # Create a copy and add source column
        df_harm = df.copy()
        df_harm['_source_file'] = filename
        
        # Apply column mappings
        mapping = mappings[filename]
        rename_dict = {orig: canonical for orig, canonical in mapping.items() 
                      if canonical != "unmapped"}
        
        df_harm = df_harm.rename(columns=rename_dict)
        
        # Drop unmapped columns to avoid clutter
        unmapped_cols = [col for col, canonical in mapping.items() if canonical == "unmapped"]
        df_harm = df_harm.drop(columns=unmapped_cols, errors='ignore')
        
        harmonized_dfs.append(df_harm)
    
    # Merge all DataFrames
    if not harmonized_dfs:
        return pd.DataFrame()
    
    # Check if participant_id exists for merging
    has_participant_id = all('participant_id' in df.columns for df in harmonized_dfs)
    
    if has_participant_id:
        # Merge on participant_id with proper suffix handling
        merged_df = harmonized_dfs[0]
        
        for i, df in enumerate(harmonized_dfs[1:], 1):
            # Get overlapping columns (except participant_id and _source_file)
            overlap_cols = set(merged_df.columns) & set(df.columns)
            overlap_cols.discard('participant_id')
            overlap_cols.discard('_source_file')
            
            # Create suffixes based on source files
            left_suffix = f'_file{i-1}' if len(overlap_cols) > 0 else ''
            right_suffix = f'_file{i}' if len(overlap_cols) > 0 else ''
            
            merged_df = pd.merge(
                merged_df, df, 
                on='participant_id', 
                how='outer', 
                suffixes=(left_suffix, right_suffix)
            )
    else:
        # Concatenate if no common ID - add file identifier to avoid duplicates
        for i, df in enumerate(harmonized_dfs):
            # Add file index to column names (except _source_file)
            if i > 0:  # Don't rename first dataframe columns
                cols_to_rename = [col for col in df.columns if col != '_source_file']
                rename_dict = {col: f"{col}_file{i}" for col in cols_to_rename}
                df = df.rename(columns=rename_dict)
                harmonized_dfs[i] = df
        
        merged_df = pd.concat(harmonized_dfs, ignore_index=True, sort=False)
    
    # Clean up any remaining duplicate columns
    if merged_df.columns.duplicated().any():
        # Keep first occurrence of duplicated columns
        cols = pd.Series(merged_df.columns)
        for dup in cols[cols.duplicated()].unique():
            cols[cols[cols == dup].index[1:]] = [f"{dup}_dup{i}" for i in range(1, sum(cols == dup))]
        merged_df.columns = cols
    
    return merged_df

def create_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for numeric columns.
    
    Args:
        df: Harmonized DataFrame
    
    Returns:
        DataFrame with summary statistics
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if not col.startswith('_')]
    
    if len(numeric_cols) == 0:
        return pd.DataFrame()
    
    summary = df[numeric_cols].describe().T
    summary['missing_count'] = df[numeric_cols].isnull().sum()
    summary['missing_percent'] = (summary['missing_count'] / len(df) * 100).round(2)
    
    # Add column info
    summary['data_type'] = [str(df[col].dtype) for col in numeric_cols]
    
    return summary.round(2)

def plot_histograms(df: pd.DataFrame) -> None:
    """
    Create histograms for numeric variables, faceted by source file.
    
    Args:
        df: Harmonized DataFrame
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if not col.startswith('_') and 'participant_id' not in col.lower()]
    
    if len(numeric_cols) == 0:
        st.warning("No numeric columns found for plotting.")
        return
    
    # Group similar columns (handle suffixes from merging)
    base_columns = {}
    for col in numeric_cols:
        # Extract base name (remove file suffixes)
        base_name = col.split('_file')[0].split('_dup')[0]
        if base_name not in base_columns:
            base_columns[base_name] = []
        base_columns[base_name].append(col)
    
    # Create plots for each base variable (limit to first 6)
    plot_count = 0
    for base_name, cols in list(base_columns.items())[:6]:
        plot_count += 1
        
        with st.expander(f"📊 Distribution of {base_name} ({len(cols)} version{'s' if len(cols) > 1 else ''})"):
            
            if len(cols) == 1:
                # Single version - plot by source file if available
                col = cols[0]
                if '_source_file' in df.columns:
                    chart_data = df[[col, '_source_file']].dropna()
                    
                    if len(chart_data) > 0:
                        chart = alt.Chart(chart_data).mark_bar(opacity=0.7).add_selection(
                            alt.selection_interval()
                        ).encode(
                            alt.X(f'{col}:Q', bin=alt.Bin(maxbins=15)),
                            alt.Y('count()'),
                            alt.Color('_source_file:N', legend=alt.Legend(title="Source File")),
                            alt.Facet('_source_file:N', columns=2)
                        ).resolve_scale(
                            y='independent'
                        ).properties(
                            width=200,
                            height=150,
                            title=f'Distribution of {base_name} by Source File'
                        )
                        st.altair_chart(chart)
                    else:
                        st.warning(f"No data available for {col}")
                else:
                    # Simple histogram
                    valid_data = df[col].dropna()
                    if len(valid_data) > 0:
                        chart = alt.Chart(pd.DataFrame({col: valid_data})).mark_bar().encode(
                            alt.X(f'{col}:Q', bin=alt.Bin(maxbins=20)),
                            alt.Y('count()')
                        ).properties(
                            width=400,
                            height=200,
                            title=f'Distribution of {base_name}'
                        )
                        st.altair_chart(chart)
                    else:
                        st.warning(f"No valid data for {col}")
            
            else:
                # Multiple versions - show side by side
                chart_data_list = []
                for col in cols:
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        for val in col_data:
                            chart_data_list.append({
                                'value': val,
                                'version': col,
                                'variable': base_name
                            })
                
                if chart_data_list:
                    chart_df = pd.DataFrame(chart_data_list)
                    
                    chart = alt.Chart(chart_df).mark_bar(opacity=0.7).encode(
                        alt.X('value:Q', bin=alt.Bin(maxbins=15)),
                        alt.Y('count()'),
                        alt.Color('version:N', legend=alt.Legend(title="Column Version")),
                        alt.Facet('version:N', columns=2)
                    ).resolve_scale(
                        y='independent'
                    ).properties(
                        width=200,
                        height=150,
                        title=f'Distribution of {base_name} (Multiple Versions)'
                    )
                    
                    st.altair_chart(chart)
                else:
                    st.warning(f"No valid data for any version of {base_name}")

def create_export_buttons(merged_df: pd.DataFrame, 
                         mappings: Dict[str, Dict[str, str]]) -> None:
    """
    Create download buttons for harmonized data and codebook.
    
    Args:
        merged_df: Harmonized DataFrame
        mappings: Column mappings used
    """
    col1, col2 = st.columns(2)
    
    with col1:
        # Export harmonized CSV
        if not merged_df.empty:
            csv_buffer = io.StringIO()
            merged_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📥 Download Harmonized CSV",
                data=csv_data,
                file_name="harmonized_rct_data.csv",
                mime="text/csv",
                help="Download the merged and harmonized dataset"
            )
    
    with col2:
        # Export codebook
        codebook_data = []
        for filename, mapping in mappings.items():
            for original, canonical in mapping.items():
                codebook_data.append({
                    "Source File": filename,
                    "Original Name": original,
                    "Final Canonical Name": canonical
                })
        
        if codebook_data:
            codebook_df = pd.DataFrame(codebook_data)
            codebook_buffer = io.StringIO()
            codebook_df.to_csv(codebook_buffer, index=False)
            codebook_csv = codebook_buffer.getvalue()
            
            st.download_button(
                label="📋 Download Codebook Lookup",
                data=codebook_csv,
                file_name="variable_codebook.csv",
                mime="text/csv",
                help="Download the variable mapping codebook"
            )

def main():
    """Main application function."""
    
    # App header
    st.title("🔄 RCT Data Harmonizer")
    st.markdown("""
    **Harmonize variable names and merge multiple CSV files from randomized controlled trials.**
    
    This tool helps you:
    - Upload multiple CSV files from different RCT studies
    - Automatically match column names using fuzzy matching
    - Review and override mappings as needed
    - Merge datasets with harmonized variable names
    - Generate summary statistics and visualizations
    - Export cleaned data and codebook
    """)
    
    # Sidebar for file upload and controls
    st.sidebar.header("📁 File Upload")
    uploaded_files = st.sidebar.file_uploader(
        "Choose CSV files",
        type="csv",
        accept_multiple_files=True,
        help="Upload multiple CSV files to harmonize and merge"
    )
    
    if not uploaded_files:
        st.info("👆 Please upload one or more CSV files to get started.")
        return
    
    # Display uploaded files
    st.sidebar.subheader("Uploaded Files")
    file_info = []
    for file in uploaded_files:
        file_info.append({"Filename": file.name, "Size": f"{file.size:,} bytes"})
    st.sidebar.dataframe(pd.DataFrame(file_info), use_container_width=True)
    
    # Load data
    with st.spinner("Loading CSV files..."):
        dataframes = load_data(uploaded_files)
    
    if not dataframes:
        st.error("No valid CSV files were loaded. Please check your files and try again.")
        return
    
    # Get canonical variables
    canonical_vars = get_canonical_variables()
    
    # Sidebar controls for matching
    st.sidebar.subheader("⚙️ Matching Settings")
    match_threshold = st.sidebar.slider(
        "Fuzzy Match Threshold",
        min_value=50.0,
        max_value=100.0,
        value=70.0,
        step=5.0,
        help="Minimum similarity score for automatic matching"
    )
    
    # Perform fuzzy matching
    with st.spinner("Performing fuzzy matching..."):
        matches = match_columns(dataframes, canonical_vars, match_threshold)
    
    # Create mapping interface
    final_mappings = create_mapping_interface(dataframes, matches, canonical_vars)
    
    # Harmonize and merge
    if st.button("🔄 Harmonize and Merge Data", type="primary"):
        with st.spinner("Harmonizing and merging datasets..."):
            merged_df = harmonize_dataframes(dataframes, final_mappings)
        
        if merged_df.empty:
            st.error("Failed to merge datasets. Please check your data and mappings.")
            return
        
        # Store in session state
        st.session_state.merged_df = merged_df
        st.session_state.final_mappings = final_mappings
        
        st.success(f"✅ Successfully merged {len(dataframes)} datasets!")
        st.info(f"Merged dataset shape: {merged_df.shape[0]} rows × {merged_df.shape[1]} columns")
    
    # Display results if available
    if 'merged_df' in st.session_state:
        merged_df = st.session_state.merged_df
        final_mappings = st.session_state.final_mappings
        
        # Show preview
        st.subheader("👀 Data Preview")
        st.dataframe(merged_df.head(10), use_container_width=True)
        
        # Summary statistics
        st.subheader("📊 Summary Statistics")
        summary_stats = create_summary_statistics(merged_df)
        if not summary_stats.empty:
            st.dataframe(summary_stats, use_container_width=True)
        else:
            st.info("No numeric columns found for summary statistics.")
        
        # Visualizations
        st.subheader("📈 Data Visualizations")
        with st.spinner("Generating visualizations..."):
            plot_histograms(merged_df)
        
        # Export buttons
        st.subheader("💾 Export Data")
        create_export_buttons(merged_df, final_mappings)
        
        # Additional info
        with st.expander("ℹ️ Dataset Information"):
            st.write("**Column Information:**")
            col_info = []
            for col in merged_df.columns:
                dtype = str(merged_df[col].dtype)
                null_count = merged_df[col].isnull().sum()
                null_pct = (null_count / len(merged_df) * 100)
                col_info.append({
                    "Column": col,
                    "Data Type": dtype,
                    "Non-Null Count": len(merged_df) - null_count,
                    "Null %": f"{null_pct:.1f}%"
                })
            st.dataframe(pd.DataFrame(col_info), use_container_width=True)

if __name__ == "__main__":
    main()




