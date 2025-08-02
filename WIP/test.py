# requirements.txt
# streamlit==1.28.0
# pandas==2.1.0
# rapidfuzz==3.3.0
# matplotlib==3.7.2
# altair==5.1.0
# plotly==5.15.0
# scipy==1.11.0

import streamlit as st
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
import matplotlib.pyplot as plt
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import io
from typing import Dict, List, Tuple, Optional, Any
import warnings
import json
from datetime import datetime
import re
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Advanced RCT Data Harmonizer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

class DataQualityChecker:
    """Advanced data quality assessment and validation."""
    
    @staticmethod
    def assess_data_quality(df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Comprehensive data quality assessment."""
        quality_report = {
            'filename': filename,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_data': {},
            'data_types': {},
            'duplicates': 0,
            'outliers': {},
            'quality_score': 0
        }
        
        # Missing data analysis
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            quality_report['missing_data'][col] = {
                'count': int(missing_count),
                'percentage': round(missing_pct, 2)
            }
        
        # Data types analysis
        quality_report['data_types'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Duplicate analysis
        quality_report['duplicates'] = int(df.duplicated().sum())
        
        # Outlier detection for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].notna().sum() > 0:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                quality_report['outliers'][col] = {
                    'count': int(len(outliers)),
                    'percentage': round((len(outliers) / len(df)) * 100, 2)
                }
        
        # Calculate overall quality score
        missing_penalty = sum([info['percentage'] for info in quality_report['missing_data'].values()]) / len(df.columns)
        duplicate_penalty = (quality_report['duplicates'] / len(df)) * 100
        outlier_penalty = sum([info['percentage'] for info in quality_report['outliers'].values()]) / max(len(quality_report['outliers']), 1)
        
        quality_report['quality_score'] = max(0, 100 - missing_penalty - duplicate_penalty - (outlier_penalty * 0.5))
        
        return quality_report

class EnhancedColumnMatcher:
    """Advanced column matching with ML-like features."""
    
    def __init__(self):
        self.canonical_vars = self._get_enhanced_canonical_variables()
        self.pattern_weights = {
            'exact_match': 100,
            'case_insensitive': 95,
            'substring_match': 80,
            'fuzzy_high': 85,
            'fuzzy_medium': 70,
            'pattern_match': 75
        }
    
    def _get_enhanced_canonical_variables(self) -> Dict[str, Dict[str, Any]]:
        """Enhanced canonical variables with patterns and metadata."""
        return {
            "participant_id": {
                "variations": ["id", "subject_id", "participant", "subj_id", "patient_id", "pid", "study_id"],
                "patterns": [r"^(id|subject|participant|patient).*", r".*_id$"],
                "data_type": "identifier",
                "description": "Unique participant identifier"
            },
            "age": {
                "variations": ["age", "age_years", "participant_age", "subject_age", "baseline_age"],
                "patterns": [r".*age.*", r".*years.*"],
                "data_type": "numeric",
                "description": "Participant age in years"
            },
            "gender": {
                "variations": ["gender", "sex", "male_female", "m_f", "gender_mf", "biological_sex"],
                "patterns": [r".*(gender|sex).*", r".*(male|female).*"],
                "data_type": "categorical",
                "description": "Participant gender/sex"
            },
            "baseline_weight": {
                "variations": ["weight", "baseline_weight", "weight_kg", "wt", "body_weight", "weight_baseline"],
                "patterns": [r".*weight.*baseline.*", r".*baseline.*weight.*", r"^weight$", r".*wt.*"],
                "data_type": "numeric",
                "description": "Baseline body weight (kg)"
            },
            "follow_up_weight": {
                "variations": ["follow_weight", "followup_weight", "weight_followup", "final_weight", "weight_final", "weight_end"],
                "patterns": [r".*weight.*(follow|final|end).*", r".*(follow|final|end).*weight.*"],
                "data_type": "numeric",
                "description": "Follow-up body weight (kg)"
            },
            "baseline_bmi": {
                "variations": ["bmi", "baseline_bmi", "bmi_baseline", "body_mass_index", "bmi_0"],
                "patterns": [r".*bmi.*baseline.*", r".*baseline.*bmi.*", r"^bmi$"],
                "data_type": "numeric",
                "description": "Baseline Body Mass Index"
            },
            "follow_up_bmi": {
                "variations": ["follow_bmi", "followup_bmi", "bmi_followup", "final_bmi", "bmi_final", "bmi_end"],
                "patterns": [r".*bmi.*(follow|final|end).*", r".*(follow|final|end).*bmi.*"],
                "data_type": "numeric",
                "description": "Follow-up Body Mass Index"
            },
            "physical_activity": {
                "variations": ["pa_change", "activity_change", "pa_minutes", "phys_act", "activity_mins", "exercise", "physical_activity"],
                "patterns": [r".*(activity|exercise|pa).*", r".*physical.*"],
                "data_type": "numeric",
                "description": "Physical activity measure"
            },
            "diet_adherence": {
                "variations": ["diet_score", "adherence", "diet_compliance", "nutrition_score", "dietary_adherence"],
                "patterns": [r".*(diet|nutrition).*", r".*adherence.*", r".*compliance.*"],
                "data_type": "numeric",
                "description": "Diet adherence score"
            },
            "systolic_bp": {
                "variations": ["sbp", "systolic", "sys_bp", "blood_pressure_sys", "systolic_bp", "bp_sys"],
                "patterns": [r".*(sbp|systolic).*", r".*blood.*pressure.*sys.*"],
                "data_type": "numeric",
                "description": "Systolic blood pressure (mmHg)"
            },
            "diastolic_bp": {
                "variations": ["dbp", "diastolic", "dia_bp", "blood_pressure_dia", "diastolic_bp", "bp_dia"],
                "patterns": [r".*(dbp|diastolic).*", r".*blood.*pressure.*dia.*"],
                "data_type": "numeric",
                "description": "Diastolic blood pressure (mmHg)"
            },
            "treatment_group": {
                "variations": ["group", "treatment", "arm", "intervention", "condition", "randomization"],
                "patterns": [r".*(group|treatment|arm|intervention).*", r".*random.*"],
                "data_type": "categorical",
                "description": "Treatment group assignment"
            },
            "dropout": {
                "variations": ["dropout", "withdrawn", "completed", "status", "drop_out", "completion_status"],
                "patterns": [r".*(dropout|withdraw|complet|status).*"],
                "data_type": "categorical",
                "description": "Study completion status"
            }
        }
    
    def match_columns_advanced(self, dataframes: Dict[str, pd.DataFrame], 
                             threshold: float = 70.0) -> Dict[str, Dict[str, Tuple[str, float, str]]]:
        """Advanced column matching with multiple strategies."""
        all_matches = {}
        
        for filename, df in dataframes.items():
            matches = {}
            
            for col in df.columns:
                best_match = self._find_best_match(col, df[col])
                matches[col] = best_match
            
            all_matches[filename] = matches
        
        return all_matches
    
    def _find_best_match(self, column_name: str, column_data: pd.Series) -> Tuple[str, float, str]:
        """Find best match using multiple strategies."""
        best_canonical = "unmapped"
        best_score = 0
        best_method = "none"
        
        col_lower = column_name.lower().strip()
        
        for canonical_name, info in self.canonical_vars.items():
            # Strategy 1: Exact match
            if col_lower in [v.lower() for v in info["variations"]]:
                return canonical_name, 100.0, "exact_match"
            
            # Strategy 2: Pattern matching
            for pattern in info.get("patterns", []):
                if re.search(pattern, col_lower):
                    score = self.pattern_weights["pattern_match"]
                    if score > best_score:
                        best_canonical = canonical_name
                        best_score = score
                        best_method = "pattern_match"
            
            # Strategy 3: Fuzzy matching with variations
            for variation in info["variations"]:
                fuzzy_score = fuzz.ratio(col_lower, variation.lower())
                if fuzzy_score >= 85:
                    adjusted_score = self.pattern_weights["fuzzy_high"]
                elif fuzzy_score >= 70:
                    adjusted_score = self.pattern_weights["fuzzy_medium"]
                else:
                    adjusted_score = fuzzy_score
                
                if adjusted_score > best_score:
                    best_canonical = canonical_name
                    best_score = adjusted_score
                    best_method = "fuzzy_match"
            
            # Strategy 4: Data type validation
            expected_type = info.get("data_type", "unknown")
            if best_canonical == canonical_name and self._validate_data_type(column_data, expected_type):
                best_score = min(100, best_score + 5)  # Bonus for type match
        
        return best_canonical, best_score, best_method

    def _validate_data_type(self, series: pd.Series, expected_type: str) -> bool:
        """Validate if column data matches expected type."""
        if expected_type == "numeric":
            return pd.api.types.is_numeric_dtype(series) or series.dtype == 'object' and self._is_convertible_to_numeric(series)
        elif expected_type == "categorical":
            return series.nunique() / len(series.dropna()) < 0.5  # Less than 50% unique values
        elif expected_type == "identifier":
            return series.nunique() == len(series.dropna())  # All unique values
        return True
    
    def _is_convertible_to_numeric(self, series: pd.Series) -> bool:
        """Check if string series can be converted to numeric."""
        try:
            pd.to_numeric(series.dropna().head(100), errors='raise')
            return True
        except:
            return False

def load_data_enhanced(uploaded_files) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict]]:
    """Enhanced data loading with quality assessment."""
    dataframes = {}
    quality_reports = {}
    
    for file in uploaded_files:
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    file.seek(0)  # Reset file pointer
                    df = pd.read_csv(file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                st.error(f"❌ Could not decode {file.name} with any encoding")
                continue
            
            # Clean column names
            df.columns = df.columns.str.strip().str.replace(r'\s+', '_', regex=True)
            
            # Basic data cleaning
            df = df.dropna(how='all')  # Remove completely empty rows
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Remove unnamed columns
            
            dataframes[file.name] = df
            
            # Generate quality report
            quality_checker = DataQualityChecker()
            quality_reports[file.name] = quality_checker.assess_data_quality(df, file.name)
            
            st.success(f"✅ Loaded {file.name}: {df.shape[0]} rows, {df.shape[1]} columns (Quality Score: {quality_reports[file.name]['quality_score']:.1f}%)")
            
        except Exception as e:
            st.error(f"❌ Error loading {file.name}: {str(e)}")
    
    return dataframes, quality_reports

def create_enhanced_mapping_interface(dataframes: Dict[str, pd.DataFrame], 
                                    matches: Dict[str, Dict[str, Tuple[str, float, str]]],
                                    canonical_vars: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    """Enhanced mapping interface with confidence indicators and bulk operations."""
    st.subheader("🎯 Advanced Column Mapping Interface")
    
    # Bulk operations
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("✅ Accept All High Confidence (≥90%)", type="secondary"):
            st.session_state.bulk_accept_high = True
    with col2:
        if st.button("🔄 Reset All to Auto-Match", type="secondary"):
            st.session_state.bulk_reset = True
    with col3:
        confidence_filter = st.selectbox("Filter by Confidence", ["All", "High (≥90%)", "Medium (70-89%)", "Low (<70%)"])
    
    canonical_options = ["unmapped"] + list(canonical_vars.keys())
    final_mappings = {}
    
    for filename, df in dataframes.items():
        st.markdown(f"### 📊 {filename}")
        
        file_matches = matches.get(filename, {})
        
        # Create enhanced mapping data
        mapping_data = []
        for col in df.columns:
            original_name = col
            if col in file_matches:
                matched_canonical, score, method = file_matches[col]
            else:
                matched_canonical, score, method = ("unmapped", 0, "none")
            
            # Confidence level
            if score >= 90:
                confidence = "🟢 High"
            elif score >= 70:
                confidence = "🟡 Medium"
            else:
                confidence = "🔴 Low"
            
            # Data preview
            sample_values = df[col].dropna().head(3).tolist()
            preview = ", ".join([str(v)[:20] for v in sample_values])
            if len(preview) > 60:
                preview = preview[:60] + "..."
            
            mapping_data.append({
                "Original Name": original_name,
                "Matched Variable": matched_canonical,
                "Confidence": confidence,
                "Score": f"{score:.1f}%",
                "Method": method,
                "Sample Values": preview,
                "Missing %": f"{(df[col].isnull().sum() / len(df) * 100):.1f}%"
            })
        
        # Apply confidence filter
        if confidence_filter != "All":
            if confidence_filter == "High (≥90%)":
                mapping_data = [row for row in mapping_data if float(row["Score"].replace('%', '')) >= 90]
            elif confidence_filter == "Medium (70-89%)":
                mapping_data = [row for row in mapping_data if 70 <= float(row["Score"].replace('%', '')) < 90]
            elif confidence_filter == "Low (<70%)":
                mapping_data = [row for row in mapping_data if float(row["Score"].replace('%', '')) < 70]
        
        # Display enhanced mapping table
        if mapping_data:
            mapping_df = pd.DataFrame(mapping_data)
            st.dataframe(mapping_df, use_container_width=True, height=min(400, len(mapping_data) * 35 + 100))
        
        # Interactive mapping override
        st.markdown("**🔧 Adjust Mappings:**")
        file_mapping = {}
        
        # Create columns for better layout
        cols_per_row = 2
        columns = st.columns(cols_per_row)
        
        for i, col in enumerate(df.columns):
            with columns[i % cols_per_row]:
                current_match = file_matches.get(col, ("unmapped", 0, "none"))[0]
                
                # Handle bulk operations
                if hasattr(st.session_state, 'bulk_accept_high') and st.session_state.bulk_accept_high:
                    if file_matches.get(col, (None, 0, None))[1] >= 90:
                        current_match = file_matches[col][0]
                
                if hasattr(st.session_state, 'bulk_reset') and st.session_state.bulk_reset:
                    current_match = file_matches.get(col, ("unmapped", 0, "none"))[0]
                
                # Get variable description
                desc = ""
                if current_match in canonical_vars:
                    desc = canonical_vars[current_match].get("description", "")
                
                override = st.selectbox(
                    f"{col}",
                    options=canonical_options,
                    index=canonical_options.index(current_match) if current_match in canonical_options else 0,
                    key=f"mapping_{filename}_{col}_{i}",
                    help=f"Current match confidence: {file_matches.get(col, (None, 0, None))[1]:.1f}%\n{desc}"
                )
                file_mapping[col] = override
        
        final_mappings[filename] = file_mapping
        st.divider()
    
    # Clear bulk operation flags
    if hasattr(st.session_state, 'bulk_accept_high'):
        del st.session_state.bulk_accept_high
    if hasattr(st.session_state, 'bulk_reset'):
        del st.session_state.bulk_reset
    
    return final_mappings

def harmonize_dataframes_enhanced(dataframes: Dict[str, pd.DataFrame], 
                                mappings: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """Enhanced harmonization with data type conversion and validation."""
    harmonized_dfs = []
    
    for filename, df in dataframes.items():
        df_harm = df.copy()
        df_harm['_source_file'] = filename
        df_harm['_source_row_index'] = df_harm.index
        
        # Apply column mappings
        mapping = mappings[filename]
        rename_dict = {orig: canonical for orig, canonical in mapping.items() 
                      if canonical != "unmapped"}
        
        df_harm = df_harm.rename(columns=rename_dict)
        
        # Data type conversion and cleaning
        matcher = EnhancedColumnMatcher()
        for new_col in df_harm.columns:
            if new_col in matcher.canonical_vars:
                expected_type = matcher.canonical_vars[new_col].get("data_type", "unknown")
                df_harm[new_col] = clean_and_convert_column(df_harm[new_col], expected_type)
        
        # Drop unmapped columns
        unmapped_cols = [col for col, canonical in mapping.items() if canonical == "unmapped"]
        df_harm = df_harm.drop(columns=unmapped_cols, errors='ignore')
        
        harmonized_dfs.append(df_harm)
    
    if not harmonized_dfs:
        return pd.DataFrame()
    
    # Smart merging strategy
    merged_df = smart_merge_dataframes(harmonized_dfs)
    
    return merged_df

def clean_and_convert_column(series: pd.Series, expected_type: str) -> pd.Series:
    """Clean and convert column based on expected data type."""
    if expected_type == "numeric":
        # Convert to numeric, handling common issues
        series = series.astype(str).str.replace(r'[^\d.-]', '', regex=True)
        series = pd.to_numeric(series, errors='coerce')
    elif expected_type == "categorical":
        # Standardize categorical values
        series = series.astype(str).str.strip().str.lower()
        # Handle common gender variations
        if series.name and 'gender' in series.name.lower():
            series = series.replace({'m': 'male', 'f': 'female', '1': 'male', '2': 'female', '0': 'male'})
    
    return series

def smart_merge_dataframes(harmonized_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Intelligently merge dataframes based on available ID columns."""
    # Check for common ID columns
    id_candidates = ['participant_id', 'subject_id', 'id']
    merge_column = None
    
    for id_col in id_candidates:
        if all(id_col in df.columns for df in harmonized_dfs):
            merge_column = id_col
            break
    
    if merge_column:
        # Merge on ID column
        merged_df = harmonized_dfs[0]
        for i, df in enumerate(harmonized_dfs[1:], 1):
            # Handle overlapping columns
            overlap_cols = set(merged_df.columns) & set(df.columns)
            overlap_cols.discard(merge_column)
            overlap_cols.discard('_source_file')
            overlap_cols.discard('_source_row_index')
            
            suffixes = (f'_study1' if i == 1 else '', f'_study{i+1}')
            merged_df = pd.merge(merged_df, df, on=merge_column, how='outer', suffixes=suffixes)
    else:
        # Concatenate with study identifiers
        for i, df in enumerate(harmonized_dfs):
            df['_study_number'] = i + 1
        merged_df = pd.concat(harmonized_dfs, ignore_index=True, sort=False)
    
    return merged_df

def create_advanced_visualizations(df: pd.DataFrame) -> None:
    """Create advanced interactive visualizations."""
    st.subheader("📈 Advanced Data Visualizations")
    
    viz_tabs = st.tabs(["📊 Distributions", "🔗 Correlations", "📈 Trends", "🎯 Quality"])
    
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if not col.startswith('_') and 'participant_id' not in col.lower()]
    
    with viz_tabs[0]:  # Distributions
        if numeric_cols:
            selected_vars = st.multiselect("Select variables to visualize", numeric_cols, default=numeric_cols[:3])
            
            if selected_vars:
                # Create subplot for distributions
                fig = make_subplots(
                    rows=len(selected_vars), cols=2,
                    subplot_titles=[f"{var} - Histogram" for var in selected_vars] + 
                                  [f"{var} - Box Plot" for var in selected_vars],
                    specs=[[{"secondary_y": False}, {"secondary_y": False}] for _ in selected_vars]
                )
                
                for i, var in enumerate(selected_vars):
                    # Histogram
                    fig.add_trace(
                        go.Histogram(x=df[var].dropna(), name=f"{var}", showlegend=False),
                        row=i+1, col=1
                    )
                    
                    # Box plot
                    fig.add_trace(
                        go.Box(y=df[var].dropna(), name=f"{var}", showlegend=False),
                        row=i+1, col=2
                    )
                
                fig.update_layout(height=300*len(selected_vars), title_text="Variable Distributions")
                st.plotly_chart(fig, use_container_width=True)
    
    with viz_tabs[1]:  # Correlations
        if len(numeric_cols) >= 2:
            correlation_vars = st.multiselect("Select variables for correlation analysis", 
                                            numeric_cols, default=numeric_cols[:5])
            
            if len(correlation_vars) >= 2:
                corr_data = df[correlation_vars].corr()
                
                fig = px.imshow(corr_data, text_auto=True, aspect="auto",
                              title="Correlation Matrix",
                              color_continuous_scale="RdBu_r")
                st.plotly_chart(fig, use_container_width=True)
    
    with viz_tabs[2]:  # Trends
        if '_source_file' in df.columns and numeric_cols:
            trend_var = st.selectbox("Select variable for trend analysis", numeric_cols)
            
            if trend_var:
                trend_data = df.groupby('_source_file')[trend_var].agg(['mean', 'std', 'count']).reset_index()
                
                fig = go.Figure()
                fig.add_trace(go.Bar(x=trend_data['_source_file'], y=trend_data['mean'],
                                   error_y=dict(type='data', array=trend_data['std']),
                                   name=f"Mean {trend_var}"))
                
                fig.update_layout(title=f"Mean {trend_var} by Study",
                                xaxis_title="Study", yaxis_title=f"Mean {trend_var}")
                st.plotly_chart(fig, use_container_width=True)
    
    with viz_tabs[3]:  # Quality
        create_data_quality_dashboard(df)

def create_data_quality_dashboard(df: pd.DataFrame) -> None:
    """Create a comprehensive data quality dashboard."""
    st.markdown("### 🎯 Data Quality Dashboard")
    
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    
    with col2:
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        completeness = ((total_cells - missing_cells) / total_cells) * 100
        st.metric("Data Completeness", f"{completeness:.1f}%")
    
    with col3:
        duplicates = df.duplicated().sum()
        st.metric("Duplicate Records", f"{duplicates:,}")
    
    with col4:
        if '_source_file' in df.columns:
            unique_studies = df['_source_file'].nunique()
            st.metric("Studies Merged", f"{unique_studies}")
    
    # Missing data heatmap
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        st.markdown("#### Missing Data by Variable")
        missing_pct = (missing_data / len(df) * 100).sort_values(ascending=False)
        
        fig = px.bar(x=missing_pct.index, y=missing_pct.values,
                    title="Missing Data Percentage by Variable",
                    labels={'x': 'Variables', 'y': 'Missing %'})
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

def create_enhanced_export_options(merged_df: pd.DataFrame, 
                                 mappings: Dict[str, Dict[str, str]],
                                 quality_reports: Dict[str, Dict]) -> None:
    """Enhanced export functionality with multiple formats and options."""
    st.subheader("💾 Export Options")
    
    export_tabs = st.tabs(["📊 Data Export", "📋 Documentation", "📈 Reports"])
    
    with export_tabs[0]:  # Data Export
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Select Export Format:**")
            export_format = st.radio("Format", ["CSV", "Excel", "JSON", "Parquet"])
            
            include_metadata = st.checkbox("Include metadata columns", value=True)
            
            if st.button("📥 Generate Export", type="primary"):
                export_df = merged_df.copy()
                
                if not include_metadata:
                    metadata_cols = [col for col in export_df.columns if col.startswith('_')]
                    export_df = export_df.drop(columns=metadata_cols)
                
                if export_format == "CSV":
                    csv_buffer = io.StringIO()
                    export_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        "📥 Download CSV",
                        csv_buffer.getvalue(),
                        f"harmonized_rct_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )
                
                elif export_format == "Excel":
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                        export_df.to_excel(writer, sheet_name='Harmonized_Data', index=False)
                        
                        # Add summary sheet
                        summary_data = {
                            'Metric': ['Total Records', 'Total Variables', 'Studies Merged', 'Export Date'],
                            'Value': [len(export_df), len(export_df.columns), 
                                    export_df.get('_source_file', pd.Series()).nunique() if '_source_file' in export_df.columns else 'N/A',
                                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
                        }
                        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                    
                    st.download_button(
                        "📥 Download Excel",
                        excel_buffer.getvalue(),
                        f"harmonized_rct_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                elif export_format == "JSON":
                    json_str = export_df.to_json(orient='records', indent=2)
                    st.download_button(
                        "📥 Download JSON",
                        json_str,
                        f"harmonized_rct_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        "application/json"
                    )
        
        with col2:
            st.markdown("**Export Preview:**")
            if not merged_df.empty:
                preview_df = merged_df.head(5)
                if not include_metadata:
                    metadata_cols = [col for col in preview_df.columns if col.startswith('_')]
                    preview_df = preview_df.drop(columns=metadata_cols)
                st.dataframe(preview_df, use_container_width=True)
    
    with export_tabs[1]:  # Documentation
        st.markdown("**📋 Generate Documentation**")
        
        doc_options = st.multiselect(
            "Select documentation components:",
            ["Variable Codebook", "Data Quality Report", "Mapping Summary", "Study Descriptions"],
            default=["Variable Codebook", "Data Quality Report"]
        )
        
        if st.button("📋 Generate Documentation"):
            doc_content = generate_comprehensive_documentation(merged_df, mappings, quality_reports, doc_options)
            
            st.download_button(
                "📋 Download Documentation",
                doc_content,
                f"rct_harmonization_documentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                "text/markdown"
            )
    
    with export_tabs[2]:  # Reports
        st.markdown("**📈 Analysis Reports**")
        
        if st.button("📊 Generate Summary Report"):
            report_content = generate_summary_report(merged_df, quality_reports)
            
            st.download_button(
                "📊 Download Summary Report",
                report_content,
                f"rct_summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                "text/html"
            )

def generate_comprehensive_documentation(df: pd.DataFrame, 
                                       mappings: Dict[str, Dict[str, str]],
                                       quality_reports: Dict[str, Dict],
                                       doc_options: List[str]) -> str:
    """Generate comprehensive documentation."""
    doc_content = f"""# RCT Data Harmonization Documentation

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
This document provides comprehensive documentation for the harmonized RCT dataset.

- **Total Records**: {len(df):,}
- **Total Variables**: {len(df.columns)}
- **Studies Merged**: {df.get('_source_file', pd.Series()).nunique() if '_source_file' in df.columns else 'N/A'}

"""
    
    if "Variable Codebook" in doc_options:
        doc_content += "\n## Variable Codebook\n\n"
        doc_content += "| Variable Name | Description | Data Type | Missing % | Source Studies |\n"
        doc_content += "|---------------|-------------|-----------|-----------|----------------|\n"
        
        matcher = EnhancedColumnMatcher()
        for col in df.columns:
            if not col.startswith('_'):
                description = "Custom variable"
                if col in matcher.canonical_vars:
                    description = matcher.canonical_vars[col].get('description', 'Standard RCT variable')
                
                data_type = str(df[col].dtype)
                missing_pct = f"{(df[col].isnull().sum() / len(df) * 100):.1f}%"
                
                # Find source studies for this variable
                source_studies = []
                for filename, mapping in mappings.items():
                    if col in mapping.values():
                        source_studies.append(filename)
                
                sources = ", ".join(source_studies) if source_studies else "Generated"
                
                doc_content += f"| {col} | {description} | {data_type} | {missing_pct} | {sources} |\n"
    
    if "Data Quality Report" in doc_options:
        doc_content += "\n## Data Quality Report\n\n"
        for filename, report in quality_reports.items():
            doc_content += f"### {filename}\n"
            doc_content += f"- **Quality Score**: {report['quality_score']:.1f}%\n"
            doc_content += f"- **Total Records**: {report['total_rows']:,}\n"
            doc_content += f"- **Variables**: {report['total_columns']}\n"
            doc_content += f"- **Duplicates**: {report['duplicates']}\n\n"
    
    if "Mapping Summary" in doc_options:
        doc_content += "\n## Variable Mapping Summary\n\n"
        for filename, mapping in mappings.items():
            doc_content += f"### {filename}\n\n"
            doc_content += "| Original Variable | Mapped To |\n"
            doc_content += "|-------------------|----------|\n"
            for orig, mapped in mapping.items():
                doc_content += f"| {orig} | {mapped} |\n"
            doc_content += "\n"
    
    return doc_content

def generate_summary_report(df: pd.DataFrame, quality_reports: Dict[str, Dict]) -> str:
    """Generate HTML summary report."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>RCT Harmonization Summary Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background: #667eea; color: white; padding: 20px; border-radius: 8px; }}
            .metric-card {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #667eea; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔬 RCT Data Harmonization Summary Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="metric-card">
            <h2>📊 Overall Statistics</h2>
            <ul>
                <li><strong>Total Records:</strong> {len(df):,}</li>
                <li><strong>Total Variables:</strong> {len(df.columns)}</li>
                <li><strong>Studies Merged:</strong> {df.get('_source_file', pd.Series()).nunique() if '_source_file' in df.columns else 'N/A'}</li>
                <li><strong>Data Completeness:</strong> {((df.size - df.isnull().sum().sum()) / df.size * 100):.1f}%</li>
            </ul>
        </div>
        
        <div class="metric-card">
            <h2>📈 Study Quality Scores</h2>
            <table>
                <tr><th>Study</th><th>Quality Score</th><th>Records</th><th>Variables</th></tr>
    """
    
    for filename, report in quality_reports.items():
        html_content += f"""
                <tr>
                    <td>{filename}</td>
                    <td>{report['quality_score']:.1f}%</td>
                    <td>{report['total_rows']:,}</td>
                    <td>{report['total_columns']}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
    </body>
    </html>
    """
    
    return html_content

def main():
    """Enhanced main application function."""
    
    # App header with custom styling
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0;">🔬 Advanced RCT Data Harmonizer</h1>
        <p style="color: #e8eaf6; margin: 5px 0 0 0;">Intelligent harmonization and analysis of randomized controlled trial datasets</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    with st.expander("✨ What's New in This Enhanced Version"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **🎯 Smart Matching:**
            - Pattern-based recognition
            - Data type validation
            - Confidence scoring
            - Bulk operations
            """)
        with col2:
            st.markdown("""
            **📊 Advanced Analytics:**
            - Interactive visualizations
            - Quality dashboards
            - Correlation analysis
            - Multi-format exports
            """)
    
    # Initialize session state
    if 'processing_stage' not in st.session_state:
        st.session_state.processing_stage = 'upload'
    
    # Sidebar setup
    st.sidebar.markdown("### 🚀 Processing Pipeline")
    
    # Progress indicator
    stages = ['upload', 'quality', 'mapping', 'harmonization', 'analysis']
    stage_names = ['📁 Upload', '🔍 Quality Check', '🎯 Mapping', '🔄 Harmonization', '📊 Analysis']
    
    for i, (stage, name) in enumerate(zip(stages, stage_names)):
        if stages.index(st.session_state.processing_stage) >= i:
            st.sidebar.markdown(f"✅ {name}")
        else:
            st.sidebar.markdown(f"⏳ {name}")
    
    # File upload section
    st.sidebar.markdown("---")
    st.sidebar.header("📁 Data Upload")
    uploaded_files = st.sidebar.file_uploader(
        "Upload RCT CSV files",
        type="csv",
        accept_multiple_files=True,
        help="Upload multiple CSV files from different RCT studies"
    )
    
    if not uploaded_files:
        st.info("👆 Upload your RCT CSV files to begin the harmonization process.")
        
        # Show example data structure
        with st.expander("📋 Expected Data Structure Examples"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Study A Format:**")
                example_a = pd.DataFrame({
                    'subj_id': [1, 2, 3],
                    'age_years': [25, 34, 42],
                    'gender': ['M', 'F', 'M'],
                    'baseline_wt': [70.5, 65.2, 85.1],
                    'group': ['intervention', 'control', 'intervention']
                })
                st.dataframe(example_a)
            
            with col2:
                st.markdown("**Study B Format:**")
                example_b = pd.DataFrame({
                    'participant_id': [101, 102, 103],
                    'age': [28, 31, 39],
                    'sex': ['male', 'female', 'male'],
                    'weight_kg': [72.1, 62.8, 78.3],
                    'treatment_arm': ['active', 'placebo', 'active']
                })
                st.dataframe(example_b)
        
        return
    
    # Load and assess data quality
    if st.session_state.processing_stage == 'upload':
        with st.spinner("🔄 Loading and assessing data quality..."):
            dataframes, quality_reports = load_data_enhanced(uploaded_files)
        
        if dataframes:
            st.session_state.dataframes = dataframes
            st.session_state.quality_reports = quality_reports
            st.session_state.processing_stage = 'quality'
            st.rerun()
    
    # Quality assessment dashboard
    if st.session_state.processing_stage == 'quality' and 'dataframes' in st.session_state:
        st.subheader("🔍 Data Quality Assessment")
        
        # Overall quality metrics
        avg_quality = np.mean([report['quality_score'] for report in st.session_state.quality_reports.values()])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Files Loaded", len(st.session_state.dataframes))
        with col2:
            st.metric("Average Quality", f"{avg_quality:.1f}%")
        with col3:
            total_records = sum(len(df) for df in st.session_state.dataframes.values())
            st.metric("Total Records", f"{total_records:,}")
        with col4:
            total_variables = sum(len(df.columns) for df in st.session_state.dataframes.values())
            st.metric("Total Variables", total_variables)
        
        # Detailed quality reports
        for filename, report in st.session_state.quality_reports.items():
            with st.expander(f"📊 {filename} - Quality Score: {report['quality_score']:.1f}%"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📈 Basic Statistics:**")
                    st.write(f"- Rows: {report['total_rows']:,}")
                    st.write(f"- Columns: {report['total_columns']}")
                    st.write(f"- Duplicates: {report['duplicates']}")
                
                with col2:
                    st.markdown("**❌ Missing Data:**")
                    missing_vars = [(var, info['percentage']) for var, info in report['missing_data'].items() 
                                  if info['percentage'] > 0]
                    if missing_vars:
                        for var, pct in sorted(missing_vars, key=lambda x: x[1], reverse=True)[:5]:
                            st.write(f"- {var}: {pct:.1f}%")
                    else:
                        st.write("- No missing data detected")
        
        if st.button("✅ Proceed to Variable Mapping", type="primary"):
            st.session_state.processing_stage = 'mapping'
            st.rerun()
    
    # Enhanced mapping interface
    if st.session_state.processing_stage == 'mapping' and 'dataframes' in st.session_state:
        # Perform advanced matching
        matcher = EnhancedColumnMatcher()
        
        with st.spinner("🎯 Performing intelligent variable matching..."):
            matches = matcher.match_columns_advanced(st.session_state.dataframes)
        
        st.session_state.matches = matches
        
        # Show matching summary
        st.subheader("🎯 Intelligent Variable Matching Results")
        
        # Matching statistics
        total_cols = sum(len(df.columns) for df in st.session_state.dataframes.values())
        high_confidence = sum(1 for file_matches in matches.values() 
                            for _, score, _ in file_matches.values() if score >= 90)
        medium_confidence = sum(1 for file_matches in matches.values() 
                              for _, score, _ in file_matches.values() if 70 <= score < 90)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Variables", total_cols)
        with col2:
            st.metric("High Confidence", f"{high_confidence} ({high_confidence/total_cols*100:.1f}%)")
        with col3:
            st.metric("Medium Confidence", f"{medium_confidence} ({medium_confidence/total_cols*100:.1f}%)")
        with col4:
            unmapped = total_cols - high_confidence - medium_confidence
            st.metric("Needs Review", f"{unmapped} ({unmapped/total_cols*100:.1f}%)")
        
        # Interactive mapping interface
        final_mappings = create_enhanced_mapping_interface(
            st.session_state.dataframes, matches, matcher.canonical_vars
        )
        
        st.session_state.final_mappings = final_mappings
        
        if st.button("🔄 Apply Harmonization", type="primary"):
            st.session_state.processing_stage = 'harmonization'
            st.rerun()
    
    # Harmonization process
    if st.session_state.processing_stage == 'harmonization' and 'final_mappings' in st.session_state:
        with st.spinner("🔄 Harmonizing and merging datasets..."):
            merged_df = harmonize_dataframes_enhanced(
                st.session_state.dataframes, 
                st.session_state.final_mappings
            )
        
        if merged_df.empty:
            st.error("❌ Harmonization failed. Please review your mappings and try again.")
            if st.button("🔙 Return to Mapping"):
                st.session_state.processing_stage = 'mapping'
                st.rerun()
            return
        
        st.session_state.merged_df = merged_df
        
        # Harmonization success message
        st.markdown("""
        <div class="success-box">
            <h3>✅ Harmonization Successful!</h3>
            <p>Your datasets have been successfully harmonized and merged.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Final Dataset Size", f"{merged_df.shape[0]:,} × {merged_df.shape[1]}")
        with col2:
            completeness = ((merged_df.size - merged_df.isnull().sum().sum()) / merged_df.size * 100)
            st.metric("Data Completeness", f"{completeness:.1f}%")
        with col3:
            if '_source_file' in merged_df.columns:
                studies_merged = merged_df['_source_file'].nunique()
                st.metric("Studies Merged", studies_merged)
        
        if st.button("📊 Proceed to Analysis", type="primary"):
            st.session_state.processing_stage = 'analysis'
            st.rerun()
    
    # Analysis and export
    if st.session_state.processing_stage == 'analysis' and 'merged_df' in st.session_state:
        merged_df = st.session_state.merged_df
        
        # Data preview
        st.subheader("👀 Harmonized Dataset Preview")
        st.dataframe(merged_df.head(10), use_container_width=True)
        
        # Advanced visualizations
        create_advanced_visualizations(merged_df)
        
        # Enhanced export options
        create_enhanced_export_options(
            merged_df, 
            st.session_state.final_mappings,
            st.session_state.quality_reports
        )
        
        # Reset option
        st.markdown("---")
        if st.button("🔄 Start New Harmonization", type="secondary"):
            for key in ['dataframes', 'quality_reports', 'matches', 'final_mappings', 'merged_df']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.processing_stage = 'upload'
            st.rerun()

if __name__ == "__main__":
    main()