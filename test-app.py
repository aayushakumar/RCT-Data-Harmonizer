# requirements.txt
# streamlit==1.28.0
# pandas==2.1.0
# rapidfuzz==3.3.0
# matplotlib==3.7.2
# altair==5.1.0
# plotly==5.17.0

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
import json
from typing import Dict, List, Tuple, Optional, Any
import warnings
import re
from datetime import datetime
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="RCT Data Harmonizer Pro",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        padding: 1rem 0;
        border-bottom: 2px solid #f0f2f6;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-card {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .success-card {
        background: #d1edff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #198754;
    }
</style>
""", unsafe_allow_html=True)

class DataQualityChecker:
    """Enhanced data quality checking and validation."""
    
    def __init__(self):
        self.quality_issues = []
        
    def check_data_quality(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Comprehensive data quality assessment."""
        issues = []
        
        # Basic checks
        total_rows = len(df)
        total_cols = len(df.columns)
        
        # Duplicate rows
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            issues.append(f"Found {duplicate_rows} duplicate rows")
            
        # Completely empty columns
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            issues.append(f"Completely empty columns: {', '.join(empty_cols)}")
            
        # High missing data columns (>50% missing)
        high_missing = []
        for col in df.columns:
            missing_pct = df[col].isnull().sum() / len(df) * 100
            if missing_pct > 50:
                high_missing.append(f"{col} ({missing_pct:.1f}% missing)")
        
        if high_missing:
            issues.append(f"High missing data columns: {', '.join(high_missing)}")
            
        # Potential ID columns
        potential_ids = []
        for col in df.columns:
            if df[col].nunique() == len(df) and not df[col].isnull().any():
                potential_ids.append(col)
                
        # Data type inconsistencies
        numeric_issues = []
        for col in df.select_dtypes(include=['object']).columns:
            # Check if supposed to be numeric
            sample_vals = df[col].dropna().astype(str).str.strip()
            if len(sample_vals) > 0:
                numeric_pattern = sample_vals.str.match(r'^-?\d*\.?\d+$')
                if numeric_pattern.sum() / len(sample_vals) > 0.8:
                    numeric_issues.append(col)
                    
        if numeric_issues:
            issues.append(f"Columns that might be numeric: {', '.join(numeric_issues)}")
            
        return {
            'filename': filename,
            'total_rows': total_rows,
            'total_cols': total_cols,
            'duplicate_rows': duplicate_rows,
            'empty_columns': empty_cols,
            'potential_id_columns': potential_ids,
            'issues': issues,
            'missing_data_summary': self._get_missing_summary(df)
        }
    
    def _get_missing_summary(self, df: pd.DataFrame) -> Dict[str, float]:
        """Get missing data summary by column."""
        return {col: df[col].isnull().sum() / len(df) * 100 
                for col in df.columns if df[col].isnull().any()}

class EnhancedVariableMatcher:
    """Improved variable matching with multiple strategies."""
    
    def __init__(self):
        self.canonical_vars = self._get_enhanced_canonical_variables()
        self.matching_strategies = [
            self._exact_match,
            self._fuzzy_match,
            self._semantic_match,
            self._pattern_match
        ]
    
    def _get_enhanced_canonical_variables(self) -> Dict[str, Dict[str, Any]]:
        """Enhanced canonical variables with metadata."""
        return {
            "participant_id": {
                "variations": ["id", "subject_id", "participant", "subj_id", "patient_id", "pid", "study_id"],
                "type": "identifier",
                "required": True,
                "description": "Unique participant identifier"
            },
            "age": {
                "variations": ["age", "age_years", "participant_age", "subject_age", "age_at_baseline"],
                "type": "numeric",
                "range": [0, 120],
                "description": "Participant age in years"
            },
            "gender": {
                "variations": ["gender", "sex", "male_female", "m_f", "gender_mf", "biological_sex"],
                "type": "categorical",
                "categories": ["M", "F", "Male", "Female", "1", "2", "0", "1"],
                "description": "Participant gender/sex"
            },
            "baseline_weight": {
                "variations": ["weight", "baseline_weight", "weight_kg", "wt", "body_weight", "weight_baseline"],
                "type": "numeric",
                "range": [20, 300],
                "unit": "kg",
                "description": "Baseline body weight"
            },
            "follow_up_weight": {
                "variations": ["follow_weight", "followup_weight", "weight_followup", "final_weight", "endline_weight"],
                "type": "numeric",
                "range": [20, 300],
                "unit": "kg",
                "description": "Follow-up body weight"
            },
            "baseline_bmi": {
                "variations": ["bmi", "baseline_bmi", "bmi_baseline", "body_mass_index", "bmi_bl"],
                "type": "numeric",
                "range": [10, 60],
                "unit": "kg/m²",
                "description": "Baseline BMI"
            },
            "follow_up_bmi": {
                "variations": ["follow_bmi", "followup_bmi", "bmi_followup", "final_bmi", "bmi_final"],
                "type": "numeric",
                "range": [10, 60],
                "unit": "kg/m²",
                "description": "Follow-up BMI"
            },
            "height": {
                "variations": ["height", "height_cm", "height_m", "ht", "stature"],
                "type": "numeric",
                "range": [100, 250],
                "unit": "cm",
                "description": "Participant height"
            },
            "physical_activity": {
                "variations": ["pa", "activity", "pa_minutes", "phys_act", "activity_mins", "exercise", "mvpa"],
                "type": "numeric",
                "range": [0, 2000],
                "unit": "minutes/week",
                "description": "Physical activity level"
            },
            "diet_adherence": {
                "variations": ["diet_score", "adherence", "diet_compliance", "nutrition_score", "dietary_adherence"],
                "type": "numeric",
                "range": [0, 100],
                "unit": "%",
                "description": "Diet adherence score"
            },
            "systolic_bp": {
                "variations": ["sbp", "systolic", "sys_bp", "blood_pressure_sys", "bp_sys"],
                "type": "numeric",
                "range": [70, 250],
                "unit": "mmHg",
                "description": "Systolic blood pressure"
            },
            "diastolic_bp": {
                "variations": ["dbp", "diastolic", "dia_bp", "blood_pressure_dia", "bp_dia"],
                "type": "numeric",
                "range": [40, 150],
                "unit": "mmHg",
                "description": "Diastolic blood pressure"
            },
            "treatment_group": {
                "variations": ["group", "treatment", "arm", "intervention", "condition", "randomization"],
                "type": "categorical",
                "description": "Treatment group assignment"
            },
            "randomization_date": {
                "variations": ["rand_date", "randomization_date", "enrollment_date", "baseline_date"],
                "type": "date",
                "description": "Date of randomization"
            },
            "dropout": {
                "variations": ["dropout", "withdrawn", "completed", "status", "drop_out", "completion_status"],
                "type": "categorical",
                "categories": ["Yes", "No", "1", "0", "Completed", "Withdrawn"],
                "description": "Study completion status"
            }
        }
    
    def _exact_match(self, column: str, canonical_name: str, variations: List[str]) -> float:
        """Exact string matching."""
        if column.lower().strip() in [v.lower().strip() for v in variations]:
            return 100.0
        return 0.0
    
    def _fuzzy_match(self, column: str, canonical_name: str, variations: List[str]) -> float:
        """Fuzzy string matching."""
        best_score = 0.0
        for variation in variations:
            score = fuzz.ratio(column.lower().strip(), variation.lower().strip())
            best_score = max(best_score, score)
        return best_score
    
    def _semantic_match(self, column: str, canonical_name: str, variations: List[str]) -> float:
        """Semantic/contextual matching based on keywords."""
        col_lower = column.lower().strip()
        
        # Keyword mapping for semantic understanding
        semantic_keywords = {
            "age": ["age", "years", "yr"],
            "weight": ["weight", "wt", "mass", "kg"],
            "height": ["height", "ht", "tall", "cm", "meter"],
            "bmi": ["bmi", "mass", "index", "body"],
            "bp": ["pressure", "bp", "systolic", "diastolic"],
            "activity": ["activity", "exercise", "physical", "pa", "mvpa"],
            "group": ["group", "arm", "treatment", "control", "intervention"],
            "id": ["id", "identifier", "number", "code"]
        }
        
        for key, keywords in semantic_keywords.items():
            if key in canonical_name and any(kw in col_lower for kw in keywords):
                return 80.0
        
        return 0.0
    
    def _pattern_match(self, column: str, canonical_name: str, variations: List[str]) -> float:
        """Pattern-based matching using regex."""
        patterns = {
            "participant_id": [r".*id.*", r".*subj.*", r".*part.*"],
            "baseline_": [r".*base.*", r".*bl.*", r".*t0.*", r".*initial.*"],
            "follow_up_": [r".*follow.*", r".*final.*", r".*end.*", r".*t1.*", r".*post.*"],
            "systolic_bp": [r".*sys.*", r".*sbp.*"],
            "diastolic_bp": [r".*dia.*", r".*dbp.*"]
        }
        
        col_lower = column.lower().strip()
        for pattern_key, regex_list in patterns.items():
            if pattern_key in canonical_name:
                for pattern in regex_list:
                    if re.search(pattern, col_lower):
                        return 75.0
        
        return 0.0
    
    def match_columns(self, dataframes: Dict[str, pd.DataFrame], threshold: float = 70.0) -> Dict[str, Dict[str, Tuple[str, float, str]]]:
        """Enhanced column matching with multiple strategies."""
        all_matches = {}
        
        for filename, df in dataframes.items():
            matches = {}
            
            for col in df.columns:
                best_canonical = "unmapped"
                best_score = 0.0
                best_strategy = "none"
                
                for canonical_name, metadata in self.canonical_vars.items():
                    variations = metadata["variations"]
                    
                    # Try all matching strategies
                    for strategy in self.matching_strategies:
                        score = strategy(col, canonical_name, variations)
                        
                        if score > best_score:
                            best_score = score
                            best_canonical = canonical_name
                            best_strategy = strategy.__name__
                            
                            # Early exit for perfect matches
                            if score == 100.0:
                                break
                    
                    if best_score == 100.0:
                        break
                
                # Only include matches above threshold
                if best_score >= threshold:
                    matches[col] = (best_canonical, best_score, best_strategy)
                else:
                    matches[col] = ("unmapped", best_score, "none")
            
            all_matches[filename] = matches
        
        return all_matches

def load_data_with_validation(uploaded_files) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict]]:
    """Load data with enhanced validation and quality checks."""
    dataframes = {}
    quality_reports = {}
    quality_checker = DataQualityChecker()
    
    for file in uploaded_files:
        try:
            # Try different encodings if needed
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
                st.error(f"❌ Could not decode {file.name}. Please check file encoding.")
                continue
            
            # Basic data cleaning
            df = df.copy()
            
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            
            # Remove completely empty rows
            df = df.dropna(how='all')
            
            # Data type inference improvements
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Try to convert to numeric
                    numeric_series = pd.to_numeric(df[col], errors='coerce')
                    if not numeric_series.isna().all():
                        # If most values are numeric, convert
                        non_na_ratio = (~numeric_series.isna()).sum() / len(df)
                        if non_na_ratio > 0.8:
                            df[col] = numeric_series
            
            dataframes[file.name] = df
            quality_reports[file.name] = quality_checker.check_data_quality(df, file.name)
            
            st.success(f"✅ Loaded {file.name}: {df.shape[0]} rows, {df.shape[1]} columns")
            
        except Exception as e:
            st.error(f"❌ Error loading {file.name}: {str(e)}")
            continue
    
    return dataframes, quality_reports

def create_advanced_mapping_interface(dataframes: Dict[str, pd.DataFrame], 
                                    matches: Dict[str, Dict[str, Tuple[str, float, str]]],
                                    matcher: EnhancedVariableMatcher) -> Dict[str, Dict[str, str]]:
    """Advanced mapping interface with data preview and validation."""
    st.subheader("📋 Advanced Column Mapping Interface")
    
    canonical_options = ["unmapped"] + list(matcher.canonical_vars.keys())
    final_mappings = {}
    
    # Global mapping suggestions
    with st.expander("🎯 Smart Mapping Suggestions", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🤖 Apply AI Suggestions (Threshold ≥ 80%)", type="secondary"):
                for filename in matches.keys():
                    for col, (canonical, score, strategy) in matches[filename].items():
                        if score >= 80 and canonical != "unmapped":
                            st.session_state[f"mapping_{filename}_{col}"] = canonical
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset All Mappings", type="secondary"):
                for filename in matches.keys():
                    for col in dataframes[filename].columns:
                        if f"mapping_{filename}_{col}" in st.session_state:
                            del st.session_state[f"mapping_{filename}_{col}"]
                st.rerun()
    
    # File-by-file mapping
    for filename, df in dataframes.items():
        with st.expander(f"📄 {filename} ({df.shape[0]} rows × {df.shape[1]} cols)", expanded=True):
            
            # Show sample data
            st.markdown("**Sample Data Preview:**")
            st.dataframe(df.head(3), use_container_width=True)
            
            # Create mapping table with enhanced information
            mapping_data = []
            file_matches = matches.get(filename, {})
            
            for col in df.columns:
                match_info = file_matches.get(col, ("unmapped", 0, "none"))
                canonical, score, strategy = match_info
                
                # Get data preview for this column
                sample_values = df[col].dropna().head(5).tolist()
                data_type = str(df[col].dtype)
                missing_pct = df[col].isnull().sum() / len(df) * 100
                
                mapping_data.append({
                    "Original Name": col,
                    "Sample Values": str(sample_values)[:50] + "..." if len(str(sample_values)) > 50 else str(sample_values),
                    "Data Type": data_type,
                    "Missing %": f"{missing_pct:.1f}%",
                    "Suggested Mapping": canonical,
                    "Confidence": f"{score:.1f}%",
                    "Strategy": strategy.replace('_', ' ').title()
                })
            
            mapping_df = pd.DataFrame(mapping_data)
            st.dataframe(mapping_df, use_container_width=True)
            
            # Interactive mapping controls
            st.markdown("**Adjust Mappings:**")
            file_mapping = {}
            
            # Create columns for better layout
            n_cols = min(3, len(df.columns))
            cols = st.columns(n_cols)
            
            for i, col in enumerate(df.columns):
                with cols[i % n_cols]:
                    current_match = file_matches.get(col, ("unmapped", 0, "none"))[0]
                    
                    # Show additional context
                    unique_vals = df[col].nunique()
                    total_vals = len(df[col].dropna())
                    
                    key = f"mapping_{filename}_{col}"
                    default_idx = canonical_options.index(current_match) if current_match in canonical_options else 0
                    
                    override = st.selectbox(
                        f"**{col}**\n`{unique_vals} unique / {total_vals} non-null`",
                        options=canonical_options,
                        index=default_idx,
                        key=key,
                        help=f"Current suggestion: {current_match} ({file_matches.get(col, ('', 0, ''))[1]:.1f}% confidence)"
                    )
                    
                    file_mapping[col] = override
                    
                    # Show canonical variable info
                    if override != "unmapped" and override in matcher.canonical_vars:
                        var_info = matcher.canonical_vars[override]
                        st.caption(f"📝 {var_info['description']}")
                        if var_info.get('range'):
                            st.caption(f"📊 Expected range: {var_info['range']}")
            
            final_mappings[filename] = file_mapping
    
    return final_mappings

def validate_mappings(dataframes: Dict[str, pd.DataFrame], 
                     mappings: Dict[str, Dict[str, str]],
                     matcher: EnhancedVariableMatcher) -> Dict[str, List[str]]:
    """Validate mappings against expected data types and ranges."""
    validation_issues = {}
    
    for filename, df in dataframes.items():
        issues = []
        file_mapping = mappings[filename]
        
        for orig_col, canonical in file_mapping.items():
            if canonical == "unmapped":
                continue
                
            if canonical not in matcher.canonical_vars:
                continue
                
            var_info = matcher.canonical_vars[canonical]
            col_data = df[orig_col].dropna()
            
            if len(col_data) == 0:
                issues.append(f"Column '{orig_col}' mapped to '{canonical}' is completely empty")
                continue
            
            # Type validation
            expected_type = var_info.get('type', 'unknown')
            
            if expected_type == 'numeric':
                if not pd.api.types.is_numeric_dtype(col_data):
                    non_numeric = col_data[pd.to_numeric(col_data, errors='coerce').isna()]
                    if len(non_numeric) > 0:
                        issues.append(f"'{orig_col}' → '{canonical}': Expected numeric, found non-numeric values: {list(non_numeric.head(3))}")
                
                # Range validation
                if 'range' in var_info and pd.api.types.is_numeric_dtype(col_data):
                    min_val, max_val = var_info['range']
                    out_of_range = col_data[(col_data < min_val) | (col_data > max_val)]
                    if len(out_of_range) > 0:
                        issues.append(f"'{orig_col}' → '{canonical}': {len(out_of_range)} values outside expected range [{min_val}, {max_val}]")
            
            elif expected_type == 'categorical':
                if 'categories' in var_info:
                    expected_cats = [str(cat).lower() for cat in var_info['categories']]
                    actual_cats = col_data.astype(str).str.lower().unique()
                    unexpected = [cat for cat in actual_cats if cat not in expected_cats]
                    if unexpected:
                        issues.append(f"'{orig_col}' → '{canonical}': Unexpected categories: {unexpected[:5]}")
        
        if issues:
            validation_issues[filename] = issues
    
    return validation_issues

def create_enhanced_visualizations(df: pd.DataFrame) -> None:
    """Create enhanced interactive visualizations with Plotly."""
    st.subheader("📊 Enhanced Data Visualizations")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if not col.startswith('_') and 'participant_id' not in col.lower()]
    
    if len(numeric_cols) == 0:
        st.warning("No numeric columns found for visualization.")
        return
    
    # Visualization controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        viz_type = st.selectbox(
            "Visualization Type",
            ["Distribution Plots", "Correlation Matrix", "Scatter Plots", "Box Plots", "Time Series"]
        )
    
    with col2:
        if viz_type in ["Scatter Plots", "Box Plots"]:
            selected_vars = st.multiselect(
                "Select Variables",
                numeric_cols,
                default=numeric_cols[:min(4, len(numeric_cols))]
            )
        else:
            selected_vars = numeric_cols
    
    with col3:
        if '_source_file' in df.columns:
            color_by_source = st.checkbox("Color by Source File", value=True)
        else:
            color_by_source = False
    
    if viz_type == "Distribution Plots":
        create_distribution_plots(df, selected_vars, color_by_source)
    elif viz_type == "Correlation Matrix":
        create_correlation_matrix(df, selected_vars)
    elif viz_type == "Scatter Plots":
        create_scatter_plots(df, selected_vars, color_by_source)
    elif viz_type == "Box Plots":
        create_box_plots(df, selected_vars, color_by_source)
    elif viz_type == "Time Series":
        create_time_series_plots(df, selected_vars)

def create_distribution_plots(df: pd.DataFrame, columns: List[str], color_by_source: bool):
    """Create interactive distribution plots."""
    n_cols = min(2, len(columns))
    
    for i in range(0, len(columns), n_cols):
        cols_subset = columns[i:i+n_cols]
        fig_cols = st.columns(n_cols)
        
        for j, col in enumerate(cols_subset):
            with fig_cols[j]:
                valid_data = df[col].dropna()
                
                if len(valid_data) == 0:
                    st.warning(f"No valid data for {col}")
                    continue
                
                if color_by_source and '_source_file' in df.columns:
                    fig = px.histogram(
                        df.dropna(subset=[col]), 
                        x=col, 
                        color='_source_file',
                        title=f'Distribution of {col}',
                        marginal="box",
                        nbins=30
                    )
                else:
                    fig = px.histogram(
                        valid_data.to_frame(), 
                        x=col,
                        title=f'Distribution of {col}',
                        marginal="box",
                        nbins=30
                    )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

def create_correlation_matrix(df: pd.DataFrame, columns: List[str]):
    """Create interactive correlation matrix."""
    if len(columns) < 2:
        st.warning("Need at least 2 numeric columns for correlation matrix.")
        return
    
    corr_data = df[columns].corr()
    
    fig = px.imshow(
        corr_data,
        labels=dict(color="Correlation"),
        title="Correlation Matrix",
        color_continuous_scale="RdBu",
        aspect="auto"
    )
    
    # Add correlation values as text
    fig.update_traces(text=np.around(corr_data.values, decimals=2), texttemplate="%{text}")
    fig.update_layout(height=max(400, len(columns) * 40))
    
    st.plotly_chart(fig, use_container_width=True)

def create_scatter_plots(df: pd.DataFrame, columns: List[str], color_by_source: bool):
    """Create interactive scatter plots."""
    if len(columns) < 2:
        st.warning("Need at least 2 variables for scatter plots.")
        return
    
    # Select pairs of variables
    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox("X Variable", columns, key="scatter_x")
    with col2:
        y_var = st.selectbox("Y Variable", [col for col in columns if col != x_var], key="scatter_y")
    
    plot_df = df[[x_var, y_var]].dropna()
    if '_source_file' in df.columns:
        plot_df['_source_file'] = df.loc[plot_df.index, '_source_file']
    
    if len(plot_df) == 0:
        st.warning("No valid data for selected variables.")
        return
    
    if color_by_source and '_source_file' in plot_df.columns:
        fig = px.scatter(
            plot_df, 
            x=x_var, 
            y=y_var, 
            color='_source_file',
            title=f'{y_var} vs {x_var}',
            trendline="ols"
        )
    else:
        fig = px.scatter(
            plot_df, 
            x=x_var, 
            y=y_var,
            title=f'{y_var} vs {x_var}',
            trendline="ols"
        )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def create_box_plots(df: pd.DataFrame, columns: List[str], color_by_source: bool):
    """Create interactive box plots."""
    if color_by_source and '_source_file' in df.columns:
        for col in columns[:4]:  # Limit to 4 plots
            valid_data = df[[col, '_source_file']].dropna()
            if len(valid_data) == 0:
                continue
                
            fig = px.box(valid_data, y=col, color='_source_file', title=f'Distribution of {col} by Source')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    else:
        # Create side-by-side box plots
        n_cols = min(2, len(columns))
        for i in range(0, len(columns[:6]), n_cols):  # Limit to 6 plots
            cols_subset = columns[i:i+n_cols]
            fig_cols = st.columns(n_cols)
            
            for j, col in enumerate(cols_subset):
                with fig_cols[j]:
                    valid_data = df[col].dropna()
                    if len(valid_data) == 0:
                        continue
                    
                    fig = px.box(y=valid_data, title=f'Distribution of {col}')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

def create_time_series_plots(df: pd.DataFrame, columns: List[str]):
    """Create time series plots if date columns are available."""
    date_cols = []
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            # Try to parse as datetime
            try:
                pd.to_datetime(df[col].dropna().head())
                date_cols.append(col)
            except:
                continue
    
    if not date_cols:
        st.info("No date/time columns detected for time series analysis.")
        return
    
    date_col = st.selectbox("Select Date Column", date_cols)
    
    if len(columns) == 0:
        st.warning("No numeric columns available for time series.")
        return
    
    # Convert date column
    df_ts = df.copy()
    df_ts[date_col] = pd.to_datetime(df_ts[date_col], errors='coerce')
    df_ts = df_ts.dropna(subset=[date_col])
    
    if len(df_ts) == 0:
        st.warning("No valid dates found.")
        return
    
    # Plot up to 3 variables
    for col in columns[:3]:
        plot_data = df_ts[[date_col, col]].dropna()
        if len(plot_data) == 0:
            continue
            
        fig = px.line(plot_data, x=date_col, y=col, title=f'{col} Over Time')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def create_comprehensive_summary(merged_df: pd.DataFrame, 
                               quality_reports: Dict[str, Dict],
                               validation_issues: Dict[str, List[str]]) -> None:
    """Create comprehensive data summary with quality metrics."""
    st.subheader("📈 Comprehensive Data Summary")
    
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Dataset Size</h3>
            <p><strong>{:,}</strong> rows<br>
            <strong>{}</strong> columns</p>
        </div>
        """.format(len(merged_df), len(merged_df.columns)), unsafe_allow_html=True)
    
    with col2:
        numeric_cols = len(merged_df.select_dtypes(include=[np.number]).columns)
        categorical_cols = len(merged_df.select_dtypes(include=['object']).columns)
        st.markdown("""
        <div class="metric-card">
            <h3>🔢 Data Types</h3>
            <p><strong>{}</strong> numeric<br>
            <strong>{}</strong> categorical</p>
        </div>
        """.format(numeric_cols, categorical_cols), unsafe_allow_html=True)
    
    with col3:
        total_missing = merged_df.isnull().sum().sum()
        missing_pct = total_missing / (len(merged_df) * len(merged_df.columns)) * 100
        st.markdown("""
        <div class="metric-card">
            <h3>❓ Missing Data</h3>
            <p><strong>{:,}</strong> values<br>
            <strong>{:.1f}%</strong> of total</p>
        </div>
        """.format(total_missing, missing_pct), unsafe_allow_html=True)
    
    with col4:
        total_issues = sum(len(issues) for issues in validation_issues.values())
        issue_color = "warning-card" if total_issues > 0 else "success-card"
        st.markdown("""
        <div class="{}">
            <h3>⚠️ Data Issues</h3>
            <p><strong>{}</strong> issues<br>
            across {} files</p>
        </div>
        """.format(issue_color, total_issues, len(validation_issues)), unsafe_allow_html=True)
    
    # Quality issues by file
    if validation_issues:
        st.markdown("### 🚨 Data Quality Issues")
        for filename, issues in validation_issues.items():
            if issues:
                with st.expander(f"Issues in {filename} ({len(issues)} issues)"):
                    for issue in issues:
                        st.warning(f"⚠️ {issue}")
    
    # Missing data heatmap
    if merged_df.isnull().sum().sum() > 0:
        st.markdown("### 🎯 Missing Data Pattern")
        
        # Calculate missing data percentage by column
        missing_data = merged_df.isnull().sum().sort_values(ascending=False)
        missing_pct = (missing_data / len(merged_df) * 100).round(1)
        
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing Percentage': missing_pct.values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        
        if len(missing_df) > 0:
            fig = px.bar(missing_df.head(15), 
                        x='Missing Percentage', 
                        y='Column',
                        title='Missing Data by Column (Top 15)',
                        orientation='h')
            fig.update_layout(height=max(400, len(missing_df.head(15)) * 25))
            st.plotly_chart(fig, use_container_width=True)

def create_advanced_export_options(merged_df: pd.DataFrame, 
                                 mappings: Dict[str, Dict[str, str]],
                                 quality_reports: Dict[str, Dict],
                                 matcher: EnhancedVariableMatcher) -> None:
    """Create advanced export options with multiple formats and metadata."""
    st.subheader("💾 Advanced Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    # Export options
    with col1:
        st.markdown("**📊 Data Exports**")
        
        # Harmonized CSV
        if not merged_df.empty:
            csv_buffer = io.StringIO()
            merged_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📥 Harmonized Data (CSV)",
                data=csv_data,
                file_name=f"harmonized_rct_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help="Download the merged and harmonized dataset"
            )
        
        # Excel export with multiple sheets
        if not merged_df.empty:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                # Main data
                merged_df.to_excel(writer, sheet_name='Harmonized_Data', index=False)
                
                # Summary statistics
                summary_stats = create_summary_statistics(merged_df)
                if not summary_stats.empty:
                    summary_stats.to_excel(writer, sheet_name='Summary_Statistics')
                
                # Variable mappings
                mapping_data = []
                for filename, mapping in mappings.items():
                    for original, canonical in mapping.items():
                        mapping_data.append({
                            "Source_File": filename,
                            "Original_Name": original,
                            "Canonical_Name": canonical
                        })
                
                if mapping_data:
                    pd.DataFrame(mapping_data).to_excel(writer, sheet_name='Variable_Mappings', index=False)
            
            excel_data = excel_buffer.getvalue()
            st.download_button(
                label="📊 Complete Report (Excel)",
                data=excel_data,
                file_name=f"rct_harmonization_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col2:
        st.markdown("**📋 Documentation**")
        
        # Enhanced codebook
        codebook_data = []
        for filename, mapping in mappings.items():
            for original, canonical in mapping.items():
                var_info = matcher.canonical_vars.get(canonical, {})
                codebook_data.append({
                    "Source_File": filename,
                    "Original_Name": original,
                    "Canonical_Name": canonical,
                    "Description": var_info.get('description', ''),
                    "Expected_Type": var_info.get('type', ''),
                    "Expected_Range": str(var_info.get('range', '')),
                    "Unit": var_info.get('unit', ''),
                    "Required": var_info.get('required', False)
                })
        
        if codebook_data:
            codebook_df = pd.DataFrame(codebook_data)
            codebook_buffer = io.StringIO()
            codebook_df.to_csv(codebook_buffer, index=False)
            codebook_csv = codebook_buffer.getvalue()
            
            st.download_button(
                label="📖 Enhanced Codebook (CSV)",
                data=codebook_csv,
                file_name=f"enhanced_codebook_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        
        # Quality report
        quality_report = create_quality_report_text(quality_reports, mappings)
        st.download_button(
            label="📋 Quality Report (TXT)",
            data=quality_report,
            file_name=f"data_quality_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain"
        )
    
    with col3:
        st.markdown("**⚙️ Configuration**")
        
        # Export configuration JSON
        config = {
            "harmonization_date": datetime.now().isoformat(),
            "files_processed": list(mappings.keys()),
            "canonical_variables": list(matcher.canonical_vars.keys()),
            "mappings": mappings,
            "data_shape": {
                "rows": len(merged_df),
                "columns": len(merged_df.columns)
            }
        }
        
        config_json = json.dumps(config, indent=2)
        st.download_button(
            label="⚙️ Configuration (JSON)",
            data=config_json,
            file_name=f"harmonization_config_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )

def create_quality_report_text(quality_reports: Dict[str, Dict], 
                             mappings: Dict[str, Dict[str, str]]) -> str:
    """Create a comprehensive quality report in text format."""
    report = []
    report.append("=" * 60)
    report.append("RCT DATA HARMONIZATION QUALITY REPORT")
    report.append("=" * 60)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Summary
    report.append("SUMMARY")
    report.append("-" * 20)
    report.append(f"Files processed: {len(quality_reports)}")
    total_rows = sum(qr['total_rows'] for qr in quality_reports.values())
    total_cols = sum(qr['total_cols'] for qr in quality_reports.values())
    report.append(f"Total rows: {total_rows:,}")
    report.append(f"Total columns: {total_cols}")
    report.append("")
    
    # File-by-file details
    for filename, qr in quality_reports.items():
        report.append(f"FILE: {filename}")
        report.append("-" * (len(filename) + 6))
        report.append(f"Dimensions: {qr['total_rows']} rows × {qr['total_cols']} columns")
        report.append(f"Duplicate rows: {qr['duplicate_rows']}")
        
        if qr['empty_columns']:
            report.append(f"Empty columns: {', '.join(qr['empty_columns'])}")
        
        if qr['potential_id_columns']:
            report.append(f"Potential ID columns: {', '.join(qr['potential_id_columns'])}")
        
        if qr['issues']:
            report.append("Data Quality Issues:")
            for issue in qr['issues']:
                report.append(f"  - {issue}")
        
        # Variable mappings
        file_mappings = mappings.get(filename, {})
        mapped_vars = [canonical for canonical in file_mappings.values() if canonical != "unmapped"]
        unmapped_vars = [orig for orig, canonical in file_mappings.items() if canonical == "unmapped"]
        
        report.append(f"Mapped variables: {len(mapped_vars)}")
        report.append(f"Unmapped variables: {len(unmapped_vars)}")
        
        if unmapped_vars:
            report.append(f"  Unmapped: {', '.join(unmapped_vars)}")
        
        report.append("")
    
    return "\n".join(report)

def create_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced summary statistics with more comprehensive metrics."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if not col.startswith('_')]
    
    if len(numeric_cols) == 0:
        return pd.DataFrame()
    
    summary = df[numeric_cols].describe().T
    
    # Add additional metrics
    summary['missing_count'] = df[numeric_cols].isnull().sum()
    summary['missing_percent'] = (summary['missing_count'] / len(df) * 100).round(2)
    summary['unique_count'] = df[numeric_cols].nunique()
    summary['data_type'] = [str(df[col].dtype) for col in numeric_cols]
    
    # Add skewness and kurtosis
    try:
        from scipy import stats
        summary['skewness'] = df[numeric_cols].skew().round(3)
        summary['kurtosis'] = df[numeric_cols].kurtosis().round(3)
    except ImportError:
        pass  # Skip if scipy not available
    
    # Outlier detection (IQR method)
    outlier_counts = []
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        outlier_counts.append(len(outliers))
    
    summary['outliers_count'] = outlier_counts
    summary['outliers_percent'] = (pd.Series(outlier_counts) / len(df) * 100).round(2)
    
    return summary.round(3)

def main():
    """Enhanced main application function."""
    
    # App header with improved styling
    st.markdown("""
    <div class="main-header">
        <h1>🔄 RCT Data Harmonizer Pro</h1>
        <p style="font-size: 1.2em; color: #666;">
            Advanced harmonization and integration of randomized controlled trial datasets
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    with st.expander("✨ New Features & Improvements", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **🎯 Enhanced Matching:**
            - Multiple matching strategies (exact, fuzzy, semantic, pattern)
            - Improved canonical variable library
            - Confidence scoring and strategy tracking
            
            **🔍 Data Quality:**
            - Comprehensive validation checks
            - Outlier detection
            - Missing data analysis
            - Data type inference
            """)
        with col2:
            st.markdown("""
            **📊 Advanced Visualizations:**
            - Interactive Plotly charts
            - Correlation matrices
            - Distribution comparisons
            - Time series analysis
            
            **💾 Export Options:**
            - Excel reports with multiple sheets
            - Enhanced codebooks
            - Quality reports
            - Configuration backups
            """)
    
    # Initialize enhanced components
    matcher = EnhancedVariableMatcher()
    
    # Sidebar for file upload and controls
    st.sidebar.header("📁 File Upload & Settings")
    uploaded_files = st.sidebar.file_uploader(
        "Choose CSV files",
        type="csv",
        accept_multiple_files=True,
        help="Upload multiple CSV files to harmonize and merge"
    )
    
    if not uploaded_files:
        st.info("👆 Please upload one or more CSV files to get started.")
        
        # Show example data structure
        with st.expander("📋 Expected Data Structure Examples"):
            st.markdown("""
            **Your CSV files should contain columns like:**
            
            - **Identifiers**: participant_id, subject_id, id
            - **Demographics**: age, gender, height, weight
            - **Clinical measures**: systolic_bp, diastolic_bp, bmi
            - **Study variables**: treatment_group, baseline_weight, follow_up_weight
            - **Outcomes**: dropout, diet_adherence, physical_activity
            
            The tool will automatically detect and suggest mappings for these variables.
            """)
        return
    
    # Display uploaded files with enhanced info
    st.sidebar.subheader("📊 Uploaded Files")
    file_info = []
    total_size = 0
    for file in uploaded_files:
        size_mb = file.size / (1024 * 1024)
        total_size += size_mb
        file_info.append({
            "Filename": file.name, 
            "Size (MB)": f"{size_mb:.2f}",
            "Status": "✅ Ready"
        })
    
    st.sidebar.dataframe(pd.DataFrame(file_info), use_container_width=True)
    st.sidebar.metric("Total Size", f"{total_size:.2f} MB")
    
    # Advanced settings
    st.sidebar.subheader("⚙️ Advanced Settings")
    match_threshold = st.sidebar.slider(
        "Fuzzy Match Threshold",
        min_value=50.0,
        max_value=100.0,
        value=75.0,
        step=5.0,
        help="Minimum similarity score for automatic matching"
    )
    
    auto_clean = st.sidebar.checkbox(
        "Auto Clean Data",
        value=True,
        help="Automatically clean data (remove empty rows/cols, infer types)"
    )
    
    show_advanced_stats = st.sidebar.checkbox(
        "Show Advanced Statistics",
        value=True,
        help="Include skewness, kurtosis, and outlier detection"
    )
    
    # Load data with enhanced validation
    with st.spinner("🔄 Loading and validating CSV files..."):
        dataframes, quality_reports = load_data_with_validation(uploaded_files)
    
    if not dataframes:
        st.error("❌ No valid CSV files were loaded. Please check your files and try again.")
        return
    
    # Show quality assessment
    st.subheader("📋 Data Quality Assessment")
    
    quality_tabs = st.tabs([f"📄 {filename}" for filename in quality_reports.keys()])
    
    for tab, (filename, qr) in zip(quality_tabs, quality_reports.items()):
        with tab:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Rows", f"{qr['total_rows']:,}")
                st.metric("Columns", qr['total_cols'])
            
            with col2:
                st.metric("Duplicates", qr['duplicate_rows'])
                st.metric("Empty Cols", len(qr['empty_columns']))
            
            with col3:
                if qr['potential_id_columns']:
                    st.success(f"✅ ID columns found: {', '.join(qr['potential_id_columns'])}")
                else:
                    st.warning("⚠️ No clear ID column detected")
            
            if qr['issues']:
                st.warning("**Data Quality Issues:**")
                for issue in qr['issues']:
                    st.write(f"• {issue}")
            else:
                st.success("✅ No major data quality issues detected")
    
    # Perform enhanced fuzzy matching
    with st.spinner("🤖 Performing intelligent column matching..."):
        matches = matcher.match_columns(dataframes, match_threshold)
    
    # Show matching summary
    st.subheader("🎯 Matching Summary")
    matching_summary = []
    for filename, file_matches in matches.items():
        mapped = sum(1 for canonical, _, _ in file_matches.values() if canonical != "unmapped")
        total = len(file_matches)
        matching_summary.append({
            "File": filename,
            "Total Columns": total,
            "Mapped": mapped,
            "Unmapped": total - mapped,
            "Success Rate": f"{mapped/total*100:.1f}%"
        })
    
    summary_df = pd.DataFrame(matching_summary)
    st.dataframe(summary_df, use_container_width=True)
    
    # Enhanced mapping interface
    final_mappings = create_advanced_mapping_interface(dataframes, matches, matcher)
    
    # Validate mappings
    validation_issues = validate_mappings(dataframes, final_mappings, matcher)
    
    # Show validation results
    if validation_issues:
        st.subheader("⚠️ Mapping Validation")
        total_issues = sum(len(issues) for issues in validation_issues.values())
        
        if total_issues > 0:
            st.warning(f"Found {total_issues} potential issues with current mappings:")
            
            for filename, issues in validation_issues.items():
                if issues:
                    with st.expander(f"Issues in {filename}"):
                        for issue in issues:
                            st.write(f"• {issue}")
        else:
            st.success("✅ All mappings passed validation checks")
    
    # Harmonize and merge with enhanced processing
    if st.button("🚀 Harmonize and Merge Data", type="primary", use_container_width=True):
        with st.spinner("🔄 Harmonizing and merging datasets..."):
            merged_df = harmonize_dataframes(dataframes, final_mappings)
        
        if merged_df.empty:
            st.error("❌ Failed to merge datasets. Please check your data and mappings.")
            return
        
        # Store in session state
        st.session_state.merged_df = merged_df
        st.session_state.final_mappings = final_mappings
        st.session_state.quality_reports = quality_reports
        st.session_state.validation_issues = validation_issues
        st.session_state.matcher = matcher
        
        st.success(f"✅ Successfully harmonized and merged {len(dataframes)} datasets!")
        st.balloons()
    
    # Display results if available
    if 'merged_df' in st.session_state:
        merged_df = st.session_state.merged_df
        final_mappings = st.session_state.final_mappings
        quality_reports = st.session_state.quality_reports
        validation_issues = st.session_state.validation_issues
        matcher = st.session_state.matcher
        
        # Comprehensive summary
        create_comprehensive_summary(merged_df, quality_reports, validation_issues)
        
        # Data preview with search and filtering
        st.subheader("👀 Interactive Data Preview")
        
        # Add search and filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            search_term = st.text_input("🔍 Search in data", placeholder="Enter search term...")
        with col2:
            if '_source_file' in merged_df.columns:
                source_filter = st.multiselect(
                    "📁 Filter by source",
                    merged_df['_source_file'].unique(),
                    default=merged_df['_source_file'].unique()
                )
            else:
                source_filter = None
        with col3:
            show_rows = st.selectbox("Show rows", [10, 25, 50, 100], index=1)
        
        # Apply filters
        display_df = merged_df.copy()
        
        if source_filter and '_source_file' in display_df.columns:
            display_df = display_df[display_df['_source_file'].isin(source_filter)]
        
        if search_term:
            # Search across all string columns
            mask = False
            for col in display_df.select_dtypes(include=['object']).columns:
                mask |= display_df[col].astype(str).str.contains(search_term, case=False, na=False)
            display_df = display_df[mask]
        
        st.dataframe(display_df.head(show_rows), use_container_width=True)
        st.caption(f"Showing {min(show_rows, len(display_df))} of {len(display_df)} rows")
        
        # Enhanced summary statistics
        st.subheader("📊 Enhanced Summary Statistics")
        summary_stats = create_summary_statistics(merged_df)
        if not summary_stats.empty:
            # Make statistics interactive
            st.dataframe(summary_stats, use_container_width=True)
            
            # Highlight potential issues
            if show_advanced_stats:
                st.markdown("**📈 Data Quality Insights:**")
                
                high_missing = summary_stats[summary_stats['missing_percent'] > 25]
                if len(high_missing) > 0:
                    st.warning(f"⚠️ Variables with >25% missing data: {', '.join(high_missing.index.tolist())}")
                
                high_outliers = summary_stats[summary_stats['outliers_percent'] > 10]
                if len(high_outliers) > 0:
                    st.info(f"📊 Variables with >10% outliers: {', '.join(high_outliers.index.tolist())}")
        else:
            st.info("ℹ️ No numeric columns found for summary statistics.")
        
        # Enhanced visualizations
        create_enhanced_visualizations(merged_df)
        
        # Advanced export options
        create_advanced_export_options(merged_df, final_mappings, quality_reports, matcher)
        
        # Additional insights
        with st.expander("🔍 Detailed Dataset Information"):
            st.markdown("**Column Details:**")
            col_details = []
            for col in merged_df.columns:
                dtype = str(merged_df[col].dtype)
                null_count = merged_df[col].isnull().sum()
                null_pct = (null_count / len(merged_df) * 100)
                unique_count = merged_df[col].nunique()
                
                col_details.append({
                    "Column": col,
                    "Data Type": dtype,
                    "Non-Null": len(merged_df) - null_count,
                    "Null %": f"{null_pct:.1f}%",
                    "Unique Values": unique_count,
                    "Sample Values": str(merged_df[col].dropna().head(3).tolist())[:50] + "..."
                })
            
            st.dataframe(pd.DataFrame(col_details), use_container_width=True)

def harmonize_dataframes(dataframes: Dict[str, pd.DataFrame], 
                        mappings: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """Enhanced harmonization with better conflict resolution."""
    harmonized_dfs = []
    
    for filename, df in dataframes.items():
        df_harm = df.copy()
        df_harm['_source_file'] = filename
        
        # Apply column mappings
        mapping = mappings[filename]
        rename_dict = {orig: canonical for orig, canonical in mapping.items() 
                      if canonical != "unmapped"}
        
        df_harm = df_harm.rename(columns=rename_dict)
        
        # Drop unmapped columns
        unmapped_cols = [col for col, canonical in mapping.items() if canonical == "unmapped"]
        df_harm = df_harm.drop(columns=unmapped_cols, errors='ignore')
        
        harmonized_dfs.append(df_harm)
    
    if not harmonized_dfs:
        return pd.DataFrame()
    
    # Enhanced merging strategy
    has_participant_id = all('participant_id' in df.columns for df in harmonized_dfs)
    
    if has_participant_id and len(harmonized_dfs) > 1:
        # Merge on participant_id with intelligent suffix handling
        merged_df = harmonized_dfs[0]
        
        for i, df in enumerate(harmonized_dfs[1:], 1):
            # Smart suffix generation based on actual conflicts
            overlap_cols = set(merged_df.columns) & set(df.columns)
            overlap_cols.discard('participant_id')
            overlap_cols.discard('_source_file')
            
            if overlap_cols:
                # Create meaningful suffixes
                left_file = merged_df['_source_file'].iloc[0] if '_source_file' in merged_df.columns else f'file_{i-1}'
                right_file = df['_source_file'].iloc[0] if '_source_file' in df.columns else f'file_{i}'
                
                left_suffix = f'_{left_file.replace(".csv", "").replace(" ", "_")}'
                right_suffix = f'_{right_file.replace(".csv", "").replace(" ", "_")}'
                
                merged_df = pd.merge(
                    merged_df, df,
                    on='participant_id',
                    how='outer',
                    suffixes=(left_suffix, right_suffix)
                )
            else:
                # No conflicts, simple merge
                merged_df = pd.merge(merged_df, df, on='participant_id', how='outer')
    else:
        # Concatenate with file identifiers
        for i, df in enumerate(harmonized_dfs):
            if i > 0:
                # Add file prefix to avoid conflicts
                file_prefix = df['_source_file'].iloc[0].replace('.csv', '').replace(' ', '_') if '_source_file' in df.columns else f'file_{i}'
                cols_to_rename = [col for col in df.columns if col != '_source_file']
                rename_dict = {col: f"{col}_{file_prefix}" for col in cols_to_rename}
                df = df.rename(columns=rename_dict)
                harmonized_dfs[i] = df
        
        merged_df = pd.concat(harmonized_dfs, ignore_index=True, sort=False)
    
    # Final cleanup of any remaining duplicated columns
    if merged_df.columns.duplicated().any():
        cols = pd.Series(merged_df.columns)
        for dup in cols[cols.duplicated()].unique():
            indices = cols[cols == dup].index
            for j, idx in enumerate(indices[1:], 1):
                cols.iloc[idx] = f"{dup}_duplicate_{j}"
        merged_df.columns = cols
    
    return merged_df

if __name__ == "__main__":
    main()