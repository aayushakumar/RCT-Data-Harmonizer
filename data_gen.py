import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

def set_random_seed(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)

def generate_study1_nutrition_trial(n_participants=150):
    """
    Generate data for Study 1: Nutrition Intervention Trial
    Focus: Weight loss through dietary intervention
    """
    set_random_seed(42)
    
    data = {
        'subject_id': [f'NUT_{i:03d}' for i in range(1, n_participants + 1)],
        'participant_age': np.random.normal(45, 12, n_participants).astype(int),
        'gender_mf': np.random.choice(['M', 'F'], n_participants, p=[0.4, 0.6]),
        'baseline_weight': np.random.normal(85, 15, n_participants),
        'weight_followup': None,  # Will calculate based on intervention
        'bmi_baseline': None,  # Will calculate
        'final_bmi': None,  # Will calculate
        'diet_compliance': np.random.beta(2, 1, n_participants) * 100,  # 0-100%
        'phys_act': np.random.normal(150, 60, n_participants),  # minutes per week
        'sys_bp': np.random.normal(128, 18, n_participants),
        'dia_bp': np.random.normal(82, 12, n_participants),
        'intervention': np.random.choice(['Control', 'Low-Carb', 'Mediterranean'], n_participants),
        'completed_study': np.random.choice([1, 0], n_participants, p=[0.85, 0.15])
    }
    
    # Calculate height and BMI
    heights = np.random.normal(1.70, 0.08, n_participants)  # meters
    data['bmi_baseline'] = data['baseline_weight'] / (heights ** 2)
    
    # Simulate weight change based on intervention
    weight_changes = []
    for i, intervention in enumerate(data['intervention']):
        if intervention == 'Control':
            change = np.random.normal(0.5, 2.5)  # slight weight gain
        elif intervention == 'Low-Carb':
            change = np.random.normal(-4.2, 3.1)  # moderate weight loss
        else:  # Mediterranean
            change = np.random.normal(-2.8, 2.8)  # mild weight loss
        
        # Factor in compliance
        compliance_factor = data['diet_compliance'][i] / 100
        actual_change = change * compliance_factor + np.random.normal(0, 1)
        weight_changes.append(actual_change)
    
    data['weight_followup'] = data['baseline_weight'] + np.array(weight_changes)
    data['final_bmi'] = data['weight_followup'] / (heights ** 2)
    
    # Add some missing data for realism
    missing_indices = np.random.choice(n_participants, size=int(0.1 * n_participants), replace=False)
    for idx in missing_indices:
        if np.random.random() > 0.5:
            data['weight_followup'][idx] = np.nan
            data['final_bmi'][idx] = np.nan
    
    return pd.DataFrame(data)

def generate_study2_exercise_trial(n_participants=120):
    """
    Generate data for Study 2: Exercise Intervention Trial
    Focus: Physical activity and cardiovascular health
    """
    set_random_seed(123)
    
    data = {
        'pid': [f'EX{i:04d}' for i in range(1, n_participants + 1)],
        'age_years': np.random.normal(52, 14, n_participants).astype(int),
        'sex': np.random.choice(['Male', 'Female'], n_participants, p=[0.45, 0.55]),
        'body_weight': np.random.normal(78, 18, n_participants),
        'follow_weight': None,  # Will calculate
        'body_mass_index': None,  # Will calculate
        'followup_bmi': None,  # Will calculate
        'PA_minutes': np.random.normal(90, 45, n_participants),  # baseline activity
        'activity_change': None,  # Will calculate
        'blood_pressure_sys': np.random.normal(132, 20, n_participants),
        'blood_pressure_dia': np.random.normal(84, 14, n_participants),
        'treatment_arm': np.random.choice(['Control', 'Moderate Exercise', 'High Intensity'], n_participants),
        'dropout': np.random.choice(['No', 'Yes'], n_participants, p=[0.88, 0.12])
    }
    
    # Calculate BMI
    heights = np.random.normal(1.68, 0.09, n_participants)
    data['body_mass_index'] = data['body_weight'] / (heights ** 2)
    
    # Simulate changes based on exercise intervention
    activity_changes = []
    weight_changes = []
    
    for i, treatment in enumerate(data['treatment_arm']):
        if treatment == 'Control':
            act_change = np.random.normal(-5, 20)  # slight decrease
            wt_change = np.random.normal(1.2, 2.0)  # slight weight gain
        elif treatment == 'Moderate Exercise':
            act_change = np.random.normal(80, 35)  # moderate increase
            wt_change = np.random.normal(-1.8, 2.5)  # mild weight loss
        else:  # High Intensity
            act_change = np.random.normal(140, 45)  # large increase
            wt_change = np.random.normal(-3.5, 3.2)  # moderate weight loss
        
        activity_changes.append(act_change)
        weight_changes.append(wt_change)
    
    data['activity_change'] = activity_changes
    data['follow_weight'] = data['body_weight'] + np.array(weight_changes)
    data['followup_bmi'] = data['follow_weight'] / (heights ** 2)
    
    # Add missing data
    missing_indices = np.random.choice(n_participants, size=int(0.08 * n_participants), replace=False)
    for idx in missing_indices:
        data['follow_weight'][idx] = np.nan
        data['followup_bmi'][idx] = np.nan
        data['activity_change'][idx] = np.nan
    
    return pd.DataFrame(data)

def generate_study3_combined_trial(n_participants=200):
    """
    Generate data for Study 3: Combined Diet + Exercise Trial
    Focus: Comprehensive lifestyle intervention
    """
    set_random_seed(456)
    
    data = {
        'participant': [f'COMB_{i:03d}' for i in range(1, n_participants + 1)],
        'age': np.random.normal(48, 16, n_participants).astype(int),
        'male_female': np.random.choice([1, 0], n_participants, p=[0.42, 0.58]),  # 1=Male, 0=Female
        'wt': np.random.normal(82, 20, n_participants),
        'final_weight': None,
        'bmi': np.random.normal(28.5, 5.2, n_participants),
        'bmi_followup': None,
        'nutrition_score': np.random.beta(1.5, 2, n_participants) * 100,
        'activity_mins': np.random.normal(120, 70, n_participants),
        'sbp': np.random.normal(135, 22, n_participants),
        'dbp': np.random.normal(86, 16, n_participants),
        'group': np.random.choice(['Control', 'Diet Only', 'Exercise Only', 'Diet+Exercise'], n_participants),
        'withdrawn': np.random.choice([0, 1], n_participants, p=[0.90, 0.10])  # 0=completed, 1=withdrawn
    }
    
    # Simulate interventions with interaction effects
    weight_changes = []
    bmi_changes = []
    
    for i, group in enumerate(data['group']):
        base_change = 0
        
        if 'Diet' in group:
            diet_effect = -2.5 * (data['nutrition_score'][i] / 100)
            base_change += diet_effect
        
        if 'Exercise' in group:
            exercise_effect = -1.8 * min(data['activity_mins'][i] / 150, 2.0)  # Cap at 2x recommended
            base_change += exercise_effect
        
        if group == 'Diet+Exercise':
            # Synergistic effect
            synergy = -1.2
            base_change += synergy
        
        if group == 'Control':
            base_change = np.random.normal(0.8, 1.5)  # slight weight gain
        
        # Add individual variation
        final_change = base_change + np.random.normal(0, 2.2)
        weight_changes.append(final_change)
        
        # BMI change proportional to weight change
        bmi_change = final_change * 0.35 + np.random.normal(0, 0.8)
        bmi_changes.append(bmi_change)
    
    data['final_weight'] = data['wt'] + np.array(weight_changes)
    data['bmi_followup'] = data['bmi'] + np.array(bmi_changes)
    
    # Add missing data patterns
    missing_indices = np.random.choice(n_participants, size=int(0.12 * n_participants), replace=False)
    for idx in missing_indices:
        if np.random.random() > 0.6:
            data['final_weight'][idx] = np.nan
            data['bmi_followup'][idx] = np.nan
    
    return pd.DataFrame(data)

def generate_study4_minimal_trial(n_participants=80):
    """
    Generate data for Study 4: Minimal dataset with different naming conventions
    Focus: Testing edge cases and minimal overlap
    """
    set_random_seed(789)
    
    data = {
        'id': [f'MIN{i:02d}' for i in range(1, n_participants + 1)],
        'participant_age': np.random.normal(41, 10, n_participants).astype(int),
        'gender': np.random.choice(['M', 'F'], n_participants),
        'weight_kg': np.random.normal(75, 12, n_participants),
        'PA_change': np.random.normal(25, 40, n_participants),
        'adherence': np.random.uniform(60, 95, n_participants),
        'systolic': np.random.normal(125, 15, n_participants),
        'condition': np.random.choice(['A', 'B'], n_participants),
        'status': np.random.choice(['Complete', 'Incomplete'], n_participants, p=[0.92, 0.08])
    }
    
    return pd.DataFrame(data)

def save_sample_datasets():
    """Generate and save all sample datasets as CSV files."""
    
    # Create output directory
    output_dir = "sample_rct_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate datasets
    datasets = {
        "nutrition_study_2023.csv": generate_study1_nutrition_trial(),
        "exercise_intervention_2024.csv": generate_study2_exercise_trial(),
        "lifestyle_combined_trial.csv": generate_study3_combined_trial(),
        "minimal_pilot_study.csv": generate_study4_minimal_trial()
    }
    
    # Save datasets
    for filename, df in datasets.items():
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"✅ Saved {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Show sample of each dataset
        print(f"   Sample columns: {list(df.columns[:5])}...")
        print(f"   Missing data: {df.isnull().sum().sum()} cells")
        print()
    
    print(f"📁 All sample datasets saved to '{output_dir}' directory")
    return datasets

def print_dataset_info():
    """Print information about the generated datasets."""
    
    print("🔬 SAMPLE RCT DATASETS OVERVIEW")
    print("=" * 50)
    
    info = [
        {
            "Study": "Nutrition Study 2023",
            "Focus": "Weight loss through dietary intervention",
            "N": 150,
            "Key Variables": "subject_id, participant_age, baseline_weight, diet_compliance",
            "Interventions": "Control, Low-Carb, Mediterranean",
            "Missing Data": "~10%"
        },
        {
            "Study": "Exercise Intervention 2024", 
            "Focus": "Physical activity and cardiovascular health",
            "N": 120,
            "Key Variables": "pid, age_years, PA_minutes, activity_change",
            "Interventions": "Control, Moderate Exercise, High Intensity",
            "Missing Data": "~8%"
        },
        {
            "Study": "Lifestyle Combined Trial",
            "Focus": "Comprehensive diet + exercise intervention",
            "N": 200,
            "Key Variables": "participant, wt, nutrition_score, activity_mins",
            "Interventions": "Control, Diet Only, Exercise Only, Diet+Exercise",
            "Missing Data": "~12%"
        },
        {
            "Study": "Minimal Pilot Study",
            "Focus": "Testing edge cases with minimal variables",
            "N": 80,
            "Key Variables": "id, weight_kg, PA_change, adherence",
            "Interventions": "A, B",
            "Missing Data": "~0%"
        }
    ]
    
    for study in info:
        print(f"📊 {study['Study']}")
        print(f"   Focus: {study['Focus']}")
        print(f"   Sample Size: {study['N']} participants")
        print(f"   Key Variables: {study['Key Variables']}")
        print(f"   Interventions: {study['Interventions']}")
        print(f"   Missing Data: {study['Missing Data']}")
        print()

def main():
    """Main function to generate sample data."""
    print("🚀 Generating Sample RCT Data for Testing...")
    print()
    
    # Print overview
    print_dataset_info()
    
    # Generate and save datasets
    datasets = save_sample_datasets()
    
    print("💡 TESTING TIPS:")
    print("- Upload all 4 CSV files to test multi-file harmonization")
    print("- Notice how column names vary (e.g., 'subject_id' vs 'pid' vs 'participant')")
    print("- Different intervention naming schemes test mapping flexibility")
    print("- Missing data patterns test robustness")
    print("- Various data types (numeric, categorical, binary) test processing")
    print()
    print("🎯 Expected Harmonization Challenges:")
    print("- participant_id: 'subject_id', 'pid', 'participant', 'id'")
    print("- weight: 'baseline_weight', 'body_weight', 'wt', 'weight_kg'")
    print("- physical activity: 'phys_act', 'PA_minutes', 'activity_mins', 'PA_change'")
    print("- treatment groups: 'intervention', 'treatment_arm', 'group', 'condition'")

if __name__ == "__main__":
    main()