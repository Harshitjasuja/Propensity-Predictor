import pandas as pd
import numpy as np
import os

def check_file_exists(filepath):
    """Check if file exists"""
    if os.path.exists(filepath):
        print(f"✓ File found: {filepath}")
        return True
    else:
        print(f"✗ File not found: {filepath}")
        return False

def check_schema(df, expected_columns=None):
    """Check dataframe schema and basic info"""
    print("=== SCHEMA VALIDATION ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Total columns: {len(df.columns)}")
    
    if expected_columns:
        actual_cols = set(df.columns)
        expected_cols = set(expected_columns)
        missing_cols = expected_cols - actual_cols
        extra_cols = actual_cols - expected_cols
        
        if missing_cols:
            print(f"✗ Missing columns: {missing_cols}")
        if extra_cols:
            print(f"ℹ Extra columns: {len(extra_cols)} additional columns found")
        if not missing_cols:
            print("✓ All expected columns present")
    
    print("\nColumn types:")
    print(df.dtypes.value_counts())

def check_missing_values(df):
    """Check for missing values"""
    print("\n=== MISSING VALUES CHECK ===")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing_Count': missing_data.values,
        'Missing_Percentage': missing_percent.values
    })
    
    # Only show columns with missing values
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
    
    if len(missing_df) > 0:
        print(f"Found {len(missing_df)} columns with missing values:")
        print(missing_df.to_string(index=False))
        
        # Critical missing value check
        critical_missing = missing_df[missing_df['Missing_Percentage'] > 50]
        if len(critical_missing) > 0:
            print(f"\n⚠️ WARNING: {len(critical_missing)} columns have >50% missing values!")
    else:
        print("✓ No missing values found")

def check_duplicates(df):
    """Check for duplicate rows"""
    print("\n=== DUPLICATE CHECK ===")
    total_duplicates = df.duplicated().sum()
    print(f"Total duplicate rows: {total_duplicates}")
    
    if total_duplicates > 0:
        duplicate_percentage = (total_duplicates / len(df)) * 100
        print(f"Duplicate percentage: {duplicate_percentage:.2f}%")
        print("⚠️ Consider removing duplicates during preprocessing")
    else:
        print("✓ No duplicate rows found")

def check_data_quality(df):
    """Check overall data quality issues"""
    print("\n=== DATA QUALITY CHECK ===")
    
    # Check for potential ID columns
    id_columns = [col for col in df.columns if 'id' in col.lower()]
    print(f"Identified ID columns: {id_columns}")
    
    # Check for date columns
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'timestamp' in col.lower()]
    print(f"Identified date columns: {date_columns}")
    
    # Check for categorical columns with high cardinality
    categorical_cols = df.select_dtypes(include=['object']).columns
    high_cardinality_cols = []
    
    for col in categorical_cols:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.5:  # More than 50% unique values
            high_cardinality_cols.append((col, df[col].nunique(), unique_ratio))
    
    if high_cardinality_cols:
        print("\n⚠️ High cardinality categorical columns (may need special handling):")
        for col, unique_count, ratio in high_cardinality_cols:
            print(f"  {col}: {unique_count} unique values ({ratio:.2%} unique)")

def check_target_variable(df, target_col='will_repurchase'):
    """Check target variable for classification problems"""
    print(f"\n=== TARGET VARIABLE CHECK ({target_col}) ===")
    
    if target_col in df.columns:
        target_counts = df[target_col].value_counts()
        target_percentages = df[target_col].value_counts(normalize=True) * 100
        
        print("Class distribution:")
        for class_val in target_counts.index:
            count = target_counts[class_val]
            pct = target_percentages[class_val]
            print(f"  Class {class_val}: {count} ({pct:.2f}%)")
        
        # Check for severe imbalance
        minority_class_pct = target_percentages.min()
        if minority_class_pct < 10:
            print(f"⚠️ WARNING: Severe class imbalance detected!")
            print(f"   Minority class: {minority_class_pct:.2f}%")
            print("   Consider using balancing techniques (SMOTE, etc.)")
        elif minority_class_pct < 30:
            print(f"⚠️ Moderate class imbalance detected: {minority_class_pct:.2f}%")
        else:
            print("✓ Reasonable class balance")
    else:
        print(f"✗ Target column '{target_col}' not found")
        print(f"Available columns: {list(df.columns[:10])}...")  # Show first 10 columns

def check_numerical_distributions(df):
    """Check numerical column distributions for outliers"""
    print("\n=== NUMERICAL DISTRIBUTIONS CHECK ===")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    outlier_summary = []
    
    for col in numerical_cols[:5]:  # Check first 5 numerical columns
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].count()
        outlier_pct = (outliers / len(df)) * 100
        
        outlier_summary.append({
            'Column': col,
            'Outliers': outliers,
            'Outlier_Pct': outlier_pct,
            'Min': df[col].min(),
            'Max': df[col].max()
        })
    
    outlier_df = pd.DataFrame(outlier_summary)
    print(outlier_df.to_string(index=False))

def validate_merged_data(filepath='data/merged_orders.csv'):
    """Main validation function for merged e-commerce data"""
    print("=== STARTING DATA VALIDATION ===")
    
    # Check if file exists
    if not check_file_exists(filepath):
        return False
    
    # Load data
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Successfully loaded data: {df.shape}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False
    
    # Expected columns for e-commerce data (adjust based on your merged data)
    expected_columns = [
        'customer_unique_id', 'order_purchase_timestamp', 'price', 
        'product_category_name', 'customer_city', 'customer_state'
    ]
    
    # Run all validation checks
    check_schema(df, expected_columns)
    check_missing_values(df)
    check_duplicates(df)
    check_data_quality(df)
    check_numerical_distributions(df)
    
    # Check target variable if it exists (after preprocessing it will be created)
    check_target_variable(df)
    
    print("\n=== VALIDATION COMPLETED ===")
    return True

if __name__ == "__main__":
    # Validate the merged data
    is_valid = validate_merged_data('data/merged_orders.csv')
    
    if is_valid:
        print("✅ Data validation completed successfully!")
        print("You can now proceed with preprocessing.")
    else:
        print("❌ Data validation failed!")
        print("Please fix the issues before proceeding.")