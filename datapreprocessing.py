import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class FastUnbiasedPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_selector = None
        
    def load_and_clean_data(self):
        """Load and clean data efficiently"""
        print("=== LOADING AND CLEANING DATA ===")
        
        df = pd.read_csv('data/merged_orders.csv')
        print(f"Raw data: {df.shape}")
        
        # Clean data
        df = df.drop_duplicates()
        df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
        
        # Fill missing values
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna('Unknown')
        
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(df[col].median())
            
        print(f"After cleaning: {df.shape}")
        return df
    
    def create_target_fast(self, df):
        """Create target variable efficiently using customer order counts"""
        print("=== CREATING TARGET VARIABLE  ===")
        
        # Sort by customer and timestamp
        df = df.sort_values(['customer_unique_id', 'order_purchase_timestamp'])
        
        # Count orders per customer
        customer_order_counts = df.groupby('customer_unique_id').size()
        df['customer_total_orders'] = df['customer_unique_id'].map(customer_order_counts)
        
        # Add order sequence number
        df['order_sequence'] = df.groupby('customer_unique_id').cumcount() + 1
        
        # SIMPLE TARGET CREATION:
        # Use first order from each customer
        # Target = 1 if customer has multiple orders (will repurchase), 0 if single order
        first_orders = df[df['order_sequence'] == 1].copy()
        first_orders['will_repurchase'] = (first_orders['customer_total_orders'] > 1).astype(int)
        
        print(f"Training data (first orders): {first_orders.shape}")
        print(f"Target distribution:\n{first_orders['will_repurchase'].value_counts()}")
        print(f"Positive class: {first_orders['will_repurchase'].mean():.1%}")
        
        return first_orders
    
    def engineer_features(self, df):
        """Create features efficiently"""
        print("=== ENGINEERING FEATURES ===")
        
        # Temporal features
        df['purchase_hour'] = df['order_purchase_timestamp'].dt.hour
        df['purchase_day_of_week'] = df['order_purchase_timestamp'].dt.dayofweek
        df['purchase_month'] = df['order_purchase_timestamp'].dt.month
        df['is_weekend'] = (df['purchase_day_of_week'] >= 5).astype(int)
        
        # Product features
        if 'product_category_name' in df.columns:
            category_counts = df['product_category_name'].value_counts()
            df['category_popularity'] = df['product_category_name'].map(category_counts)
        
        # Payment features
        if 'payment_type' in df.columns:
            df['is_credit_card'] = (df['payment_type'] == 'credit_card').astype(int)
        
        if 'payment_installments' in df.columns:
            df['has_installments'] = (df['payment_installments'] > 1).astype(int)
        
        # Price features
        if 'price' in df.columns:
            df['price_percentile'] = df['price'].rank(pct=True)
            df['is_high_value'] = (df['price'] > df['price'].quantile(0.8)).astype(int)
        
        print(f"After feature engineering: {df.shape}")
        return df
    
    def prepare_final_features(self, df):
        """Prepare features for modeling"""
        print("=== PREPARING FINAL FEATURES ===")
        
        # Drop non-predictive columns
        columns_to_drop = [
            'order_id', 'customer_id', 'customer_unique_id',
            'order_purchase_timestamp', 'order_approved_at',
            'order_delivered_carrier_date', 'order_delivered_customer_date',
            'order_estimated_delivery_date', 'shipping_limit_date',
            'customer_zip_code_prefix', 'seller_id', 'product_id',
            'customer_total_orders', 'order_sequence'  # Helper columns
        ]
        
        columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        df_features = df.drop(columns=columns_to_drop)
        
        # Encode categorical variables
        categorical_cols = df_features.select_dtypes(include=['object']).columns.tolist()
        if 'will_repurchase' in categorical_cols:
            categorical_cols.remove('will_repurchase')
        
        for col in categorical_cols:
            if col in df_features.columns:
                le = LabelEncoder()
                df_features[col] = le.fit_transform(df_features[col].astype(str))
                self.label_encoders[col] = le
        
        print(f"Final features: {df_features.shape}")
        return df_features
    
    def split_and_process(self, df):
        """Split and process data"""
        print("=== SPLITTING AND PROCESSING DATA ===")
        
        # Separate features and target
        X = df.drop('will_repurchase', axis=1)
        y = df['will_repurchase']
        
        print(f"Target distribution: {y.value_counts().to_dict()}")
        print(f"Class balance: {y.mean():.1%} positive")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Apply SMOTE to training data only
        print("Applying SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        print(f"After SMOTE: {pd.Series(y_train_balanced).value_counts().to_dict()}")
        
        # Scale features
        print("Scaling features...")
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train_balanced),
            columns=X_train.columns
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns
        )
        
        # Feature selection
        print("Selecting top 20 features...")
        self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=20)
        X_train_final = self.feature_selector.fit_transform(X_train_scaled, y_train_balanced)
        X_test_final = self.feature_selector.transform(X_test_scaled)
        
        # Get selected feature names
        selected_features = X_train_scaled.columns[self.feature_selector.get_support()].tolist()
        
        # Convert back to DataFrames
        X_train_final = pd.DataFrame(X_train_final, columns=selected_features)
        X_test_final = pd.DataFrame(X_test_final, columns=selected_features)
        
        print(f"Final shapes - Train: {X_train_final.shape}, Test: {X_test_final.shape}")
        print(f"Selected features: {selected_features[:10]}...")  # Show first 10
        
        return X_train_final, X_test_final, y_train_balanced, y_test
    
    def save_data(self, X_train, X_test, y_train, y_test):
        """Save processed data"""
        print("=== SAVING PROCESSED DATA ===")
        
        X_train.to_csv('data/X_train_unbiased.csv', index=False)
        X_test.to_csv('data/X_test_unbiased.csv', index=False)
        pd.Series(y_train, name='will_repurchase').to_csv('data/y_train_unbiased.csv', index=False)
        pd.Series(y_test, name='will_repurchase').to_csv('data/y_test_unbiased.csv', index=False)
        
        print("✅ Unbiased data saved:")
        print("   📊 X_train_unbiased.csv")
        print("   📊 X_test_unbiased.csv")
        print("   📊 y_train_unbiased.csv") 
        print("   📊 y_test_unbiased.csv")
    
    def run_fast_preprocessing(self):
        """Run complete fast preprocessing"""
        print("🔧 STARTING FAST UNBIASED PREPROCESSING")
        print("=" * 60)
        
        # Load and clean
        df = self.load_and_clean_data()
        
        # Create target (fast method)
        df = self.create_target_fast(df)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Prepare features
        df = self.prepare_final_features(df)
        
        # Split and process
        X_train, X_test, y_train, y_test = self.split_and_process(df)
        
        # Save data
        self.save_data(X_train, X_test, y_train, y_test)
        
        print("\n🎉 FAST PREPROCESSING COMPLETED!")
        print("=" * 60)
        
        return True

if __name__ == "__main__":
    print("Starting FAST unbiased preprocessing...")
    
    try:
        preprocessor = FastUnbiasedPreprocessor()
        success = preprocessor.run_fast_preprocessing()
        
        if success:
            print("\n✅ SUCCESS! Ready for model training")
            print("Expected realistic results: AUC 0.65-0.85")
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()




