import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

class SimpleNextPurchasePrep:
    def __init__(self):
        self.customer_profiles = None
        self.product_profiles = None
        
    def load_and_process_data(self):
        """Load and process data in one go"""
        print("🚀 STARTING SIMPLE NEXT PURCHASE DATA PREPARATION")
        print("=" * 60)
        
        # Load data
        print("=== LOADING RAW DATA ===")
        df = pd.read_csv('data/merged_orders.csv')
        df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
        
        print(f"✅ Raw data loaded: {df.shape}")
        print(f"✅ Unique customers: {df['customer_unique_id'].nunique():,}")
        print(f"✅ Unique product categories: {df['product_category_name'].nunique():,}")
        
        # Create customer profiles
        print("\n=== CREATING CUSTOMER PROFILES ===")
        customer_profiles = df.groupby('customer_unique_id').agg({
            'order_id': 'nunique',
            'product_category_name': 'count',
            'price': ['sum', 'mean'],
            'order_purchase_timestamp': ['min', 'max']
        }).reset_index()
        
        # Flatten column names
        customer_profiles.columns = [
            'customer_id', 'total_orders', 'total_items', 
            'total_spent', 'avg_price', 'first_purchase', 'last_purchase'
        ]
        
        # Add derived metrics
        customer_profiles['avg_order_value'] = customer_profiles['total_spent'] / customer_profiles['total_orders']
        customer_profiles['days_active'] = (customer_profiles['last_purchase'] - customer_profiles['first_purchase']).dt.days
        
        print(f"✅ Customer profiles created: {len(customer_profiles):,}")
        
        # Create product profiles
        print("\n=== CREATING PRODUCT PROFILES ===")
        product_profiles = df.groupby('product_category_name').agg({
            'customer_unique_id': ['count', 'nunique'],
            'price': ['sum', 'mean']
        }).reset_index()
        
        # Flatten columns
        product_profiles.columns = [
            'product_category', 'total_sales', 'unique_customers', 
            'total_revenue', 'avg_price'
        ]
        
        product_profiles = product_profiles.sort_values('total_sales', ascending=False)
        product_profiles['popularity_rank'] = range(1, len(product_profiles) + 1)
        
        print(f"✅ Product profiles created: {len(product_profiles):,}")
        print("✅ Top 5 categories:")
        for _, row in product_profiles.head().iterrows():
            print(f"   {row['product_category']:30} - {row['total_sales']:,} sales")
        
        # Create simple product transitions
        print("\n=== CREATING PRODUCT TRANSITIONS ===")
        
        # Get customers with multiple purchases
        multi_purchase_customers = customer_profiles[customer_profiles['total_orders'] > 1]['customer_id'].tolist()
        multi_df = df[df['customer_unique_id'].isin(multi_purchase_customers)]
        
        # Sort by customer and time
        multi_df_sorted = multi_df.sort_values(['customer_unique_id', 'order_purchase_timestamp'])
        
        # Create transitions by comparing consecutive purchases
        transitions = []
        for customer_id in multi_purchase_customers:
            customer_data = multi_df_sorted[multi_df_sorted['customer_unique_id'] == customer_id]
            categories = customer_data['product_category_name'].tolist()
            
            # Create sequential pairs
            for i in range(len(categories) - 1):
                transitions.append({
                    'from_category': categories[i],
                    'to_category': categories[i + 1]
                })
        
        # Count transitions
        if transitions:
            transitions_df = pd.DataFrame(transitions)
            transition_counts = transitions_df.groupby(['from_category', 'to_category']).size().reset_index(name='count')
            transition_counts = transition_counts.sort_values('count', ascending=False)
        else:
            transition_counts = pd.DataFrame(columns=['from_category', 'to_category', 'count'])
        
        print(f"✅ Product transitions created: {len(transition_counts):,}")
        
        # Create interaction matrix
        print("\n=== CREATING INTERACTION MATRIX ===")
        interaction_matrix = df.groupby(['customer_unique_id', 'product_category_name']).size().reset_index(name='interactions')
        interaction_pivot = interaction_matrix.pivot(
            index='customer_unique_id',
            columns='product_category_name',
            values='interactions'
        ).fillna(0)
        
        print(f"✅ Interaction matrix: {interaction_pivot.shape}")
        
        # Save everything
        print("\n=== SAVING DATASETS ===")
        customer_profiles.to_csv('data/customer_profiles.csv', index=False)
        product_profiles.to_csv('data/product_profiles.csv', index=False)
        transition_counts.to_csv('data/product_transitions.csv', index=False)
        interaction_pivot.to_csv('data/customer_product_matrix.csv')
        
        # Create sample customer sequences (JSON)
        sample_customers = customer_profiles.head(100)['customer_id'].tolist()
        sequences = {}
        
        for customer_id in sample_customers:
            customer_data = df[df['customer_unique_id'] == customer_id].sort_values('order_purchase_timestamp')
            sequence = []
            
            for _, row in customer_data.iterrows():
                sequence.append({
                    'timestamp': row['order_purchase_timestamp'].isoformat(),
                    'product_category': row['product_category_name'],
                    'price': float(row['price']) if pd.notna(row['price']) else 0.0
                })
            
            sequences[str(customer_id)] = sequence
        
        with open('data/customer_purchase_sequences.json', 'w') as f:
            json.dump(sequences, f, indent=2)
        
        print("✅ All datasets saved:")
        print("   📊 customer_profiles.csv")
        print("   📊 product_profiles.csv")
        print("   📊 product_transitions.csv")
        print("   📊 customer_product_matrix.csv")
        print("   📊 customer_purchase_sequences.json")
        
        print(f"\n🎉 SIMPLE DATA PREPARATION COMPLETED!")
        print("=" * 60)
        print("📊 SUMMARY:")
        print(f"   Customers: {len(customer_profiles):,}")
        print(f"   Product categories: {len(product_profiles):,}")
        print(f"   Product transitions: {len(transition_counts):,}")
        print(f"   Multi-purchase customers: {len(multi_purchase_customers):,}")
        print("\n✅ Ready for recommendation engine!")
        
        return True

if __name__ == "__main__":
    try:
        prep = SimpleNextPurchasePrep()
        success = prep.load_and_process_data()
        
        if success:
            print("\n🚀 SUCCESS! Ready to build recommendation engines")
            
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()


