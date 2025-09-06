import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveMarketBasketAnalysis:
    def __init__(self):
        self.df = None
        self.frequent_itemsets = None
        self.association_rules_df = None
        self.transaction_matrix = None
        self.analysis_results = {}
        
    def load_and_prepare_data(self):
        """Load and prepare data for market basket analysis"""
        print("=== LOADING DATA FOR MARKET BASKET ANALYSIS ===")
        
        try:
            self.df = pd.read_csv('data/merged_orders.csv')
            self.df['order_purchase_timestamp'] = pd.to_datetime(self.df['order_purchase_timestamp'])
            
            # Clean product category names immediately
            print(f"Original data shape: {self.df.shape}")
            print(f"NaN values in product_category_name: {self.df['product_category_name'].isna().sum()}")
            
            # Remove rows with missing product categories
            self.df = self.df.dropna(subset=['product_category_name']).copy()
            self.df['product_category_name'] = self.df['product_category_name'].astype(str)
            
            print(f"Cleaned data shape: {self.df.shape}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def comprehensive_category_analysis(self):
        """Comprehensive product category analysis"""
        print("\n=== COMPREHENSIVE CATEGORY ANALYSIS ===")
        
        # Basic category statistics
        category_stats = self.df.groupby('product_category_name').agg({
            'order_id': 'nunique',
            'customer_unique_id': 'nunique', 
            'price': ['sum', 'mean', 'median', 'std', 'count'],
            'freight_value': 'mean',
            'product_weight_g': 'mean'
        }).round(2)
        
        # Flatten columns
        category_stats.columns = [
            'unique_orders', 'unique_customers', 'total_revenue', 'avg_price', 
            'median_price', 'price_std', 'total_items', 'avg_freight', 'avg_weight'
        ]
        
        # Calculate additional metrics
        category_stats['revenue_per_customer'] = category_stats['total_revenue'] / category_stats['unique_customers']
        category_stats['items_per_order'] = category_stats['total_items'] / category_stats['unique_orders']
        category_stats['price_volatility'] = category_stats['price_std'] / category_stats['avg_price']
        
        # Market share analysis
        total_revenue = category_stats['total_revenue'].sum()
        total_orders = category_stats['unique_orders'].sum()
        category_stats['revenue_share'] = (category_stats['total_revenue'] / total_revenue * 100).round(2)
        category_stats['order_share'] = (category_stats['unique_orders'] / total_orders * 100).round(2)
        
        # Customer penetration
        total_customers = self.df['customer_unique_id'].nunique()
        category_stats['customer_penetration'] = (category_stats['unique_customers'] / total_customers * 100).round(2)
        
        # Sort by revenue and add rankings
        category_stats = category_stats.sort_values('total_revenue', ascending=False)
        category_stats['revenue_rank'] = range(1, len(category_stats) + 1)
        
        self.analysis_results['category_stats'] = category_stats
        
        print(f"Category analysis completed for {len(category_stats)} categories")
        print("\nTop 5 categories by revenue:")
        for i, (category, data) in enumerate(category_stats.head().iterrows(), 1):
            print(f"{i}. {category}: ${data['total_revenue']:,.0f} ({data['revenue_share']:.1f}% share)")
        
        return category_stats
    
    def seasonal_analysis(self):
        """Analyze seasonal purchasing patterns"""
        print("\n=== SEASONAL ANALYSIS ===")
        
        # Extract time features
        self.df['month'] = self.df['order_purchase_timestamp'].dt.month
        self.df['quarter'] = self.df['order_purchase_timestamp'].dt.quarter
        self.df['day_of_week'] = self.df['order_purchase_timestamp'].dt.day_name()
        self.df['hour'] = self.df['order_purchase_timestamp'].dt.hour
        
        # Monthly analysis
        monthly_stats = self.df.groupby(['month', 'product_category_name']).agg({
            'order_id': 'nunique',
            'price': 'sum'
        }).reset_index()
        
        # Quarterly analysis
        quarterly_revenue = self.df.groupby('quarter')['price'].sum()
        quarterly_orders = self.df.groupby('quarter')['order_id'].nunique()
        
        # Day of week analysis
        dow_analysis = self.df.groupby('day_of_week').agg({
            'order_id': 'nunique',
            'price': ['sum', 'mean'],
            'customer_unique_id': 'nunique'
        }).round(2)
        
        # Hour analysis
        hourly_analysis = self.df.groupby('hour').agg({
            'order_id': 'nunique',
            'price': 'sum'
        })
        
        self.analysis_results['seasonal'] = {
            'monthly_stats': monthly_stats,
            'quarterly_revenue': quarterly_revenue,
            'quarterly_orders': quarterly_orders,
            'dow_analysis': dow_analysis,
            'hourly_analysis': hourly_analysis
        }
        
        print("Seasonal analysis completed")
        return self.analysis_results['seasonal']
    
    def customer_behavior_analysis(self):
        """Analyze customer purchasing behavior patterns"""
        print("\n=== CUSTOMER BEHAVIOR ANALYSIS ===")
        
        # Customer-level analysis
        customer_behavior = self.df.groupby('customer_unique_id').agg({
            'order_id': 'nunique',
            'product_category_name': ['nunique', 'count'],
            'price': ['sum', 'mean', 'std'],
            'order_purchase_timestamp': ['min', 'max', 'count']
        }).round(2)
        
        # Flatten columns
        customer_behavior.columns = [
            'total_orders', 'unique_categories', 'total_items', 
            'total_spent', 'avg_item_price', 'price_std',
            'first_purchase', 'last_purchase', 'item_count'
        ]
        
        # Calculate derived metrics
        customer_behavior['avg_order_value'] = customer_behavior['total_spent'] / customer_behavior['total_orders']
        customer_behavior['category_diversity'] = customer_behavior['unique_categories'] / customer_behavior['total_orders']
        customer_behavior['purchase_frequency'] = customer_behavior['total_orders'] / 365  # Assuming 1 year period
        
        # Customer segmentation - FIXED: Handle duplicate bin edges
        customer_behavior['recency_days'] = (pd.Timestamp.now() - customer_behavior['last_purchase']).dt.days
        
        # Use rank-based scoring instead of qcut for frequency and monetary to avoid duplicate edge issues
        try:
            customer_behavior['frequency_score'] = pd.qcut(
                customer_behavior['total_orders'], 
                5, 
                labels=[1,2,3,4,5], 
                duplicates='drop'  # This handles duplicate edges
            )
        except ValueError:
            # Fallback: use cut instead of qcut for frequency
            max_orders = customer_behavior['total_orders'].max()
            bins = np.linspace(customer_behavior['total_orders'].min(), max_orders, 6)
            customer_behavior['frequency_score'] = pd.cut(
                customer_behavior['total_orders'], 
                bins=bins, 
                labels=[1,2,3,4,5], 
                include_lowest=True
            )
        
        try:
            customer_behavior['monetary_score'] = pd.qcut(
                customer_behavior['total_spent'], 
                5, 
                labels=[1,2,3,4,5], 
                duplicates='drop'  # This handles duplicate edges
            )
        except ValueError:
            # Fallback: use cut instead of qcut for monetary
            max_spent = customer_behavior['total_spent'].max()
            bins = np.linspace(customer_behavior['total_spent'].min(), max_spent, 6)
            customer_behavior['monetary_score'] = pd.cut(
                customer_behavior['total_spent'], 
                bins=bins, 
                labels=[1,2,3,4,5], 
                include_lowest=True
            )
        
        # Cross-category purchasing patterns
        cross_category = self.df.groupby(['customer_unique_id', 'product_category_name']).size().unstack(fill_value=0)
        
        # Only calculate correlation if we have sufficient data
        if cross_category.shape[1] > 1:
            category_correlation = cross_category.corr()
        else:
            category_correlation = pd.DataFrame()
        
        self.analysis_results['customer_behavior'] = {
            'customer_stats': customer_behavior,
            'category_correlation': category_correlation,
            'cross_category_matrix': cross_category
        }
        
        print(f"Customer behavior analysis completed for {len(customer_behavior)} customers")
        return self.analysis_results['customer_behavior']
    
    def market_basket_analysis(self, min_support=0.01, min_threshold=0.1):
        """Perform comprehensive market basket analysis"""
        print("\n=== MARKET BASKET ANALYSIS ===")
        
        # Clean data: remove NaN values and ensure all categories are strings
        df_clean = self.df.dropna(subset=['product_category_name']).copy()
        df_clean['product_category_name'] = df_clean['product_category_name'].astype(str)
        
        # Create transaction data
        transactions = df_clean.groupby('order_id')['product_category_name'].apply(list).tolist()
        
        # Additional cleaning: remove any remaining NaN or invalid entries
        cleaned_transactions = []
        for transaction in transactions:
            cleaned_transaction = [item for item in transaction if pd.notna(item) and str(item) != 'nan']
            if len(cleaned_transaction) > 0:  # Only keep non-empty transactions
                cleaned_transactions.append(cleaned_transaction)
        
        transactions = cleaned_transactions
        
        print(f"Total transactions to analyze: {len(transactions)}")
        
        # Filter out single-item transactions for better association rules
        multi_item_transactions = [t for t in transactions if len(t) > 1]
        print(f"Multi-item transactions: {len(multi_item_transactions)}")
        
        if len(multi_item_transactions) == 0:
            print("No multi-item transactions found!")
            return None
        
        # Use TransactionEncoder for binary matrix
        te = TransactionEncoder()
        te_ary = te.fit(multi_item_transactions).transform(multi_item_transactions)
        transaction_df = pd.DataFrame(te_ary, columns=te.columns_)
        
        self.transaction_matrix = transaction_df
        
        print(f"Created transaction matrix: {transaction_df.shape}")
        
        # Generate frequent itemsets with adaptive support
        print("Generating frequent itemsets...")
        
        # Start with higher support and reduce if needed
        support_levels = [0.05, 0.03, 0.01, 0.005]
        
        for support in support_levels:
            print(f"Trying support level: {support}")
            self.frequent_itemsets = apriori(transaction_df, min_support=support, use_colnames=True)
            
            if len(self.frequent_itemsets) > 0:
                print(f"Found {len(self.frequent_itemsets)} frequent itemsets with support >= {support}")
                break
        
        if len(self.frequent_itemsets) == 0:
            print("No frequent itemsets found with any support threshold")
            return None
        
        # Generate association rules with adaptive threshold
        print("Generating association rules...")
        
        confidence_levels = [0.3, 0.2, 0.1, 0.05]
        
        for confidence in confidence_levels:
            try:
                print(f"Trying confidence level: {confidence}")
                self.association_rules_df = association_rules(
                    self.frequent_itemsets, 
                    metric="confidence", 
                    min_threshold=confidence
                )
                
                if len(self.association_rules_df) > 0:
                    print(f"Generated {len(self.association_rules_df)} association rules with confidence >= {confidence}")
                    break
            except Exception as e:
                print(f"Error generating rules with confidence {confidence}: {e}")
                continue
        
        if self.association_rules_df is None or len(self.association_rules_df) == 0:
            print("No association rules could be generated")
            return None
        
        # Analyze rules
        rules_summary = {
            'total_rules': len(self.association_rules_df),
            'avg_confidence': self.association_rules_df['confidence'].mean(),
            'avg_support': self.association_rules_df['support'].mean(),
            'avg_lift': self.association_rules_df['lift'].mean(),
            'high_confidence_rules': len(self.association_rules_df[self.association_rules_df['confidence'] > 0.7]),
            'high_lift_rules': len(self.association_rules_df[self.association_rules_df['lift'] > 2])
        }
        
        self.analysis_results['market_basket'] = {
            'frequent_itemsets': self.frequent_itemsets,
            'association_rules': self.association_rules_df,
            'rules_summary': rules_summary,
            'transaction_matrix': transaction_df
        }
        
        print(f"Market basket analysis summary:")
        for key, value in rules_summary.items():
            print(f"  {key}: {value}")
        
        return self.analysis_results['market_basket']
    
    def geographic_analysis(self):
        """Analyze purchasing patterns by geography"""
        print("\n=== GEOGRAPHIC ANALYSIS ===")
        
        # State-level analysis
        state_stats = self.df.groupby('customer_state').agg({
            'order_id': 'nunique',
            'customer_unique_id': 'nunique',
            'price': ['sum', 'mean'],
            'product_category_name': 'nunique'
        }).round(2)
        
        state_stats.columns = [
            'total_orders', 'total_customers', 'total_revenue', 
            'avg_order_value', 'category_diversity'
        ]
        
        # Calculate market penetration by state
        state_stats['orders_per_customer'] = state_stats['total_orders'] / state_stats['total_customers']
        state_stats['revenue_per_customer'] = state_stats['total_revenue'] / state_stats['total_customers']
        
        # City-level top performers
        city_stats = self.df.groupby(['customer_state', 'customer_city']).agg({
            'order_id': 'nunique',
            'price': 'sum',
            'customer_unique_id': 'nunique'
        }).round(2)
        
        city_stats.columns = ['orders', 'revenue', 'customers']
        city_stats = city_stats.sort_values('revenue', ascending=False)
        
        self.analysis_results['geographic'] = {
            'state_stats': state_stats,
            'top_cities': city_stats.head(20)
        }
        
        print(f"Geographic analysis completed:")
        print(f"  States analyzed: {len(state_stats)}")
        print(f"  Top state by revenue: {state_stats.sort_values('total_revenue', ascending=False).index[0]}")
        
        return self.analysis_results['geographic']
    
    def price_analysis(self):
        """Comprehensive price and value analysis"""
        print("\n=== PRICE ANALYSIS ===")
        
        # Price distribution analysis
        price_stats = {
            'overall_price_stats': self.df['price'].describe(),
            'price_quartiles': self.df['price'].quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]),
            'price_by_category': self.df.groupby('product_category_name')['price'].describe()
        }
        
        # Price elasticity approximation (frequency vs price relationship)
        try:
            price_bins = pd.qcut(self.df['price'], q=10, labels=False, duplicates='drop')
            self.df['price_bin'] = price_bins
            
            price_elasticity = self.df.groupby('price_bin').agg({
                'order_id': 'count',
                'price': 'mean'
            })
            price_elasticity['demand_index'] = price_elasticity['order_id'] / price_elasticity['order_id'].max()
        except ValueError:
            # Fallback: use regular cut
            max_price = self.df['price'].max()
            min_price = self.df['price'].min()
            bins = np.linspace(min_price, max_price, 11)
            self.df['price_bin'] = pd.cut(self.df['price'], bins=bins, labels=False, include_lowest=True)
            
            price_elasticity = self.df.groupby('price_bin').agg({
                'order_id': 'count',
                'price': 'mean'
            })
            price_elasticity['demand_index'] = price_elasticity['order_id'] / price_elasticity['order_id'].max()
        
        # Category price positioning
        category_price_position = self.df.groupby('product_category_name').agg({
            'price': ['mean', 'median', 'std', 'min', 'max', 'count']
        }).round(2)
        
        category_price_position.columns = [
            'avg_price', 'median_price', 'price_std', 'min_price', 
            'max_price', 'volume'
        ]
        
        # Price competitiveness score
        overall_avg = self.df['price'].mean()
        category_price_position['competitiveness_score'] = (
            overall_avg / category_price_position['avg_price']
        ).round(2)
        
        self.analysis_results['price_analysis'] = {
            'price_stats': price_stats,
            'price_elasticity': price_elasticity,
            'category_pricing': category_price_position
        }
        
        print("Price analysis completed")
        return self.analysis_results['price_analysis']
    
    def cohort_analysis(self):
        """Customer cohort analysis for retention insights"""
        print("\n=== COHORT ANALYSIS ===")
        
        try:
            # Prepare data for cohort analysis
            self.df['order_period'] = self.df['order_purchase_timestamp'].dt.to_period('M')
            
            # Get customer's first purchase month
            customer_cohorts = self.df.groupby('customer_unique_id')['order_purchase_timestamp'].min().reset_index()
            customer_cohorts.columns = ['customer_unique_id', 'cohort_month']
            customer_cohorts['cohort_month'] = customer_cohorts['cohort_month'].dt.to_period('M')
            
            # Merge back to main dataframe
            df_cohort = self.df.merge(customer_cohorts, on='customer_unique_id')
            
            # Calculate period number (months since first purchase)
            df_cohort['period_number'] = (df_cohort['order_period'] - df_cohort['cohort_month']).apply(lambda x: x.n)
            
            # Create cohort table
            cohort_data = df_cohort.groupby(['cohort_month', 'period_number'])['customer_unique_id'].nunique().reset_index()
            cohort_table = cohort_data.pivot(index='cohort_month', columns='period_number', values='customer_unique_id')
            
            # Calculate retention rates
            cohort_sizes = df_cohort.groupby('cohort_month')['customer_unique_id'].nunique()
            retention_table = cohort_table.divide(cohort_sizes, axis=0)
            
            self.analysis_results['cohort_analysis'] = {
                'cohort_table': cohort_table,
                'retention_table': retention_table,
                'cohort_sizes': cohort_sizes
            }
            
            print("Cohort analysis completed")
            return self.analysis_results['cohort_analysis']
            
        except Exception as e:
            print(f"Error in cohort analysis: {e}")
            return None
    
    def advanced_basket_patterns(self):
        """Advanced market basket pattern analysis"""
        print("\n=== ADVANCED BASKET PATTERNS ===")
        
        if self.association_rules_df is None or len(self.association_rules_df) == 0:
            print("No association rules available. Running market basket analysis first...")
            self.market_basket_analysis()
        
        # Analyze basket sizes - use cleaned data
        df_clean = self.df.dropna(subset=['product_category_name']).copy()
        basket_sizes = df_clean.groupby('order_id')['product_category_name'].count()
        basket_analysis = {
            'avg_basket_size': basket_sizes.mean(),
            'median_basket_size': basket_sizes.median(),
            'max_basket_size': basket_sizes.max(),
            'single_item_orders': (basket_sizes == 1).sum(),
            'multi_item_orders': (basket_sizes > 1).sum(),
            'large_baskets': (basket_sizes >= 5).sum()
        }
        
        # Top product combinations (if rules exist)
        if self.association_rules_df is not None and len(self.association_rules_df) > 0:
            # Strong rules (high confidence and lift)
            strong_rules = self.association_rules_df[
                (self.association_rules_df['confidence'] > 0.5) & 
                (self.association_rules_df['lift'] > 1.5)
            ].sort_values(['lift', 'confidence'], ascending=False)
            
            # Complementary products (high lift)
            complementary = self.association_rules_df.nlargest(10, 'lift')[
                ['antecedents', 'consequents', 'support', 'confidence', 'lift']
            ]
            
            # Most frequent combinations
            frequent_combos = self.association_rules_df.nlargest(10, 'support')[
                ['antecedents', 'consequents', 'support', 'confidence', 'lift']
            ]
        else:
            strong_rules = pd.DataFrame()
            complementary = pd.DataFrame()
            frequent_combos = pd.DataFrame()
        
        self.analysis_results['advanced_patterns'] = {
            'basket_analysis': basket_analysis,
            'strong_rules': strong_rules,
            'complementary_products': complementary,
            'frequent_combinations': frequent_combos
        }
        
        print(f"Advanced pattern analysis completed:")
        print(f"  Average basket size: {basket_analysis['avg_basket_size']:.2f}")
        print(f"  Multi-item orders: {basket_analysis['multi_item_orders']:,}")
        if self.association_rules_df is not None:
            print(f"  Strong association rules: {len(strong_rules)}")
        
        return self.analysis_results['advanced_patterns']
    
    def payment_behavior_analysis(self):
        """Analyze payment method preferences and patterns"""
        print("\n=== PAYMENT BEHAVIOR ANALYSIS ===")
        
        # Check if payment columns exist
        payment_columns = ['payment_type', 'payment_installments']
        missing_columns = [col for col in payment_columns if col not in self.df.columns]
        
        if missing_columns:
            print(f"Warning: Missing payment columns: {missing_columns}")
            print("Skipping payment behavior analysis")
            return None
        
        # Payment method analysis
        payment_stats = self.df.groupby('payment_type').agg({
            'order_id': 'nunique',
            'customer_unique_id': 'nunique',
            'price': ['sum', 'mean'],
            'payment_installments': 'mean'
        }).round(2)
        
        payment_stats.columns = [
            'total_orders', 'unique_customers', 'total_revenue', 
            'avg_order_value', 'avg_installments'
        ]
        
        # Installment analysis
        installment_analysis = self.df.groupby('payment_installments').agg({
            'order_id': 'count',
            'price': ['mean', 'sum'],
            'payment_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
        }).round(2)
        
        # Category preferences by payment method
        payment_category = pd.crosstab(
            self.df['payment_type'], 
            self.df['product_category_name'], 
            normalize='index'
        ) * 100
        
        self.analysis_results['payment_behavior'] = {
            'payment_stats': payment_stats,
            'installment_analysis': installment_analysis,
            'payment_category_preferences': payment_category
        }
        
        print("Payment behavior analysis completed")
        return self.analysis_results['payment_behavior']
    
    def generate_business_insights(self):
        """Generate actionable business insights"""
        print("\n=== GENERATING BUSINESS INSIGHTS ===")
        
        insights = {
            'category_insights': [],
            'customer_insights': [],
            'pricing_insights': [],
            'basket_insights': [],
            'geographic_insights': []
        }
        
        # Category insights
        if 'category_stats' in self.analysis_results:
            category_stats = self.analysis_results['category_stats']
            top_revenue_cat = category_stats.index[0]
            top_penetration_cat = category_stats.sort_values('customer_penetration', ascending=False).index[0]
            
            insights['category_insights'] = [
                f"Top revenue generator: {top_revenue_cat} (${category_stats.loc[top_revenue_cat, 'total_revenue']:,.0f})",
                f"Highest customer penetration: {top_penetration_cat} ({category_stats.loc[top_penetration_cat, 'customer_penetration']:.1f}%)",
                f"Most volatile pricing: {category_stats.sort_values('price_volatility', ascending=False).index[0]}"
            ]
        
        # Customer insights
        if 'customer_behavior' in self.analysis_results:
            customer_stats = self.analysis_results['customer_behavior']['customer_stats']
            repeat_rate = (len(customer_stats[customer_stats['total_orders'] > 1]) / len(customer_stats)) * 100
            avg_ltv = customer_stats['total_spent'].mean()
            
            insights['customer_insights'] = [
                f"Customer repeat rate: {repeat_rate:.1f}%",
                f"Average customer lifetime value: ${avg_ltv:.2f}",
                f"High-value customers (>$200): {len(customer_stats[customer_stats['total_spent'] > 200]):,}"
            ]
        
        # Basket insights
        if 'advanced_patterns' in self.analysis_results:
            basket_stats = self.analysis_results['advanced_patterns']['basket_analysis']
            
            insights['basket_insights'] = [
                f"Average basket size: {basket_stats['avg_basket_size']:.2f} items",
                f"Cross-selling opportunity: {basket_stats['single_item_orders']:,} single-item orders",
                f"Large basket potential: {basket_stats['large_baskets']:,} orders with 5+ items"
            ]
        
        # Geographic insights
        if 'geographic' in self.analysis_results:
            state_stats = self.analysis_results['geographic']['state_stats']
            top_state = state_stats.sort_values('total_revenue', ascending=False).index[0]
            
            insights['geographic_insights'] = [
                f"Top performing state: {top_state}",
                f"Geographic concentration: Top 5 states account for {state_stats.sort_values('total_revenue', ascending=False).head()['total_revenue'].sum() / state_stats['total_revenue'].sum() * 100:.1f}% of revenue"
            ]
        
        self.analysis_results['business_insights'] = insights
        return insights
    
    def save_analysis_results(self):
        """Save all analysis results"""
        print("\n=== SAVING ANALYSIS RESULTS ===")
        
        import os
        os.makedirs('analysis_results', exist_ok=True)
        
        try:
            # Save individual analysis components
            if 'category_stats' in self.analysis_results:
                self.analysis_results['category_stats'].to_csv('analysis_results/category_analysis.csv')
                
            if 'customer_behavior' in self.analysis_results:
                self.analysis_results['customer_behavior']['customer_stats'].to_csv('analysis_results/customer_behavior.csv')
                if not self.analysis_results['customer_behavior']['category_correlation'].empty:
                    self.analysis_results['customer_behavior']['category_correlation'].to_csv('analysis_results/category_correlation.csv')
            
            if 'market_basket' in self.analysis_results and self.association_rules_df is not None:
                self.association_rules_df.to_csv('analysis_results/association_rules.csv', index=False)
                self.frequent_itemsets.to_csv('analysis_results/frequent_itemsets.csv', index=False)
            
            if 'geographic' in self.analysis_results:
                self.analysis_results['geographic']['state_stats'].to_csv('analysis_results/geographic_analysis.csv')
            
            # Save summary insights
            if 'business_insights' in self.analysis_results:
                import json
                with open('analysis_results/business_insights.json', 'w') as f:
                    json.dump(self.analysis_results['business_insights'], f, indent=2)
            
            print("Analysis results saved to 'analysis_results/' directory")
            
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def print_summary_report(self):
        """Print a comprehensive summary report"""
        print("\n" + "="*70)
        print("📊 COMPREHENSIVE MARKET BASKET ANALYSIS REPORT")
        print("="*70)
        
        if 'category_stats' in self.analysis_results:
            print(f"\n🏷️  CATEGORY PERFORMANCE:")
            category_stats = self.analysis_results['category_stats']
            print(f"   • Total categories: {len(category_stats)}")
            print(f"   • Top performer: {category_stats.index[0]} (${category_stats.iloc[0]['total_revenue']:,.0f})")
            print(f"   • Market concentration: Top 5 categories = {category_stats.head()['revenue_share'].sum():.1f}% of revenue")
        
        if 'customer_behavior' in self.analysis_results:
            print(f"\n👥 CUSTOMER BEHAVIOR:")
            customer_stats = self.analysis_results['customer_behavior']['customer_stats']
            print(f"   • Total customers: {len(customer_stats):,}")
            print(f"   • Average orders per customer: {customer_stats['total_orders'].mean():.2f}")
            print(f"   • Average spend per customer: ${customer_stats['total_spent'].mean():.2f}")
        
        if 'market_basket' in self.analysis_results:
            print(f"\n🛒 MARKET BASKET INSIGHTS:")
            basket_results = self.analysis_results['market_basket']
            if 'rules_summary' in basket_results:
                rules_summary = basket_results['rules_summary']
                print(f"   • Association rules found: {rules_summary['total_rules']}")
                print(f"   • Average confidence: {rules_summary['avg_confidence']:.2f}")
                print(f"   • Average lift: {rules_summary['avg_lift']:.2f}")
        
        if 'advanced_patterns' in self.analysis_results:
            print(f"\n📈 BASKET PATTERNS:")
            basket_analysis = self.analysis_results['advanced_patterns']['basket_analysis']
            print(f"   • Average basket size: {basket_analysis['avg_basket_size']:.2f} items")
            print(f"   • Single-item orders: {basket_analysis['single_item_orders']:,} ({basket_analysis['single_item_orders']/(basket_analysis['single_item_orders']+basket_analysis['multi_item_orders'])*100:.1f}%)")
            print(f"   • Multi-item orders: {basket_analysis['multi_item_orders']:,}")
        
        if 'geographic' in self.analysis_results:
            print(f"\n🌎 GEOGRAPHIC INSIGHTS:")
            state_stats = self.analysis_results['geographic']['state_stats']
            top_state = state_stats.sort_values('total_revenue', ascending=False).index[0]
            print(f"   • Top state: {top_state} (${state_stats.loc[top_state, 'total_revenue']:,.0f})")
            print(f"   • States analyzed: {len(state_stats)}")
        
        print(f"\n✅ Analysis complete! Check 'analysis_results/' directory for detailed reports.")
        print("="*70)
    
    def run_comprehensive_analysis(self):
        """Run complete market basket and business analysis"""
        print("🚀 STARTING COMPREHENSIVE MARKET BASKET ANALYSIS")
        print("=" * 70)
        
        # Load data
        if not self.load_and_prepare_data():
            return False
        
        try:
            # Run all analyses
            self.comprehensive_category_analysis()
            self.seasonal_analysis()
            self.customer_behavior_analysis()
            self.market_basket_analysis()
            self.geographic_analysis()
            self.price_analysis()
            self.payment_behavior_analysis()
            self.cohort_analysis()
            
            # Advanced pattern analysis
            self.advanced_basket_patterns()
            
            # Generate insights
            self.generate_business_insights()
            
            # Save results
            self.save_analysis_results()
            
            # Print summary report
            self.print_summary_report()
            
            return True
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    try:
        analyzer = ComprehensiveMarketBasketAnalysis()
        success = analyzer.run_comprehensive_analysis()
        
        if success:
            print("\n✅ COMPREHENSIVE MARKET BASKET ANALYSIS SUCCESSFUL!")
            print("🎯 Ready for dashboard integration!")
        else:
            print("\n❌ ANALYSIS FAILED!")
            
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()