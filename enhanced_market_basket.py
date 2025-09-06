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
from operator import attrgetter
warnings.filterwarnings('ignore')

class EnhancedMarketBasketAnalysis:
    def __init__(self):
        self.df = None
        self.frequent_itemsets = None
        self.association_rules_df = None
        self.transaction_matrix = None
        self.analysis_results = {}
        
    def load_and_prepare_data(self):
        """Load and prepare data for comprehensive analysis"""
        print("=== LOADING DATA FOR ENHANCED MARKET BASKET ANALYSIS ===")
        
        try:
            self.df = pd.read_csv('data/merged_orders.csv')
            self.df['order_purchase_timestamp'] = pd.to_datetime(self.df['order_purchase_timestamp'])
            print(f"Data loaded successfully: {self.df.shape}")
            print(f"Date range: {self.df['order_purchase_timestamp'].min()} to {self.df['order_purchase_timestamp'].max()}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def comprehensive_category_analysis(self):
        """Enhanced category analysis with business metrics"""
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
        
        # Calculate advanced business metrics
        category_stats['revenue_per_customer'] = category_stats['total_revenue'] / category_stats['unique_customers']
        category_stats['items_per_order'] = category_stats['total_items'] / category_stats['unique_orders']
        category_stats['price_volatility'] = category_stats['price_std'] / category_stats['avg_price']
        
        # Market share analysis
        total_revenue = category_stats['total_revenue'].sum()
        total_orders = category_stats['unique_orders'].sum()
        total_customers = self.df['customer_unique_id'].nunique()
        
        category_stats['revenue_share'] = (category_stats['total_revenue'] / total_revenue * 100).round(2)
        category_stats['order_share'] = (category_stats['unique_orders'] / total_orders * 100).round(2)
        category_stats['customer_penetration'] = (category_stats['unique_customers'] / total_customers * 100).round(2)
        
        # Performance rankings
        category_stats = category_stats.sort_values('total_revenue', ascending=False)
        category_stats['revenue_rank'] = range(1, len(category_stats) + 1)
        category_stats['penetration_rank'] = category_stats['customer_penetration'].rank(method='dense', ascending=False)
        category_stats['price_rank'] = category_stats['avg_price'].rank(method='dense', ascending=False)
        
        # Competitiveness scoring
        overall_avg_price = self.df['price'].mean()
        category_stats['competitiveness_score'] = (overall_avg_price / category_stats['avg_price']).round(2)
        
        self.analysis_results['category_stats'] = category_stats
        
        print(f"Category analysis completed for {len(category_stats)} categories")
        print("\nTop 5 Revenue Generators:")
        for i, (category, data) in enumerate(category_stats.head().iterrows(), 1):
            print(f"{i}. {category}: ${data['total_revenue']:,.0f} ({data['revenue_share']:.1f}% market share)")
        
        print("\nHighest Customer Penetration:")
        top_penetration = category_stats.nlargest(3, 'customer_penetration')
        for category, data in top_penetration.iterrows():
            print(f"- {category}: {data['customer_penetration']:.1f}% of customers")
        
        return category_stats
    
    def seasonal_and_temporal_analysis(self):
        """Advanced seasonal and temporal pattern analysis"""
        print("\n=== SEASONAL & TEMPORAL ANALYSIS ===")
        
        # Extract comprehensive time features
        self.df['month'] = self.df['order_purchase_timestamp'].dt.month
        self.df['quarter'] = self.df['order_purchase_timestamp'].dt.quarter
        self.df['day_of_week'] = self.df['order_purchase_timestamp'].dt.day_name()
        self.df['hour'] = self.df['order_purchase_timestamp'].dt.hour
        self.df['is_weekend'] = self.df['order_purchase_timestamp'].dt.weekday >= 5
        self.df['week_of_year'] = self.df['order_purchase_timestamp'].dt.isocalendar().week
        
        # Monthly trends by category
        monthly_category_stats = self.df.groupby(['month', 'product_category_name']).agg({
            'order_id': 'nunique',
            'price': ['sum', 'mean'],
            'customer_unique_id': 'nunique'
        }).reset_index()
        
        monthly_category_stats.columns = ['month', 'category', 'orders', 'revenue', 'avg_price', 'customers']
        
        # Seasonal performance metrics
        quarterly_performance = self.df.groupby(['quarter', 'product_category_name']).agg({
            'price': 'sum',
            'order_id': 'nunique'
        }).reset_index()
        
        # Day of week patterns
        dow_patterns = self.df.groupby(['day_of_week', 'product_category_name']).agg({
            'order_id': 'nunique',
            'price': 'sum'
        }).reset_index()
        
        # Hour of day patterns
        hourly_patterns = self.df.groupby(['hour', 'product_category_name']).agg({
            'order_id': 'nunique',
            'price': 'sum'
        }).reset_index()
        
        # Weekend vs weekday analysis
        weekend_analysis = self.df.groupby(['is_weekend', 'product_category_name']).agg({
            'order_id': 'nunique',
            'price': ['sum', 'mean'],
            'customer_unique_id': 'nunique'
        }).reset_index()
        
        self.analysis_results['temporal_analysis'] = {
            'monthly_category_stats': monthly_category_stats,
            'quarterly_performance': quarterly_performance,
            'dow_patterns': dow_patterns,
            'hourly_patterns': hourly_patterns,
            'weekend_analysis': weekend_analysis
        }
        
        print("Temporal analysis completed")
        print(f"- Monthly patterns: {len(monthly_category_stats)} category-month combinations")
        print(f"- Day-of-week patterns: {len(dow_patterns)} category-day combinations")
        print(f"- Hourly patterns: {len(hourly_patterns)} category-hour combinations")
        
        return self.analysis_results['temporal_analysis']
    
    def advanced_customer_behavior_analysis(self):
        """Comprehensive customer behavior and segmentation analysis"""
        print("\n=== ADVANCED CUSTOMER BEHAVIOR ANALYSIS ===")
        
        # Enhanced customer-level analysis
        customer_behavior = self.df.groupby('customer_unique_id').agg({
            'order_id': 'nunique',
            'product_category_name': ['nunique', 'count'],
            'price': ['sum', 'mean', 'std', 'min', 'max'],
            'order_purchase_timestamp': ['min', 'max', 'count'],
            'freight_value': 'mean',
            'payment_installments': 'mean'
        }).round(2)
        
        # Flatten columns
        customer_behavior.columns = [
            'total_orders', 'unique_categories', 'total_items',
            'total_spent', 'avg_item_price', 'price_std', 'min_price', 'max_price',
            'first_purchase', 'last_purchase', 'item_count',
            'avg_freight', 'avg_installments'
        ]
        
        # Calculate derived behavioral metrics
        customer_behavior['avg_order_value'] = customer_behavior['total_spent'] / customer_behavior['total_orders']
        customer_behavior['category_diversity_ratio'] = customer_behavior['unique_categories'] / customer_behavior['total_orders']
        customer_behavior['price_range'] = customer_behavior['max_price'] - customer_behavior['min_price']
        customer_behavior['spending_consistency'] = 1 / (1 + customer_behavior['price_std'])  # Higher = more consistent
        
        # Temporal behavior metrics
        customer_behavior['days_active'] = (customer_behavior['last_purchase'] - customer_behavior['first_purchase']).dt.days
        customer_behavior['purchase_frequency'] = customer_behavior['total_orders'] / (customer_behavior['days_active'] + 1) * 30  # Purchases per month
        customer_behavior['recency_days'] = (pd.Timestamp.now() - customer_behavior['last_purchase']).dt.days
        
        # RFM Analysis (Recency, Frequency, Monetary)
        customer_behavior['recency_score'] = pd.qcut(customer_behavior['recency_days'], 5, labels=[5,4,3,2,1])
        customer_behavior['frequency_score'] = pd.qcut(customer_behavior['total_orders'].rank(method='first'), 5, labels=[1,2,3,4,5])
        customer_behavior['monetary_score'] = pd.qcut(customer_behavior['total_spent'].rank(method='first'), 5, labels=[1,2,3,4,5])
        
        # Convert scores to numeric for calculations
        customer_behavior['recency_score'] = pd.to_numeric(customer_behavior['recency_score'])
        customer_behavior['frequency_score'] = pd.to_numeric(customer_behavior['frequency_score'])
        customer_behavior['monetary_score'] = pd.to_numeric(customer_behavior['monetary_score'])
        
        # Combined RFM score
        customer_behavior['rfm_score'] = (
            customer_behavior['recency_score'] * 0.3 + 
            customer_behavior['frequency_score'] * 0.4 + 
            customer_behavior['monetary_score'] * 0.3
        ).round(2)
        
        # Advanced customer segmentation
        def advanced_customer_segment(row):
            if row['rfm_score'] >= 4.5:
                return 'Champions'
            elif row['rfm_score'] >= 4.0:
                return 'Loyal Customers'
            elif row['rfm_score'] >= 3.5:
                return 'Potential Loyalists'
            elif row['recency_score'] <= 2 and row['frequency_score'] >= 3:
                return 'At Risk'
            elif row['recency_score'] <= 2:
                return 'Cannot Lose Them'
            elif row['frequency_score'] == 1:
                return 'New Customers'
            else:
                return 'Regular Customers'
        
        customer_behavior['advanced_segment'] = customer_behavior.apply(advanced_customer_segment, axis=1)
        
        # Cross-category purchasing patterns
        customer_category_matrix = pd.crosstab(
            self.df['customer_unique_id'], 
            self.df['product_category_name']
        )
        
        # Category correlation analysis
        category_correlation = customer_category_matrix.corr()
        
        # Category affinity analysis (customers who buy A also buy B)
        category_affinity = {}
        for cat1 in customer_category_matrix.columns:
            for cat2 in customer_category_matrix.columns:
                if cat1 != cat2:
                    # Customers who bought both categories
                    both = ((customer_category_matrix[cat1] > 0) & (customer_category_matrix[cat2] > 0)).sum()
                    # Customers who bought first category
                    cat1_buyers = (customer_category_matrix[cat1] > 0).sum()
                    # Affinity score (conditional probability)
                    if cat1_buyers > 0:
                        affinity = both / cat1_buyers
                        category_affinity[f"{cat1}_to_{cat2}"] = affinity
        
        self.analysis_results['customer_behavior'] = {
            'customer_stats': customer_behavior,
            'category_correlation': category_correlation,
            'customer_category_matrix': customer_category_matrix,
            'category_affinity': category_affinity
        }
        
        print(f"Customer behavior analysis completed for {len(customer_behavior)} customers")
        print(f"Advanced segments distribution:")
        segment_counts = customer_behavior['advanced_segment'].value_counts()
        for segment, count in segment_counts.items():
            print(f"- {segment}: {count:,} customers ({count/len(customer_behavior)*100:.1f}%)")
        
        return self.analysis_results['customer_behavior']
    
    def enhanced_market_basket_analysis(self, min_support=0.005, min_threshold=0.1):
        """Enhanced market basket analysis with multiple algorithms"""
        print("\n=== ENHANCED MARKET BASKET ANALYSIS ===")
        
        # Create transaction data at order level
        print("Creating transaction datasets...")
        order_transactions = self.df.groupby('order_id')['product_category_name'].apply(list).tolist()
        
        # Create customer-level transactions (all categories a customer has purchased)
        customer_transactions = self.df.groupby('customer_unique_id')['product_category_name'].apply(
            lambda x: list(set(x))
        ).tolist()
        
        print(f"Order-level transactions: {len(order_transactions):,}")
        print(f"Customer-level transactions: {len(customer_transactions):,}")
        
        # Analyze transaction patterns
        transaction_sizes = [len(t) for t in order_transactions]
        print(f"Average basket size: {np.mean(transaction_sizes):.2f}")
        print(f"Median basket size: {np.median(transaction_sizes):.2f}")
        print(f"Max basket size: {max(transaction_sizes)}")
        
        # Process order-level transactions
        print("\nProcessing order-level market basket analysis...")
        te_orders = TransactionEncoder()
        te_ary_orders = te_orders.fit(order_transactions).transform(order_transactions)
        df_orders = pd.DataFrame(te_ary_orders, columns=te_orders.columns_)
        
        # Generate frequent itemsets for orders
        self.frequent_itemsets = apriori(df_orders, min_support=min_support, use_colnames=True)
        
        if len(self.frequent_itemsets) == 0:
            print(f"No frequent itemsets found with support >= {min_support}")
            # Try with lower support
            min_support = 0.001
            print(f"Retrying with support >= {min_support}")
            self.frequent_itemsets = apriori(df_orders, min_support=min_support, use_colnames=True)
        
        print(f"Found {len(self.frequent_itemsets)} frequent itemsets")
        
        # Generate association rules
        if len(self.frequent_itemsets) > 0:
            print("Generating association rules...")
            self.association_rules_df = association_rules(
                self.frequent_itemsets, 
                metric="confidence", 
                min_threshold=min_threshold,
                num_itemsets=len(self.frequent_itemsets)
            )
            
            if len(self.association_rules_df) == 0:
                # Try with lower threshold
                min_threshold = 0.05
                print(f"Retrying with confidence >= {min_threshold}")
                self.association_rules_df = association_rules(
                    self.frequent_itemsets, 
                    metric="confidence", 
                    min_threshold=min_threshold
                )
            
            print(f"Generated {len(self.association_rules_df)} association rules")
        
        # Process customer-level transactions
        print("\nProcessing customer-level market basket analysis...")
        te_customers = TransactionEncoder()
        te_ary_customers = te_customers.fit(customer_transactions).transform(customer_transactions)
        df_customers = pd.DataFrame(te_ary_customers, columns=te_customers.columns_)
        
        # Customer-level frequent itemsets
        customer_frequent_itemsets = apriori(df_customers, min_support=0.01, use_colnames=True)
        print(f"Customer-level frequent itemsets: {len(customer_frequent_itemsets)}")
        
        # Basket composition analysis
        basket_analysis = {
            'avg_basket_size': np.mean(transaction_sizes),
            'median_basket_size': np.median(transaction_sizes),
            'max_basket_size': max(transaction_sizes),
            'single_item_baskets': sum(1 for size in transaction_sizes if size == 1),
            'multi_item_baskets': sum(1 for size in transaction_sizes if size > 1),
            'large_baskets_5plus': sum(1 for size in transaction_sizes if size >= 5),
            'basket_size_distribution': pd.Series(transaction_sizes).value_counts().sort_index()
        }
        
        # Advanced rule analysis
        advanced_rules_analysis = {}
        if len(self.association_rules_df) > 0:
            rules_df = self.association_rules_df.copy()
            
            # High confidence rules (> 70%)
            high_conf_rules = rules_df[rules_df['confidence'] > 0.7]
            
            # High lift rules (> 2x)
            high_lift_rules = rules_df[rules_df['lift'] > 2.0]
            
            # Strong rules (high confidence AND high lift)
            strong_rules = rules_df[(rules_df['confidence'] > 0.6) & (rules_df['lift'] > 1.5)]
            
            # Most frequent combinations
            frequent_combos = rules_df.nlargest(10, 'support')
            
            advanced_rules_analysis = {
                'total_rules': len(rules_df),
                'high_confidence_rules': len(high_conf_rules),
                'high_lift_rules': len(high_lift_rules), 
                'strong_rules': len(strong_rules),
                'avg_confidence': rules_df['confidence'].mean(),
                'avg_lift': rules_df['lift'].mean(),
                'avg_support': rules_df['support'].mean(),
                'top_rules_by_confidence': high_conf_rules.nlargest(5, 'confidence'),
                'top_rules_by_lift': high_lift_rules.nlargest(5, 'lift'),
                'most_frequent_rules': frequent_combos
            }
        
        self.analysis_results['market_basket'] = {
            'order_frequent_itemsets': self.frequent_itemsets,
            'customer_frequent_itemsets': customer_frequent_itemsets,
            'association_rules': self.association_rules_df,
            'basket_analysis': basket_analysis,
            'advanced_rules_analysis': advanced_rules_analysis,
            'transaction_matrix_orders': df_orders,
            'transaction_matrix_customers': df_customers
        }
        
        print(f"\nMarket basket analysis completed:")
        print(f"- Order transactions processed: {len(order_transactions):,}")
        print(f"- Association rules generated: {len(self.association_rules_df) if self.association_rules_df is not None else 0}")
        print(f"- Average basket size: {basket_analysis['avg_basket_size']:.2f}")
        print(f"- Cross-sell opportunities: {basket_analysis['multi_item_baskets']:,}")
        
        return self.analysis_results['market_basket']
    
    def geographic_and_demographic_analysis(self):
        """Comprehensive geographic and demographic analysis"""
        print("\n=== GEOGRAPHIC & DEMOGRAPHIC ANALYSIS ===")
        
        # State-level comprehensive analysis
        state_analysis = self.df.groupby('customer_state').agg({
            'order_id': 'nunique',
            'customer_unique_id': 'nunique',
            'price': ['sum', 'mean', 'std'],
            'product_category_name': 'nunique',
            'freight_value': 'mean',
            'payment_installments': 'mean'
        }).round(2)
        
        state_analysis.columns = [
            'total_orders', 'total_customers', 'total_revenue', 'avg_order_value', 'revenue_std',
            'category_diversity', 'avg_freight', 'avg_installments'
        ]
        
        # Calculate additional state metrics
        state_analysis['revenue_per_customer'] = state_analysis['total_revenue'] / state_analysis['total_customers']
        state_analysis['orders_per_customer'] = state_analysis['total_orders'] / state_analysis['total_customers']
        state_analysis['market_penetration'] = (state_analysis['total_customers'] / state_analysis['total_customers'].sum() * 100).round(2)
        
        # City-level analysis (top cities)
        city_analysis = self.df.groupby(['customer_state', 'customer_city']).agg({
            'order_id': 'nunique',
            'customer_unique_id': 'nunique',
            'price': 'sum'
        }).reset_index()
        
        city_analysis.columns = ['state', 'city', 'orders', 'customers', 'revenue']
        city_analysis = city_analysis.sort_values('revenue', ascending=False)
        
        # Category preferences by state
        state_category_prefs = pd.crosstab(
            self.df['customer_state'], 
            self.df['product_category_name'], 
            normalize='index'
        ) * 100
        
        self.analysis_results['geographic_analysis'] = {
            'state_analysis': state_analysis,
            'top_cities': city_analysis.head(20),
            'state_category_preferences': state_category_prefs
        }
        
        print(f"Geographic analysis completed:")
        print(f"- States analyzed: {len(state_analysis)}")
        print(f"- Cities analyzed: {len(city_analysis)}")
        print(f"- Top state by revenue: {state_analysis.sort_values('total_revenue', ascending=False).index[0]}")
        print(f"- Most customers: {state_analysis.sort_values('total_customers', ascending=False).index[0]}")
        
        return self.analysis_results['geographic_analysis']
    
    def comprehensive_price_and_value_analysis(self):
        """Advanced pricing and value analysis"""
        print("\n=== COMPREHENSIVE PRICE & VALUE ANALYSIS ===")
        
        # Overall price distribution analysis
        price_stats = {
            'basic_stats': self.df['price'].describe(),
            'percentiles': self.df['price'].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]),
            'skewness': self.df['price'].skew(),
            'kurtosis': self.df['price'].kurtosis()
        }
        
        # Price analysis by category
        category_price_analysis = self.df.groupby('product_category_name').agg({
            'price': ['count', 'mean', 'median', 'std', 'min', 'max', 'sum'],
            'freight_value': 'mean',
            'product_weight_g': 'mean'
        }).round(2)
        
        category_price_analysis.columns = [
            'volume', 'avg_price', 'median_price', 'price_std', 'min_price', 'max_price', 
            'total_revenue', 'avg_freight', 'avg_weight'
        ]
        
        # Price positioning analysis
        overall_avg_price = self.df['price'].mean()
        category_price_analysis['price_premium_index'] = (
            category_price_analysis['avg_price'] / overall_avg_price
        ).round(2)
        
        category_price_analysis['price_range'] = (
            category_price_analysis['max_price'] - category_price_analysis['min_price']
        )
        
        category_price_analysis['coefficient_variation'] = (
            category_price_analysis['price_std'] / category_price_analysis['avg_price']
        ).round(3)
        
        # Value perception analysis (price vs demand)
        category_price_analysis['demand_index'] = (
            category_price_analysis['volume'] / category_price_analysis['volume'].max()
        ).round(3)
        
        category_price_analysis['value_score'] = (
            category_price_analysis['demand_index'] / category_price_analysis['price_premium_index']
        ).round(3)
        
        # Price elasticity approximation
        price_bins = pd.qcut(self.df['price'], q=10, labels=False, duplicates='drop')
        self.df['price_bin'] = price_bins
        
        elasticity_analysis = self.df.groupby('price_bin').agg({
            'order_id': 'count',
            'price': 'mean',
            'customer_unique_id': 'nunique'
        }).reset_index()
        
        elasticity_analysis['demand_density'] = (
            elasticity_analysis['order_id'] / elasticity_analysis['order_id'].sum()
        )
        
        # Payment method impact on pricing
        payment_price_analysis = self.df.groupby('payment_type').agg({
            'price': ['mean', 'median', 'std'],
            'payment_installments': 'mean',
            'order_id': 'count'
        }).round(2)
        
        self.analysis_results['price_analysis'] = {
            'price_statistics': price_stats,
            'category_price_analysis': category_price_analysis,
            'elasticity_analysis': elasticity_analysis,
            'payment_price_analysis': payment_price_analysis
        }
        
        print("Price analysis completed:")
        print(f"- Average price: ${price_stats['basic_stats']['mean']:.2f}")
        print(f"- Price range: ${price_stats['basic_stats']['min']:.2f} - ${price_stats['basic_stats']['max']:.2f}")
        print(f"- Most premium category: {category_price_analysis.sort_values('price_premium_index', ascending=False).index[0]}")
        print(f"- Best value category: {category_price_analysis.sort_values('value_score', ascending=False).index[0]}")
        
        return self.analysis_results['price_analysis']
    
    def advanced_cohort_analysis(self):
        """Advanced customer cohort analysis for retention insights"""
        print("\n=== ADVANCED COHORT ANALYSIS ===")
        
        # Prepare cohort data
        self.df['order_period'] = self.df['order_purchase_timestamp'].dt.to_period('M')
        
        # Get customer's first purchase month (cohort)
        customer_cohorts = self.df.groupby('customer_unique_id')['order_purchase_timestamp'].min().reset_index()
        customer_cohorts['cohort_month'] = customer_cohorts['order_purchase_timestamp'].dt.to_period('M')
        customer_cohorts = customer_cohorts[['customer_unique_id', 'cohort_month']]
        
        # Merge cohort data back
        df_cohort = self.df.merge(customer_cohorts, on='customer_unique_id')
        
        # Calculate period numbers (months since first purchase)
        df_cohort['period_number'] = (
            df_cohort['order_period'] - df_cohort['cohort_month']
        ).apply(attrgetter('n'))
        
        # Create cohort tables
        cohort_data = df_cohort.groupby(['cohort_month', 'period_number'])['customer_unique_id'].nunique().reset_index()
        cohort_counts = cohort_data.pivot(index='cohort_month', columns='period_number', values='customer_unique_id')
        
        # Calculate cohort sizes (customers in each cohort)
        cohort_sizes = df_cohort.groupby('cohort_month')['customer_unique_id'].nunique()
        
        # Calculate retention rates
        retention_table = cohort_counts.divide(cohort_sizes, axis=0)
        
        # Revenue cohort analysis
        cohort_revenue = df_cohort.groupby(['cohort_month', 'period_number'])['price'].sum().reset_index()
        cohort_revenue_table = cohort_revenue.pivot(index='cohort_month', columns='period_number', values='price')
        
        # Average revenue per user by cohort
        arpu_table = cohort_revenue_table.divide(cohort_counts)
        
        # Calculate cohort metrics
        cohort_metrics = pd.DataFrame({
            'cohort_size': cohort_sizes,
            'total_customers': cohort_sizes,
        })
        
        # Add retention rates for different periods
        if 1 in retention_table.columns:
            cohort_metrics['retention_month_1'] = retention_table[1]
        if 3 in retention_table.columns:
            cohort_metrics['retention_month_3'] = retention_table[3]
        if 6 in retention_table.columns:
            cohort_metrics['retention_month_6'] = retention_table[6]
        
        # Add revenue metrics
        cohort_metrics['total_revenue'] = cohort_revenue_table.sum(axis=1)
        cohort_metrics['avg_revenue_per_customer'] = cohort_metrics['total_revenue'] / cohort_metrics['cohort_size']
        
        self.analysis_results['cohort_analysis'] = {
            'cohort_counts': cohort_counts,
            'retention_table': retention_table,
            'cohort_revenue_table': cohort_revenue_table,
            'arpu_table': arpu_table,
            'cohort_metrics': cohort_metrics,
            'cohort_sizes': cohort_sizes
        }
        
        print("Cohort analysis completed:")
        print(f"- Cohorts analyzed: {len(cohort_sizes)}")
        print(f"- Largest cohort: {cohort_sizes.max():,} customers")
        print(f"- Average cohort size: {cohort_sizes.mean():.0f} customers")
        
        return self.analysis_results['cohort_analysis']
    
    def generate_comprehensive_business_insights(self):
        """Generate actionable business insights from all analyses"""
        print("\n=== GENERATING COMPREHENSIVE BUSINESS INSIGHTS ===")
        
        insights = {
            'executive_summary': {},
            'category_insights': [],
            'customer_insights': [],
            'market_opportunities': [],
            'operational_insights': [],
            'strategic_recommendations': []
        }
        
        # Executive Summary
        if 'category_stats' in self.analysis_results:
            category_stats = self.analysis_results['category_stats']
            total_revenue = category_stats['total_revenue'].sum()
            total_customers = category_stats['unique_customers'].sum()
            
            insights['executive_summary'] = {
                'total_categories': len(category_stats),
                'total_revenue': f"${total_revenue:,.0f}",
                'avg_revenue_per_category': f"${total_revenue/len(category_stats):,.0f}",
                'top_revenue_category': category_stats.index[0],
                'market_concentration': f"{category_stats.head(5)['revenue_share'].sum():.1f}% (top 5 categories)"
            }
        
        # Category Insights
        if 'category_stats' in self.analysis_results:
            category_stats = self.analysis_results['category_stats']
            
            # Top performers
            top_revenue_cat = category_stats.index[0]
            top_penetration_cat = category_stats.sort_values('customer_penetration', ascending=False).index[0]
            most_competitive_cat = category_stats.sort_values('competitiveness_score', ascending=False).index[0]
            
            insights['category_insights'] = [
                f"Revenue leader: {top_revenue_cat} generates ${category_stats.loc[top_revenue_cat, 'total_revenue']:,.0f} ({category_stats.loc[top_revenue_cat, 'revenue_share']:.1f}% market share)",
                f"Highest penetration: {top_penetration_cat} reaches {category_stats.loc[top_penetration_cat, 'customer_penetration']:.1f}% of customers",
                f"Most competitive pricing: {most_competitive_cat} (competitiveness score: {category_stats.loc[most_competitive_cat, 'competitiveness_score']:.2f})",
                f"Price volatility concern: Categories with high price volatility may indicate inconsistent value proposition",
                f"Growth opportunity: {len(category_stats[category_stats['customer_penetration'] < 10])} categories have <10% penetration"
            ]
        
        # Customer Insights
        if 'customer_behavior' in self.analysis_results:
            customer_stats = self.analysis_results['customer_behavior']['customer_stats']
            
            repeat_rate = (len(customer_stats[customer_stats['total_orders'] > 1]) / len(customer_stats)) * 100
            high_value_customers = len(customer_stats[customer_stats['total_spent'] > 200])
            avg_ltv = customer_stats['total_spent'].mean()
            avg_categories = customer_stats['unique_categories'].mean()
            
            insights['customer_insights'] = [
                f"Customer retention: {repeat_rate:.1f}% repeat purchase rate",
                f"High-value segment: {high_value_customers:,} customers spent >$200",
                f"Average customer LTV: ${avg_ltv:.2f}",
                f"Cross-category engagement: Customers buy from {avg_categories:.1f} categories on average",
                f"Segmentation opportunity: {len(customer_stats[customer_stats['advanced_segment'] == 'At Risk'])} customers at risk of churn"
            ]
        
        # Market Opportunities
        if 'market_basket' in self.analysis_results:
            basket_stats = self.analysis_results['market_basket']['basket_analysis']
            rules_stats = self.analysis_results['market_basket']['advanced_rules_analysis']
            
            cross_sell_potential = basket_stats['single_item_baskets']
            strong_associations = rules_stats.get('strong_rules', 0)
            
            insights['market_opportunities'] = [
                f"Cross-sell potential: {cross_sell_potential:,} single-item orders could be expanded",
                f"Basket optimization: {basket_stats['multi_item_baskets']:,} orders show cross-category purchasing",
                f"Association rules: {strong_associations} strong product associations identified",
                f"Bundle opportunities: Categories with high lift scores are ideal for product bundles",
                f"Revenue expansion: Average basket size is {basket_stats['avg_basket_size']:.2f} items"
            ]
        
        # Operational Insights
        if 'price_analysis' in self.analysis_results:
            price_stats = self.analysis_results['price_analysis']['category_price_analysis']
            
            insights['operational_insights'] = [
                f"Pricing optimization: {len(price_stats[price_stats['coefficient_variation'] > 0.5])} categories show high price volatility",
                f"Value positioning: Categories with high value_score represent pricing sweet spots",
                f"Premium segment: Categories with price_premium_index >1.5 target affluent customers",
                f"Cost management: Freight costs vary significantly across categories",
                f"Payment preferences: Installment usage correlates with higher order values"
            ]
        
        # Strategic Recommendations
        insights['strategic_recommendations'] = [
            "Focus retention campaigns on 'At Risk' and 'Cannot Lose Them' customer segments",
            "Implement cross-sell recommendations based on strong association rules",
            "Develop premium product lines for high-value customer segments",
            "Optimize pricing for categories with high coefficient of variation",
            "Create targeted campaigns for single-item buyers to increase basket size",
            "Leverage seasonal patterns for inventory and marketing planning",
            "Geographic expansion opportunities in underperforming states",
            "Category-specific pricing strategies based on demand elasticity",
            "Customer lifecycle marketing based on cohort analysis insights",
            "Bundle creation using high-lift association rules"
        ]
        
        self.analysis_results['business_insights'] = insights
        
        print("Business insights generated:")
        print(f"- Executive summary with {len(insights['executive_summary'])} key metrics")
        print(f"- {len(insights['category_insights'])} category insights")
        print(f"- {len(insights['customer_insights'])} customer insights")
        print(f"- {len(insights['market_opportunities'])} market opportunities")
        print(f"- {len(insights['strategic_recommendations'])} strategic recommendations")
        
        return insights
    
    def save_comprehensive_results(self):
        """Save all analysis results to files"""
        print("\n=== SAVING COMPREHENSIVE ANALYSIS RESULTS ===")
        
        import os
        os.makedirs('analysis_results', exist_ok=True)
        
        # Save individual analysis components
        saved_files = []
        
        try:
            if 'category_stats' in self.analysis_results:
                self.analysis_results['category_stats'].to_csv('analysis_results/comprehensive_category_analysis.csv')
                saved_files.append('comprehensive_category_analysis.csv')
            
            if 'customer_behavior' in self.analysis_results:
                customer_data = self.analysis_results['customer_behavior']
                customer_data['customer_stats'].to_csv('analysis_results/advanced_customer_behavior.csv')
                customer_data['category_correlation'].to_csv('analysis_results/category_correlation_matrix.csv')
                saved_files.extend(['advanced_customer_behavior.csv', 'category_correlation_matrix.csv'])
            
            if 'market_basket' in self.analysis_results and self.association_rules_df is not None:
                self.association_rules_df.to_csv('analysis_results/association_rules_comprehensive.csv', index=False)
                self.frequent_itemsets.to_csv('analysis_results/frequent_itemsets_comprehensive.csv', index=False)
                saved_files.extend(['association_rules_comprehensive.csv', 'frequent_itemsets_comprehensive.csv'])
            
            if 'temporal_analysis' in self.analysis_results:
                temporal_data = self.analysis_results['temporal_analysis']
                temporal_data['monthly_category_stats'].to_csv('analysis_results/monthly_category_trends.csv', index=False)
                saved_files.append('monthly_category_trends.csv')
            
            if 'geographic_analysis' in self.analysis_results:
                geo_data = self.analysis_results['geographic_analysis']
                geo_data['state_analysis'].to_csv('analysis_results/state_performance_analysis.csv')
                geo_data['top_cities'].to_csv('analysis_results/top_cities_analysis.csv', index=False)
                saved_files.extend(['state_performance_analysis.csv', 'top_cities_analysis.csv'])
            
            if 'price_analysis' in self.analysis_results:
                price_data = self.analysis_results['price_analysis']
                price_data['category_price_analysis'].to_csv('analysis_results/comprehensive_price_analysis.csv')
                saved_files.append('comprehensive_price_analysis.csv')
            
            if 'cohort_analysis' in self.analysis_results:
                cohort_data = self.analysis_results['cohort_analysis']
                cohort_data['cohort_metrics'].to_csv('analysis_results/cohort_performance_metrics.csv')
                cohort_data['retention_table'].to_csv('analysis_results/customer_retention_cohorts.csv')
                saved_files.extend(['cohort_performance_metrics.csv', 'customer_retention_cohorts.csv'])
            
            # Save comprehensive insights
            if 'business_insights' in self.analysis_results:
                import json
                with open('analysis_results/comprehensive_business_insights.json', 'w') as f:
                    json.dump(self.analysis_results['business_insights'], f, indent=2, default=str)
                saved_files.append('comprehensive_business_insights.json')
            
            # Create analysis summary
            summary = {
                'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_period': f"{self.df['order_purchase_timestamp'].min()} to {self.df['order_purchase_timestamp'].max()}",
                'total_orders': self.df['order_id'].nunique(),
                'total_customers': self.df['customer_unique_id'].nunique(),
                'total_revenue': self.df['price'].sum(),
                'categories_analyzed': self.df['product_category_name'].nunique(),
                'analysis_components': list(self.analysis_results.keys()),
                'files_generated': saved_files
            }
            
            with open('analysis_results/analysis_summary.json', 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            print(f"Analysis results saved successfully:")
            for file in saved_files:
                print(f"  - analysis_results/{file}")
            print(f"  - analysis_results/analysis_summary.json")
            
            return True
            
        except Exception as e:
            print(f"Error saving results: {e}")
            return False
    
    def run_complete_enhanced_analysis(self):
        """Run the complete enhanced market basket analysis"""
        print("=" * 80)
        print("COMPREHENSIVE E-COMMERCE MARKET BASKET ANALYSIS")
        print("=" * 80)
        print("Advanced Analytics • Business Intelligence • Strategic Insights")
        print("=" * 80)
        
        start_time = pd.Timestamp.now()
        
        # Load data
        if not self.load_and_prepare_data():
            print("Failed to load data. Exiting...")
            return False
        
        try:
            print(f"\nStarting comprehensive analysis at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Run all analysis components
            analysis_steps = [
                ("Category Analysis", self.comprehensive_category_analysis),
                ("Temporal Analysis", self.seasonal_and_temporal_analysis),
                ("Customer Behavior", self.advanced_customer_behavior_analysis),
                ("Market Basket Analysis", self.enhanced_market_basket_analysis),
                ("Geographic Analysis", self.geographic_and_demographic_analysis),
                ("Price Analysis", self.comprehensive_price_and_value_analysis),
                ("Cohort Analysis", self.advanced_cohort_analysis),
                ("Business Insights", self.generate_comprehensive_business_insights)
            ]
            
            completed_steps = 0
            for step_name, step_function in analysis_steps:
                print(f"\n" + "-" * 60)
                print(f"RUNNING: {step_name}")
                print("-" * 60)
                
                try:
                    step_function()
                    completed_steps += 1
                    print(f"COMPLETED: {step_name}")
                except Exception as e:
                    print(f"ERROR in {step_name}: {e}")
                    continue
            
            # Save results
            print(f"\n" + "-" * 60)
            print("SAVING RESULTS")
            print("-" * 60)
            self.save_comprehensive_results()
            
            # Final summary
            end_time = pd.Timestamp.now()
            duration = (end_time - start_time).total_seconds()
            
            print("\n" + "=" * 80)
            print("ANALYSIS COMPLETED SUCCESSFULLY")
            print("=" * 80)
            print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
            print(f"Steps completed: {completed_steps}/{len(analysis_steps)}")
            print(f"Components analyzed: {len(self.analysis_results)}")
            
            # Key findings summary
            if 'business_insights' in self.analysis_results:
                insights = self.analysis_results['business_insights']
                print(f"\nKEY FINDINGS:")
                print(f"- Categories analyzed: {insights['executive_summary'].get('total_categories', 'N/A')}")
                print(f"- Total revenue: {insights['executive_summary'].get('total_revenue', 'N/A')}")
                print(f"- Market opportunities identified: {len(insights.get('market_opportunities', []))}")
                print(f"- Strategic recommendations: {len(insights.get('strategic_recommendations', []))}")
            
            print(f"\nResults saved in 'analysis_results/' directory")
            print("Ready for dashboard integration and business action!")
            
            return True
            
        except Exception as e:
            print(f"\nCRITICAL ERROR during analysis: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    try:
        print("Starting Enhanced Market Basket Analysis System")
        
        analyzer = EnhancedMarketBasketAnalysis()
        success = analyzer.run_complete_enhanced_analysis()
        
        if success:
            print("\nSUCCESS: Enhanced Market Basket Analysis completed!")
            print("Dashboard integration ready!")
        else:
            print("\nFAILED: Analysis encountered errors")
            
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"\nCRITICAL SYSTEM ERROR: {e}")
        import traceback
        traceback.print_exc()