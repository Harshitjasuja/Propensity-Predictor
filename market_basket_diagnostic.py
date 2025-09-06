import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')

def diagnose_market_basket_data():
    """Diagnostic function to understand market basket analysis issues"""
    print("🔍 DIAGNOSING MARKET BASKET ANALYSIS DATA")
    print("=" * 60)
    
    try:
        # Load data
        df = pd.read_csv('data/merged_orders.csv')
        print(f"✅ Data loaded: {df.shape}")
        
        # Check for NaN values
        nan_count = df['product_category_name'].isna().sum()
        print(f"📊 NaN values in product_category_name: {nan_count}")
        
        # Clean data
        df_clean = df.dropna(subset=['product_category_name']).copy()
        df_clean['product_category_name'] = df_clean['product_category_name'].astype(str)
        print(f"✅ Cleaned data shape: {df_clean.shape}")
        
        # Analyze transaction patterns
        print(f"\n📈 TRANSACTION ANALYSIS:")
        
        # Basic transaction stats
        total_orders = df_clean['order_id'].nunique()
        total_items = len(df_clean)
        avg_items_per_order = total_items / total_orders
        
        print(f"   • Total unique orders: {total_orders:,}")
        print(f"   • Total items: {total_items:,}")
        print(f"   • Average items per order: {avg_items_per_order:.2f}")
        
        # Basket size distribution
        basket_sizes = df_clean.groupby('order_id')['product_category_name'].count()
        print(f"\n🛒 BASKET SIZE DISTRIBUTION:")
        print(f"   • Single-item orders: {(basket_sizes == 1).sum():,} ({(basket_sizes == 1).mean()*100:.1f}%)")
        print(f"   • Multi-item orders: {(basket_sizes > 1).sum():,} ({(basket_sizes > 1).mean()*100:.1f}%)")
        print(f"   • Orders with 3+ items: {(basket_sizes >= 3).sum():,} ({(basket_sizes >= 3).mean()*100:.1f}%)")
        print(f"   • Orders with 5+ items: {(basket_sizes >= 5).sum():,} ({(basket_sizes >= 5).mean()*100:.1f}%)")
        print(f"   • Maximum basket size: {basket_sizes.max()}")
        
        # Category frequency analysis
        category_freq = df_clean['product_category_name'].value_counts()
        print(f"\n🏷️ CATEGORY FREQUENCY:")
        print(f"   • Total categories: {len(category_freq)}")
        print(f"   • Most frequent category: {category_freq.index[0]} ({category_freq.iloc[0]:,} items)")
        print(f"   • Least frequent category: {category_freq.index[-1]} ({category_freq.iloc[-1]:,} items)")
        
        # Show top 10 categories
        print(f"\n📊 TOP 10 CATEGORIES BY FREQUENCY:")
        for i, (cat, freq) in enumerate(category_freq.head(10).items(), 1):
            percentage = (freq / len(df_clean)) * 100
            print(f"   {i:2d}. {cat}: {freq:,} items ({percentage:.1f}%)")
        
        # Create transactions for market basket analysis
        transactions = df_clean.groupby('order_id')['product_category_name'].apply(list).tolist()
        multi_item_transactions = [t for t in transactions if len(t) > 1]
        
        print(f"\n🔗 ASSOCIATION ANALYSIS READINESS:")
        print(f"   • Total transactions: {len(transactions):,}")
        print(f"   • Multi-item transactions: {len(multi_item_transactions):,}")
        print(f"   • Multi-item percentage: {len(multi_item_transactions)/len(transactions)*100:.1f}%")
        
        if len(multi_item_transactions) == 0:
            print("❌ NO MULTI-ITEM TRANSACTIONS FOUND!")
            print("   Cannot perform association rule mining.")
            return False
        
        # Test transaction encoding
        print(f"\n🧪 TESTING TRANSACTION ENCODING:")
        
        try:
            te = TransactionEncoder()
            te_ary = te.fit(multi_item_transactions).transform(multi_item_transactions)
            transaction_df = pd.DataFrame(te_ary, columns=te.columns_)
            
            print(f"   ✅ Transaction matrix created: {transaction_df.shape}")
            print(f"   • Categories in matrix: {len(te.columns_)}")
            print(f"   • Transactions encoded: {len(transaction_df)}")
            
            # Check support levels for different categories
            support_analysis = transaction_df.mean().sort_values(ascending=False)
            print(f"\n📈 CATEGORY SUPPORT LEVELS:")
            print(f"   • Highest support: {support_analysis.index[0]} ({support_analysis.iloc[0]:.4f})")
            print(f"   • Lowest support: {support_analysis.index[-1]} ({support_analysis.iloc[-1]:.4f})")
            print(f"   • Categories with >1% support: {(support_analysis > 0.01).sum()}")
            print(f"   • Categories with >0.5% support: {(support_analysis > 0.005).sum()}")
            
            # Test frequent itemsets with very low support
            print(f"\n🔍 TESTING FREQUENT ITEMSETS:")
            
            support_levels = [0.1, 0.05, 0.02, 0.01, 0.005, 0.001]
            
            for support in support_levels:
                try:
                    frequent_itemsets = apriori(transaction_df, min_support=support, use_colnames=True)
                    if len(frequent_itemsets) > 0:
                        print(f"   ✅ Support {support}: {len(frequent_itemsets)} itemsets found")
                        
                        # Show itemset size distribution
                        itemset_sizes = frequent_itemsets['itemsets'].apply(len)
                        print(f"      - Single items: {(itemset_sizes == 1).sum()}")
                        print(f"      - Pairs: {(itemset_sizes == 2).sum()}")
                        print(f"      - Triplets: {(itemset_sizes == 3).sum()}")
                        
                        # Test association rules
                        if len(frequent_itemsets[itemset_sizes >= 2]) > 0:
                            try:
                                rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
                                print(f"      - Association rules: {len(rules)}")
                                break
                            except Exception as rule_error:
                                print(f"      - Rule generation failed: {rule_error}")
                        else:
                            print(f"      - No itemsets of size 2+ for rules")
                    else:
                        print(f"   ❌ Support {support}: No itemsets found")
                except Exception as e:
                    print(f"   ❌ Support {support}: Error - {e}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Transaction encoding failed: {e}")
            print(f"   • Sample transactions: {multi_item_transactions[:3]}")
            return False
            
    except Exception as e:
        print(f"❌ DIAGNOSTIC FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def quick_market_basket_test():
    """Quick test of market basket analysis with very low thresholds"""
    print("\n🚀 QUICK MARKET BASKET TEST")
    print("=" * 50)
    
    try:
        # Load and clean data
        df = pd.read_csv('data/merged_orders.csv')
        df_clean = df.dropna(subset=['product_category_name']).copy()
        df_clean['product_category_name'] = df_clean['product_category_name'].astype(str)
        
        # Create transactions
        transactions = df_clean.groupby('order_id')['product_category_name'].apply(list).tolist()
        multi_item_transactions = [t for t in transactions if len(t) > 1]
        
        print(f"Multi-item transactions: {len(multi_item_transactions)}")
        
        if len(multi_item_transactions) == 0:
            print("❌ No multi-item transactions for analysis")
            return
        
        # Encode transactions
        te = TransactionEncoder()
        te_ary = te.fit(multi_item_transactions).transform(multi_item_transactions)
        transaction_df = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Find frequent itemsets with very low support
        frequent_itemsets = apriori(transaction_df, min_support=0.001, use_colnames=True)
        print(f"Frequent itemsets found: {len(frequent_itemsets)}")
        
        if len(frequent_itemsets) > 0:
            # Generate association rules
            try:
                rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.01)
                print(f"Association rules generated: {len(rules)}")
                
                if len(rules) > 0:
                    # Show top rules
                    print(f"\n🏆 TOP 5 ASSOCIATION RULES:")
                    top_rules = rules.nlargest(5, 'lift')
                    for i, (_, rule) in enumerate(top_rules.iterrows(), 1):
                        antecedent = list(rule['antecedents'])[0] if len(rule['antecedents']) == 1 else str(rule['antecedents'])
                        consequent = list(rule['consequents'])[0] if len(rule['consequents']) == 1 else str(rule['consequents'])
                        print(f"   {i}. {antecedent} → {consequent}")
                        print(f"      Support: {rule['support']:.4f}, Confidence: {rule['confidence']:.4f}, Lift: {rule['lift']:.2f}")
                    
                    # Save results
                    import os
                    os.makedirs('analysis_results', exist_ok=True)
                    rules.to_csv('analysis_results/association_rules.csv', index=False)
                    frequent_itemsets.to_csv('analysis_results/frequent_itemsets.csv', index=False)
                    print(f"\n✅ Results saved to analysis_results/")
                    
                else:
                    print("❌ No association rules met the minimum thresholds")
            except Exception as e:
                print(f"❌ Error generating association rules: {e}")
        else:
            print("❌ No frequent itemsets found")
            
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run diagnostic first
    success = diagnose_market_basket_data()
    
    if success:
        print(f"\n" + "="*60)
        # Run quick test
        quick_market_basket_test()
    else:
        print("\n❌ Diagnostic failed - check data quality issues")