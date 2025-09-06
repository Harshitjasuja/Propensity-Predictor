import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import random
import json
from datetime import datetime, timedelta

class IndividualCustomerPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = []
        self.model_loaded = False
        
    def load_trained_model(self):
        """Load the trained model and preprocessing components"""
        try:
            # Try to load the best model
            import os, glob
            
            model_files = glob.glob('models/best_*_model_*.pkl')
            if not model_files:
                print("No trained model found. Using fallback prediction method.")
                return False
                
            model_path = model_files[0]
            self.model = joblib.load(model_path)
            print(f"Model loaded from: {model_path}")
            
            # Load feature names if available
            try:
                feature_file = 'data/X_train.csv'
                if os.path.exists(feature_file):
                    sample_data = pd.read_csv(feature_file, nrows=1)
                    self.feature_names = sample_data.columns.tolist()
                    print(f"Feature names loaded: {len(self.feature_names)} features")
            except:
                print("Could not load feature names, using fallback method")
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def preprocess_customer_input(self, customer_data):
        """Preprocess customer input data for prediction"""
        try:
            # Create feature vector based on input
            features = {}
            
            # Demographic features
            features['age_group_encoded'] = self.encode_age_group(customer_data.get('age_group', '26-35'))
            features['income_level_encoded'] = self.encode_income_level(customer_data.get('income_level', 'Medium'))
            features['customer_state_encoded'] = self.encode_state(customer_data.get('customer_state', 'SP'))
            
            # Purchase history features
            features['previous_orders'] = customer_data.get('previous_orders', 1)
            features['avg_order_value'] = customer_data.get('avg_order_value', 75)
            features['days_since_last_purchase'] = customer_data.get('days_since_last_purchase', 30)
            features['category_diversity'] = len(customer_data.get('favorite_categories', ['Electronics']))
            
            # Behavioral features
            features['preferred_payment_encoded'] = self.encode_payment_type(customer_data.get('preferred_payment', 'credit_card'))
            features['uses_installments'] = 1 if customer_data.get('uses_installments', False) else 0
            features['price_sensitivity_encoded'] = self.encode_price_sensitivity(customer_data.get('price_sensitivity', 'Medium'))
            
            # Calculate derived features
            features['is_repeat_customer'] = 1 if features['previous_orders'] > 1 else 0
            features['high_value_customer'] = 1 if features['avg_order_value'] > 100 else 0
            features['recent_purchase'] = 1 if features['days_since_last_purchase'] <= 30 else 0
            features['diverse_buyer'] = 1 if features['category_diversity'] >= 3 else 0
            
            # Add temporal features (current date)
            now = datetime.now()
            features['purchase_hour'] = now.hour
            features['purchase_month'] = now.month
            features['is_weekend'] = 1 if now.weekday() >= 5 else 0
            
            return features
            
        except Exception as e:
            print(f"Error preprocessing data: {e}")
            return None
    
    def encode_age_group(self, age_group):
        """Encode age group"""
        age_mapping = {'18-25': 1, '26-35': 2, '36-45': 3, '46-55': 4, '56+': 5}
        return age_mapping.get(age_group, 2)
    
    def encode_income_level(self, income_level):
        """Encode income level"""
        income_mapping = {
            'Low (<$30k)': 1, 'Medium ($30k-$60k)': 2, 
            'High ($60k-$100k)': 3, 'Premium (>$100k)': 4
        }
        return income_mapping.get(income_level, 2)
    
    def encode_state(self, state):
        """Encode customer state"""
        # Simple encoding based on major states
        state_mapping = {'SP': 1, 'RJ': 2, 'MG': 3, 'RS': 4, 'PR': 5, 'SC': 6, 'BA': 7}
        return state_mapping.get(state, 8)  # 8 for 'Other'
    
    def encode_payment_type(self, payment_type):
        """Encode payment type"""
        payment_mapping = {'credit_card': 1, 'boleto': 2, 'debit_card': 3, 'voucher': 4}
        return payment_mapping.get(payment_type, 1)
    
    def encode_price_sensitivity(self, sensitivity):
        """Encode price sensitivity"""
        sensitivity_mapping = {'Very Low': 1, 'Low': 2, 'Medium': 3, 'High': 4, 'Very High': 5}
        return sensitivity_mapping.get(sensitivity, 3)
    
    def predict_purchase_propensity(self, customer_data):
        """Predict purchase propensity for a customer"""
        if self.model_loaded and self.model is not None:
            try:
                # Use trained model
                features = self.preprocess_customer_input(customer_data)
                if features is None:
                    return self.fallback_prediction(customer_data)
                
                # Create feature vector (simplified)
                feature_vector = np.array([[
                    features['previous_orders'],
                    features['avg_order_value'],
                    features['days_since_last_purchase'],
                    features['category_diversity'],
                    features['age_group_encoded'],
                    features['income_level_encoded'],
                    features['is_repeat_customer'],
                    features['high_value_customer'],
                    features['recent_purchase'],
                    features['diverse_buyer']
                ]])
                
                # Get prediction probability
                if hasattr(self.model, 'predict_proba'):
                    propensity = self.model.predict_proba(feature_vector)[0][1]  # Probability of class 1
                else:
                    prediction = self.model.predict(feature_vector)[0]
                    propensity = max(0.1, min(0.9, prediction))  # Ensure reasonable range
                
                return propensity
                
            except Exception as e:
                print(f"Error using trained model: {e}")
                return self.fallback_prediction(customer_data)
        else:
            return self.fallback_prediction(customer_data)
    
    def fallback_prediction(self, customer_data):
        """Fallback prediction method when model is not available"""
        base_score = 0.35
        
        # Age factor
        age_multipliers = {"18-25": 1.1, "26-35": 1.2, "36-45": 1.15, "46-55": 1.05, "56+": 0.9}
        age_group = customer_data.get('age_group', '26-35')
        base_score *= age_multipliers.get(age_group, 1.0)
        
        # Income factor
        income_multipliers = {
            "Low (<$30k)": 0.8, "Medium ($30k-$60k)": 1.0, 
            "High ($60k-$100k)": 1.3, "Premium (>$100k)": 1.6
        }
        income_level = customer_data.get('income_level', 'Medium ($30k-$60k)')
        base_score *= income_multipliers.get(income_level, 1.0)
        
        # Purchase history factors
        previous_orders = customer_data.get('previous_orders', 1)
        if previous_orders == 0:
            base_score *= 0.6  # New customer
        elif previous_orders >= 5:
            base_score *= 1.4  # Loyal customer
        elif previous_orders >= 2:
            base_score *= 1.2  # Repeat customer
        
        # Order value factor
        avg_order_value = customer_data.get('avg_order_value', 75)
        if avg_order_value > 150:
            base_score *= 1.2
        elif avg_order_value < 50:
            base_score *= 0.9
        
        # Recency factor
        days_since_last_purchase = customer_data.get('days_since_last_purchase', 30)
        if days_since_last_purchase <= 7:
            base_score *= 1.3
        elif days_since_last_purchase <= 30:
            base_score *= 1.1
        elif days_since_last_purchase > 180:
            base_score *= 0.7
        
        # Category diversity factor
        favorite_categories = customer_data.get('favorite_categories', ['Electronics'])
        if len(favorite_categories) >= 3:
            base_score *= 1.15
        
        # Payment behavior factor
        preferred_payment = customer_data.get('preferred_payment', 'credit_card')
        if preferred_payment == 'credit_card':
            base_score *= 1.1
        
        uses_installments = customer_data.get('uses_installments', False)
        if uses_installments:
            base_score *= 1.05
        
        # Price sensitivity factor
        price_sensitivity = customer_data.get('price_sensitivity', 'Medium')
        sensitivity_multipliers = {
            "Very Low": 1.2, "Low": 1.1, "Medium": 1.0, "High": 0.9, "Very High": 0.8
        }
        base_score *= sensitivity_multipliers.get(price_sensitivity, 1.0)
        
        # Add random variance for realism
        variance = random.uniform(-0.05, 0.05)
        final_score = max(0.05, min(0.95, base_score + variance))
        
        return final_score
    
    def generate_dynamic_discount(self, propensity_score, avg_order_value, price_sensitivity):
        """Generate dynamic discount based on propensity and customer profile"""
        base_discount = 5  # Minimum discount
        
        # Propensity-based discount (inverse relationship - lower propensity gets higher discount)
        if propensity_score < 0.3:
            propensity_discount = 25  # High discount for low propensity
        elif propensity_score < 0.5:
            propensity_discount = 18
        elif propensity_score < 0.7:
            propensity_discount = 12
        else:
            propensity_discount = 7  # Small discount for high propensity
        
        # Price sensitivity adjustment
        sensitivity_adjustment = {
            "Very High": 1.5, "High": 1.3, "Medium": 1.0, "Low": 0.8, "Very Low": 0.6
        }
        
        final_discount = base_discount + (propensity_discount * sensitivity_adjustment.get(price_sensitivity, 1.0))
        final_discount = max(5, min(final_discount, 35))  # Cap between 5% and 35%
        
        # Calculate savings
        savings_amount = avg_order_value * (final_discount / 100)
        
        # Generate offer details
        if propensity_score < 0.4:
            offer_type = "🎯 RETENTION SPECIAL"
            urgency_message = "We miss you! Here's an exclusive offer to welcome you back."
            campaign_type = "Win-Back Campaign"
        elif final_discount > 20:
            offer_type = "🔥 FLASH SALE"
            urgency_message = "Limited time! This amazing discount expires in 48 hours."
            campaign_type = "High-Impact Conversion"
        elif propensity_score > 0.7:
            offer_type = "⭐ VIP APPRECIATION"
            urgency_message = "Thank you for being a valued customer!"
            campaign_type = "Loyalty Reward"
        else:
            offer_type = "💰 PERSONALIZED DEAL"
            urgency_message = "Specially curated offer just for you!"
            campaign_type = "Standard Promotion"
        
        return {
            'discount_percent': final_discount,
            'savings_amount': savings_amount,
            'offer_type': offer_type,
            'urgency_message': urgency_message,
            'campaign_type': campaign_type,
            'validity_days': 7 if propensity_score < 0.4 else 3,
            'recommendation_confidence': 'High' if propensity_score < 0.4 or propensity_score > 0.7 else 'Medium'
        }
    
    def generate_personalized_recommendations(self, customer_data, propensity_score):
        """Generate personalized product recommendations"""
        favorite_categories = customer_data.get('favorite_categories', ['Electronics'])
        
        # Category-based recommendations
        category_recommendations = {
            "Electronics": [
                {"category": "IT Accessories", "base_price": 45, "appeal_score": 0.85},
                {"category": "Telephony", "base_price": 120, "appeal_score": 0.75}
            ],
            "Beauty & Health": [
                {"category": "Perfumery", "base_price": 65, "appeal_score": 0.80},
                {"category": "Health & Wellness", "base_price": 35, "appeal_score": 0.70}
            ],
            "Furniture & Decor": [
                {"category": "Bed, Table & Bath", "base_price": 85, "appeal_score": 0.75},
                {"category": "Home Comfort", "base_price": 95, "appeal_score": 0.80}
            ],
            "Sports & Leisure": [
                {"category": "Outdoor Equipment", "base_price": 110, "appeal_score": 0.70},
                {"category": "Fitness", "base_price": 75, "appeal_score": 0.85}
            ]
        }
        
        recommendations = []
        
        # Add recommendations based on favorite categories
        for fav_cat in favorite_categories[:2]:
            if fav_cat in category_recommendations:
                for rec in category_recommendations[fav_cat]:
                    rec['relevance_score'] = propensity_score * rec['appeal_score']
                    rec['reasoning'] = f"Matches your interest in {fav_cat}"
                    rec['confidence'] = 'High' if rec['relevance_score'] > 0.6 else 'Medium'
                    recommendations.append(rec)
        
        # Add trending/popular recommendations
        trending_recs = [
            {
                "category": "Beauty & Health", "base_price": 55, "appeal_score": 0.75,
                "reasoning": "Trending category with high satisfaction", "relevance_score": propensity_score * 0.7
            },
            {
                "category": "Home & Garden", "base_price": 85, "appeal_score": 0.70,
                "reasoning": "Popular choice for home improvement", "relevance_score": propensity_score * 0.65
            },
            {
                "category": "Books & Media", "base_price": 35, "appeal_score": 0.80,
                "reasoning": "High engagement category", "relevance_score": propensity_score * 0.60
            }
        ]
        
        # Add confidence levels to trending recommendations
        for rec in trending_recs:
            rec['confidence'] = 'High' if rec['relevance_score'] > 0.5 else 'Medium'
        
        recommendations.extend(trending_recs)
        
        # Sort by relevance score and return top recommendations
        recommendations = sorted(recommendations, key=lambda x: x['relevance_score'], reverse=True)
        
        # Add personalization factors
        for i, rec in enumerate(recommendations):
            # Adjust price based on customer profile
            income_level = customer_data.get('income_level', 'Medium ($30k-$60k)')
            if 'Premium' in income_level or 'High' in income_level:
                rec['suggested_price'] = rec['base_price'] * 1.2  # Premium pricing
            else:
                rec['suggested_price'] = rec['base_price']
            
            # Add recommendation rank
            rec['rank'] = i + 1
        
        return recommendations[:6]  # Return top 6 recommendations
    
    def calculate_customer_lifetime_value(self, customer_data, propensity_score):
        """Calculate predicted customer lifetime value"""
        base_ltv = customer_data.get('avg_order_value', 75)
        previous_orders = customer_data.get('previous_orders', 1)
        
        # Estimate future purchase frequency based on propensity
        monthly_purchase_probability = propensity_score * 0.3  # Max 30% chance per month
        
        # Calculate 12-month projected LTV
        projected_orders_12m = monthly_purchase_probability * 12
        projected_ltv_12m = projected_orders_12m * base_ltv
        
        # Historical LTV factor
        if previous_orders > 1:
            historical_ltv = previous_orders * customer_data.get('avg_order_value', 75)
            # Weight between historical and projected
            final_ltv = (historical_ltv * 0.7) + (projected_ltv_12m * 0.3)
        else:
            final_ltv = projected_ltv_12m
        
        return {
            'projected_ltv_12m': projected_ltv_12m,
            'historical_ltv': previous_orders * customer_data.get('avg_order_value', 75),
            'final_ltv_estimate': final_ltv,
            'monthly_purchase_probability': monthly_purchase_probability,
            'projected_orders_12m': projected_orders_12m
        }
    
    def generate_campaign_strategy(self, customer_data, propensity_score, discount_info):
        """Generate comprehensive campaign strategy"""
        strategy = {
            'primary_objective': '',
            'campaign_type': '',
            'communication_channels': [],
            'timing_strategy': '',
            'budget_allocation': '',
            'success_metrics': [],
            'follow_up_actions': []
        }
        
        # Determine primary objective based on propensity
        if propensity_score < 0.3:
            strategy['primary_objective'] = 'Customer Retention & Re-engagement'
            strategy['campaign_type'] = 'Win-Back Campaign'
            strategy['communication_channels'] = ['Email', 'SMS', 'Push Notification', 'Direct Mail']
            strategy['timing_strategy'] = 'Immediate deployment with 7-day follow-up sequence'
            strategy['budget_allocation'] = 'High Priority - Premium budget allocation'
            
        elif propensity_score < 0.5:
            strategy['primary_objective'] = 'Purchase Conversion'
            strategy['campaign_type'] = 'Conversion Campaign'
            strategy['communication_channels'] = ['Email', 'SMS', 'Retargeting Ads']
            strategy['timing_strategy'] = 'Deploy within 24 hours, follow-up in 3 days'
            strategy['budget_allocation'] = 'Standard budget allocation'
            
        elif propensity_score < 0.7:
            strategy['primary_objective'] = 'Cross-sell & Upsell'
            strategy['campaign_type'] = 'Growth Campaign'
            strategy['communication_channels'] = ['Email', 'In-app messaging']
            strategy['timing_strategy'] = 'Optimal timing based on purchase patterns'
            strategy['budget_allocation'] = 'Moderate budget allocation'
            
        else:
            strategy['primary_objective'] = 'Customer Delight & Loyalty'
            strategy['campaign_type'] = 'Loyalty Campaign'
            strategy['communication_channels'] = ['Email', 'VIP notifications']
            strategy['timing_strategy'] = 'Premium timing with personalized touch'
            strategy['budget_allocation'] = 'VIP budget allocation'
        
        # Success metrics
        strategy['success_metrics'] = [
            'Click-through rate',
            'Conversion rate',
            'Revenue per customer',
            'Customer satisfaction score'
        ]
        
        # Follow-up actions
        if propensity_score < 0.4:
            strategy['follow_up_actions'] = [
                'Send reminder after 3 days',
                'Offer additional incentive after 7 days',
                'Personal outreach after 14 days'
            ]
        else:
            strategy['follow_up_actions'] = [
                'Send thank you message after purchase',
                'Request feedback after delivery',
                'Offer complementary products'
            ]
        
        return strategy
    
    def predict_and_recommend(self, customer_data):
        """Main function to predict and generate comprehensive recommendations"""
        try:
            # Load model if not already loaded
            if not self.model_loaded:
                self.load_trained_model()
            
            # Predict propensity
            propensity_score = self.predict_purchase_propensity(customer_data)
            
            # Generate dynamic discount
            discount_info = self.generate_dynamic_discount(
                propensity_score, 
                customer_data.get('avg_order_value', 75),
                customer_data.get('price_sensitivity', 'Medium')
            )
            
            # Generate product recommendations
            recommendations = self.generate_personalized_recommendations(customer_data, propensity_score)
            
            # Calculate LTV
            ltv_analysis = self.calculate_customer_lifetime_value(customer_data, propensity_score)
            
            # Generate campaign strategy
            campaign_strategy = self.generate_campaign_strategy(customer_data, propensity_score, discount_info)
            
            # Risk assessment
            risk_level = "Low" if propensity_score > 0.6 else "Medium" if propensity_score > 0.3 else "High"
            opportunity_level = "High" if propensity_score > 0.7 else "Medium" if propensity_score > 0.4 else "Low"
            
            # Customer segment classification
            segment = self.classify_customer_segment(customer_data, propensity_score)
            
            # Compile comprehensive results
            results = {
                'prediction_results': {
                    'propensity_score': propensity_score,
                    'propensity_percentage': propensity_score * 100,
                    'risk_level': risk_level,
                    'opportunity_level': opportunity_level,
                    'customer_segment': segment,
                    'prediction_confidence': 'High' if self.model_loaded else 'Medium'
                },
                'discount_offer': discount_info,
                'product_recommendations': recommendations,
                'ltv_analysis': ltv_analysis,
                'campaign_strategy': campaign_strategy,
                'analysis_metadata': {
                    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'model_used': 'Trained ML Model' if self.model_loaded else 'Heuristic Model',
                    'features_analyzed': len(self.preprocess_customer_input(customer_data) or {}),
                    'recommendation_engine': 'Advanced Personalization'
                }
            }
            
            return results
            
        except Exception as e:
            print(f"Error in prediction and recommendation: {e}")
            return None
    
    def classify_customer_segment(self, customer_data, propensity_score):
        """Classify customer into business segment"""
        previous_orders = customer_data.get('previous_orders', 1)
        avg_order_value = customer_data.get('avg_order_value', 75)
        days_since_last_purchase = customer_data.get('days_since_last_purchase', 30)
        
        if previous_orders == 0:
            return "New Customer"
        elif previous_orders == 1 and days_since_last_purchase > 90:
            return "One-Time Buyer"
        elif propensity_score > 0.7 and avg_order_value > 150:
            return "VIP Customer"
        elif previous_orders >= 5 and propensity_score > 0.6:
            return "Loyal Customer"
        elif avg_order_value > 100:
            return "High-Value Customer"
        elif propensity_score < 0.3:
            return "At-Risk Customer"
        else:
            return "Regular Customer"
    
    def export_results_to_json(self, results, filename=None):
        """Export prediction results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'customer_prediction_{timestamp}.json'
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results exported to: {filename}")
            return filename
        except Exception as e:
            print(f"Error exporting results: {e}")
            return None
    
    def batch_predict(self, customer_list):
        """Process multiple customers in batch"""
        batch_results = []
        
        for i, customer_data in enumerate(customer_list):
            print(f"Processing customer {i+1}/{len(customer_list)}")
            result = self.predict_and_recommend(customer_data)
            if result:
                result['customer_id'] = customer_data.get('customer_id', f'customer_{i+1}')
                batch_results.append(result)
        
        return batch_results

def demo_individual_prediction():
    """Demo function to show individual prediction capabilities"""
    print("Individual Customer Predictor Demo")
    print("=" * 50)
    
    # Initialize predictor
    predictor = IndividualCustomerPredictor()
    
    # Sample customer data
    sample_customer = {
        'customer_id': 'demo_customer_001',
        'age_group': '26-35',
        'income_level': 'High ($60k-$100k)',
        'customer_state': 'SP',
        'customer_city': 'São Paulo',
        'previous_orders': 3,
        'avg_order_value': 125.50,
        'days_since_last_purchase': 45,
        'favorite_categories': ['Electronics', 'Beauty & Health'],
        'preferred_payment': 'credit_card',
        'uses_installments': True,
        'price_sensitivity': 'Medium'
    }
    
    print("Sample Customer Profile:")
    for key, value in sample_customer.items():
        print(f"  {key}: {value}")
    
    print("\nGenerating Predictions and Recommendations...")
    
    # Get predictions
    results = predictor.predict_and_recommend(sample_customer)
    
    if results:
        print("\n" + "=" * 50)
        print("PREDICTION RESULTS")
        print("=" * 50)
        
        pred = results['prediction_results']
        print(f"Purchase Propensity: {pred['propensity_percentage']:.1f}%")
        print(f"Customer Segment: {pred['customer_segment']}")
        print(f"Risk Level: {pred['risk_level']}")
        print(f"Opportunity Level: {pred['opportunity_level']}")
        
        print("\nDYNAMIC DISCOUNT OFFER:")
        discount = results['discount_offer']
        print(f"  {discount['offer_type']}")
        print(f"  Discount: {discount['discount_percent']:.0f}%")
        print(f"  Savings: ${discount['savings_amount']:.2f}")
        print(f"  Message: {discount['urgency_message']}")
        
        print("\nTOP PRODUCT RECOMMENDATIONS:")
        for rec in results['product_recommendations'][:3]:
            print(f"  {rec['rank']}. {rec['category']} - ${rec['suggested_price']:.2f}")
            print(f"     Relevance: {rec['relevance_score']:.2f} | {rec['reasoning']}")
        
        print("\nLIFETIME VALUE ANALYSIS:")
        ltv = results['ltv_analysis']
        print(f"  Projected 12M LTV: ${ltv['projected_ltv_12m']:.2f}")
        print(f"  Monthly Purchase Probability: {ltv['monthly_purchase_probability']:.1%}")
        
        print(f"\nCAMPAIGN STRATEGY:")
        strategy = results['campaign_strategy']
        print(f"  Objective: {strategy['primary_objective']}")
        print(f"  Campaign Type: {strategy['campaign_type']}")
        print(f"  Channels: {', '.join(strategy['communication_channels'])}")
        
    else:
        print("Error generating predictions")

if __name__ == "__main__":
    demo_individual_prediction()