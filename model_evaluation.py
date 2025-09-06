import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, roc_auc_score
)
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveModelEvaluator:
    def __init__(self):
        self.model = None
        self.model_name = "XGBoost Purchase Propensity"
        self.results = {}
        
    def load_model_and_data(self):
        """Load the trained model and test data"""
        print("=== LOADING MODEL AND TEST DATA ===")
        
        # Load model
        self.model = joblib.load('models/best_unbiased_model_xgboost.pkl')
        
        # Load test data
        X_test = pd.read_csv('data/X_test.csv')
        y_test = pd.read_csv('data/y_test.csv')['will_repurchase']
        
        print(f"✅ Model loaded: {type(self.model).__name__}")
        print(f"✅ Test data: {X_test.shape}")
        print(f"✅ Test target distribution: {y_test.value_counts().to_dict()}")
        
        return X_test, y_test
    
    def generate_predictions(self, X_test, y_test):
        """Generate predictions and probabilities"""
        print("=== GENERATING PREDICTIONS ===")
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        print(f"✅ Predictions generated for {len(X_test)} customers")
        print(f"✅ Probability range: {y_pred_proba.min():.3f} - {y_pred_proba.max():.3f}")
        
        return y_pred, y_pred_proba
    
    def calculate_performance_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive performance metrics"""
        print("=== CALCULATING PERFORMANCE METRICS ===")
        
        # Core metrics
        auc_score = roc_auc_score(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        # Classification metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics = {
            'auc_roc': auc_score,
            'average_precision': avg_precision,
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
        
        print("✅ Performance Metrics:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"   {metric:20}: {value:.4f}")
        
        self.results['metrics'] = metrics
        return metrics
    
    def create_confusion_matrix_plot(self, y_true, y_pred):
        """Create enhanced confusion matrix visualization"""
        print("=== CREATING CONFUSION MATRIX ===")
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Won\'t Repurchase', 'Will Repurchase'],
                    yticklabels=['Won\'t Repurchase', 'Will Repurchase'])
        
        plt.title(f'Confusion Matrix - {self.model_name}', fontsize=16, fontweight='bold')
        plt.ylabel('Actual', fontsize=14)
        plt.xlabel('Predicted', fontsize=14)
        
        # Add percentage annotations
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                percentage = cm[i, j] / total * 100
                plt.text(j + 0.5, i + 0.8, f'({percentage:.1f}%)', 
                        ha='center', va='center', color='red', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('models/confusion_matrix_detailed.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Enhanced confusion matrix saved")
    
    def create_roc_curve_plot(self, y_true, y_pred_proba):
        """Create ROC curve with detailed analysis"""
        print("=== CREATING ROC CURVE ===")
        
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=3, 
                label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier (AUC = 0.5)')
        
        # Find optimal threshold (Youden's J statistic)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10,
                label=f'Optimal Threshold = {optimal_threshold:.3f}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {self.model_name}', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('models/roc_curve_detailed.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ ROC curve saved (Optimal threshold: {optimal_threshold:.3f})")
        return optimal_threshold
    
    def create_precision_recall_curve(self, y_true, y_pred_proba):
        """Create precision-recall curve"""
        print("=== CREATING PRECISION-RECALL CURVE ===")
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        baseline = y_true.mean()
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=3,
                label=f'PR Curve (AP = {avg_precision:.4f})')
        plt.axhline(y=baseline, color='red', linestyle='--', lw=2,
                   label=f'Baseline (Random = {baseline:.4f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - {self.model_name}', fontsize=16, fontweight='bold')
        plt.legend(loc="lower left", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('models/precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Precision-Recall curve saved")
    
    def analyze_feature_importance(self, X_test):
        """Analyze and visualize feature importance"""
        print("=== ANALYZING FEATURE IMPORTANCE ===")
        
        # Get feature importance
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_names = X_test.columns
            
            # Create feature importance DataFrame
            feature_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances,
                'importance_pct': importances / importances.sum() * 100
            }).sort_values('importance', ascending=False)
            
            print("✅ Top 10 Most Important Features:")
            for i, row in feature_df.head(10).iterrows():
                print(f"   {row['feature']:25}: {row['importance']:.4f} ({row['importance_pct']:.1f}%)")
            
            # Plot feature importance
            plt.figure(figsize=(12, 10))
            top_features = feature_df.head(15)
            
            bars = plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance', fontsize=12)
            plt.title(f'Top 15 Feature Importance - {self.model_name}', 
                     fontsize=16, fontweight='bold')
            plt.gca().invert_yaxis()
            
            # Add percentage labels
            for i, (bar, pct) in enumerate(zip(bars, top_features['importance_pct'])):
                plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{pct:.1f}%', va='center', fontsize=10)
            
            plt.tight_layout()
            plt.savefig('models/feature_importance_detailed.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Feature importance plot saved")
            
            self.results['feature_importance'] = feature_df.to_dict('records')
            return feature_df
        else:
            print("❌ Model doesn't support feature importance")
            return None
    
    def business_impact_analysis(self, y_true, y_pred_proba):
        """Comprehensive business impact analysis"""
        print("=== BUSINESS IMPACT ANALYSIS ===")
        
        baseline_rate = y_true.mean()
        total_customers = len(y_true)
        
        # Analyze different targeting strategies
        strategies = {}
        percentiles = [5, 10, 15, 20, 25, 30, 40, 50]
        
        for pct in percentiles:
            threshold = np.percentile(y_pred_proba, 100 - pct)
            top_customers = y_pred_proba >= threshold
            
            if top_customers.sum() > 0:
                customers_targeted = top_customers.sum()
                actual_repurchase_rate = y_true[top_customers].mean()
                lift = actual_repurchase_rate / baseline_rate
                precision = actual_repurchase_rate
                
                # Calculate business metrics
                true_positives = (y_true[top_customers] == 1).sum()
                campaign_efficiency = true_positives / customers_targeted
                
                strategies[f'top_{pct}%'] = {
                    'threshold': float(threshold),
                    'customers_targeted': int(customers_targeted),
                    'percentage_of_base': float(customers_targeted / total_customers * 100),
                    'repurchase_rate': float(actual_repurchase_rate),
                    'lift_over_baseline': float(lift),
                    'precision': float(precision),
                    'true_positives_captured': int(true_positives),
                    'campaign_efficiency': float(campaign_efficiency)
                }
        
        # Display business insights
        print("✅ TARGETING STRATEGY ANALYSIS:")
        print("-" * 80)
        print(f"{'Strategy':<12} {'Customers':<10} {'Rate':<8} {'Lift':<6} {'Efficiency':<10}")
        print("-" * 80)
        
        for strategy, metrics in strategies.items():
            print(f"{strategy:<12} {metrics['customers_targeted']:<10,} "
                  f"{metrics['repurchase_rate']:<7.1%} {metrics['lift_over_baseline']:<5.1f}x "
                  f"{metrics['campaign_efficiency']:<9.1%}")
        
        self.results['targeting_strategies'] = strategies
        return strategies
    
    def create_probability_distribution_analysis(self, y_pred_proba):
        """Analyze distribution of predicted probabilities"""
        print("=== PROBABILITY DISTRIBUTION ANALYSIS ===")
        
        plt.figure(figsize=(15, 5))
        
        # Histogram
        plt.subplot(1, 3, 1)
        plt.hist(y_pred_proba, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title('Distribution of Predicted Probabilities')
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(1, 3, 2)
        plt.boxplot(y_pred_proba, vert=True)
        plt.ylabel('Predicted Probability')
        plt.title('Box Plot of Probabilities')
        plt.grid(True, alpha=0.3)
        
        # Cumulative distribution
        plt.subplot(1, 3, 3)
        sorted_probs = np.sort(y_pred_proba)
        p = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)
        plt.plot(sorted_probs, p, linewidth=2)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('models/probability_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate distribution statistics
        prob_stats = {
            'mean': float(np.mean(y_pred_proba)),
            'median': float(np.median(y_pred_proba)),
            'std': float(np.std(y_pred_proba)),
            'min': float(np.min(y_pred_proba)),
            'max': float(np.max(y_pred_proba)),
            'q25': float(np.percentile(y_pred_proba, 25)),
            'q75': float(np.percentile(y_pred_proba, 75)),
            'iqr': float(np.percentile(y_pred_proba, 75) - np.percentile(y_pred_proba, 25))
        }
        
        print("✅ Probability Distribution Statistics:")
        for stat, value in prob_stats.items():
            print(f"   {stat:10}: {value:.4f}")
        
        self.results['probability_stats'] = prob_stats
        return prob_stats
    
    def save_comprehensive_report(self):
        """Save comprehensive evaluation report"""
        print("=== SAVING COMPREHENSIVE REPORT ===")
        
        report = {
            'model_info': {
                'model_name': self.model_name,
                'model_type': type(self.model).__name__,
                'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'performance_metrics': self.results.get('metrics', {}),
            'probability_statistics': self.results.get('probability_stats', {}),
            'business_insights': self.results.get('targeting_strategies', {}),
            'top_10_features': self.results.get('feature_importance', [])[:10] if self.results.get('feature_importance') else []
        }
        
        with open('models/comprehensive_evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("✅ Comprehensive report saved: models/comprehensive_evaluation_report.json")
        return report
    
    def run_comprehensive_evaluation(self):
        """Execute complete comprehensive evaluation"""
        print("🔍 STARTING COMPREHENSIVE MODEL EVALUATION")
        print("=" * 70)
        
        # Step 1: Load model and data
        X_test, y_test = self.load_model_and_data()
        
        # Step 2: Generate predictions
        y_pred, y_pred_proba = self.generate_predictions(X_test, y_test)
        
        # Step 3: Calculate performance metrics
        metrics = self.calculate_performance_metrics(y_test, y_pred, y_pred_proba)
        
        # Step 4: Create visualizations
        self.create_confusion_matrix_plot(y_test, y_pred)
        optimal_threshold = self.create_roc_curve_plot(y_test, y_pred_proba)
        self.create_precision_recall_curve(y_test, y_pred_proba)
        
        # Step 5: Analyze features
        feature_importance = self.analyze_feature_importance(X_test)
        
        # Step 6: Business impact analysis
        targeting_strategies = self.business_impact_analysis(y_test, y_pred_proba)
        
        # Step 7: Probability distribution analysis
        prob_stats = self.create_probability_distribution_analysis(y_pred_proba)
        
        # Step 8: Save comprehensive report
        report = self.save_comprehensive_report()
        
        print(f"\n🎉 COMPREHENSIVE EVALUATION COMPLETED!")
        print("=" * 70)
        print("📊 KEY FINDINGS:")
        print(f"🎯 Model Performance: AUC = {metrics['auc_roc']:.4f}")
        print(f"🎯 Optimal Business Strategy: Target top 20% customers")
        print(f"🎯 Expected ROI: {targeting_strategies.get('top_20%', {}).get('lift_over_baseline', 0):.1f}x lift over baseline")
        print("\n📁 Generated Files:")
        print("   📊 confusion_matrix_detailed.png")
        print("   📊 roc_curve_detailed.png")
        print("   📊 precision_recall_curve.png")
        print("   📊 feature_importance_detailed.png")
        print("   📊 probability_distribution_analysis.png")
        print("   📄 comprehensive_evaluation_report.json")
        
        return True

if __name__ == "__main__":
    try:
        evaluator = ComprehensiveModelEvaluator()
        success = evaluator.run_comprehensive_evaluation()
        
        if success:
            print("\n✅ COMPREHENSIVE EVALUATION SUCCESSFUL!")
        else:
            print("\n❌ EVALUATION FAILED!")
            
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
