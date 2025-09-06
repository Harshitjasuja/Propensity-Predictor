import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve
import matplotlib.pyplot as plt
import joblib
import json
from datetime import datetime

print("🚀 STARTING UNBIASED MODEL TRAINING & EVALUATION")
print("=" * 60)

# Load unbiased data
print("Loading unbiased preprocessed data...")
X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv') 
y_train = pd.read_csv('data/y_train.csv')['will_repurchase']
y_test = pd.read_csv('data/y_test.csv')['will_repurchase']

print(f"✅ Training set: {X_train.shape}")
print(f"✅ Test set: {X_test.shape}")
print(f"✅ Test target distribution: {y_test.value_counts().to_dict()}")

# Initialize models
models = {
    'logistic_regression': LogisticRegression(
        random_state=42, class_weight='balanced', max_iter=1000
    ),
    'random_forest': RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight='balanced'
    ),
    'xgboost': xgb.XGBClassifier(
        random_state=42, eval_metric='logloss'
    ),
    'lightgbm': lgb.LGBMClassifier(
        random_state=42, class_weight='balanced', verbose=-1
    )
}

results = {}

# Train and evaluate each model
print("\n=== TRAINING AND EVALUATING MODELS ===")
for name, model in models.items():
    print(f"\n🔄 Training {name}...")
    
    # Special handling for XGBoost
    if name == 'lightgbm':
        (f"  LightGBM using balanced class weights")
    elif name == 'xgboost':
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count
        model.set_params(scale_pos_weight=scale_pos_weight)
        print(f" XGBoost scale_pos_weight:{scale_pos_weight:.2f}")

    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    results[name] = {
        'model': model,
        'auc_score': auc_score,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"✅ {name} - AUC: {auc_score:.4f}")

# Find best model
best_model_name = max(results.keys(), key=lambda k: results[k]['auc_score'])
best_auc = results[best_model_name]['auc_score']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"🏆 BEST AUC: {best_auc:.4f}")

# Detailed evaluation of best model
print(f"\n📊 DETAILED EVALUATION - {best_model_name.upper()}:")
best_predictions = results[best_model_name]['predictions']
print(classification_report(y_test, best_predictions, 
                          target_names=['No Repurchase', 'Will Repurchase']))

# Business insights
print("\n💰 BUSINESS INSIGHTS:")
best_probabilities = results[best_model_name]['probabilities']
percentiles = [10, 20, 30]
baseline_rate = y_test.mean()

for pct in percentiles:
    threshold = np.percentile(best_probabilities, 100 - pct)
    top_customers = best_probabilities >= threshold
    
    if top_customers.sum() > 0:
        actual_rate = y_test[top_customers].mean()
        lift = actual_rate / baseline_rate
        
        print(f"🎯 Top {pct}% customers:")
        print(f"   Target: {top_customers.sum():,} customers")
        print(f"   Expected repurchase rate: {actual_rate:.1%}")
        print(f"   Lift over baseline: {lift:.2f}x")

# Save models and results
import os
os.makedirs('models', exist_ok=True)

# Save best model
best_model_path = f'models/best_unbiased_model_{best_model_name}.pkl'
joblib.dump(results[best_model_name]['model'], best_model_path)
print(f"\n✅ Best model saved: {best_model_path}")

# Save results summary
summary = {
    'training_type': 'unbiased',
    'best_model_name': best_model_name,
    'best_auc_score': best_auc,
    'all_results': {name: result['auc_score'] for name, result in results.items()},
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset_info': {
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'features': X_train.shape[1],
        'baseline_repurchase_rate': float(baseline_rate)
    }
}

with open('models/unbiased_training_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("✅ Training results saved: models/unbiased_training_results.json")

print(f"\n🎉 MODEL TRAINING COMPLETED!")
print("=" * 60)
print("📊 MODEL COMPARISON:")
print("-" * 40)
for name, result in results.items():
    status = "👑" if name == best_model_name else "  "
    print(f"{status} {name:20} | AUC: {result['auc_score']:.4f}")
print("-" * 40)

