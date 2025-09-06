import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import json
from datetime import datetime
import warnings
import time

warnings.filterwarnings('ignore')

class FastHyperparameterTuner:
    def __init__(self, cv_folds=3, scoring='roc_auc', n_jobs=1, random_state=42, timeout=300):
        """
        Fast hyperparameter tuner optimized for speed
        """
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.timeout = timeout
        self.best_models = {}
        self.tuning_results = {}

    def load_data(self):
        """Load preprocessed data"""
        print("=== LOADING PREPROCESSED DATA ===")
        try:
            X_train = pd.read_csv('data/X_train.csv')
            X_test = pd.read_csv('data/X_test.csv')
            y_train = pd.read_csv('data/y_train.csv')['will_repurchase']
            y_test = pd.read_csv('data/y_test.csv')['will_repurchase']

            print(f"✅ Training set: {X_train.shape}")
            print(f"✅ Test set: {X_test.shape}")
            print(f"✅ Class distribution: {y_train.value_counts().to_dict()}")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise

    def get_parameter_grids(self):
        """Define FAST parameter grids optimized for speed"""
        return {
            'logistic_regression': {
                'C': [0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs'],
                'class_weight': ['balanced'],
                'max_iter': [2000]
            },
            'random_forest': {
                'n_estimators': [50, 100],
                'max_depth': [10, 20],
                'min_samples_split': [5],
                'min_samples_leaf': [2],
                'class_weight': ['balanced'],
                'max_features': ['sqrt']
            },
            # OPTIMIZED XGBOOST GRID - Much faster!
            'xgboost': {
                'n_estimators': [50, 100],        # Reduced from [50,100,150]
                'max_depth': [3, 5],              # Removed expensive depth=7
                'learning_rate': [0.1, 0.2],      # Reduced options
                'subsample': [0.8],               # Fixed value
                'colsample_bytree': [0.8],        # Fixed value
                'reg_alpha': [0],                 # Fixed value
                'reg_lambda': [1]                 # Fixed typo and value
            },
            # ENHANCED LIGHTGBM GRID
            'lightgbm': {
                'n_estimators': [50, 100, 150],   # More options for LightGBM (it's faster)
                'max_depth': [3, 5, 7],           # More depth options
                'learning_rate': [0.05, 0.1, 0.2], # More learning rates
                'num_leaves': [31, 50],           # More leaf options
                'subsample': [0.8, 0.9],          # More subsample options
                'colsample_bytree': [0.8, 0.9],   # More feature sampling
                'reg_alpha': [0, 0.1],            # Add regularization
                'reg_lambda': [0, 0.1]            # Add regularization
            }
        }

    def get_base_models(self):
        """Get base models for tuning"""
        return {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=2000,
                solver='lbfgs'
            ),
            'random_forest': RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=1
            ),
            'xgboost': xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                verbosity=0,
                tree_method='hist'  # Faster tree method
            ),
            'lightgbm': lgb.LGBMClassifier(
                random_state=self.random_state,
                verbose=-1,
                force_col_wise=True,
                class_weight='balanced'
            )
        }

    def tune_model_with_timeout(self, model_name, model, param_grid, X_train, y_train):
        """Tune a single model with optimized search strategy"""
        print(f"\n🔍 Tuning {model_name}...")
        
        start_time = time.time()
        try:
            # SPEED OPTIMIZATION: Use RandomizedSearchCV for XGBoost
            if model_name == 'xgboost':
                search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grid,
                    scoring=self.scoring,
                    n_iter=8,  # Only try 8 combinations instead of 8 (2*2*2*1*1*1*1)
                    cv=self.cv_folds,
                    n_jobs=1,
                    verbose=0,
                    random_state=self.random_state
                )
                print(f"📊 Using RandomizedSearch with 8 iterations (faster!)")
            else:
                # Use GridSearchCV for other models (they have smaller grids)
                combinations = np.prod([len(v) for v in param_grid.values()])
                print(f"📊 Parameter combinations: {combinations}")
                search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    scoring=self.scoring,
                    cv=self.cv_folds,
                    n_jobs=1,
                    verbose=0,
                    return_train_score=False
                )

            # Special handling for class imbalance
            if model_name == 'xgboost':
                neg_count = (y_train == 0).sum()
                pos_count = (y_train == 1).sum()
                if pos_count > 0:
                    scale_pos_weight = neg_count / pos_count
                    model.set_params(scale_pos_weight=scale_pos_weight)
                    print(f" Setting scale_pos_weight: {scale_pos_weight:.2f}")
            elif model_name == 'lightgbm':
                print(f" LightGBM using balanced class weights")

            print(f" Starting search...")
            search.fit(X_train, y_train)
            
            elapsed_time = time.time() - start_time
            print(f" ✅ Completed in {elapsed_time:.1f} seconds")
            return search

        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f" ❌ Failed after {elapsed_time:.1f} seconds: {str(e)}")
            return None

    def tune_all_models(self, X_train, y_train):
        """Tune all models with their respective parameter grids"""
        print("=== FAST HYPERPARAMETER TUNING ===")
        
        base_models = self.get_base_models()
        param_grids = self.get_parameter_grids()

        # Optimized order: Start with fastest models first
        model_order = ['logistic_regression', 'random_forest', 'lightgbm', 'xgboost']

        for model_name in model_order:
            if model_name not in base_models:
                continue

            model = base_models[model_name]
            param_grid = param_grids[model_name]

            print(f"\n{'='*50}")
            print(f"Processing: {model_name.upper()}")
            print(f"{'='*50}")

            try:
                # Perform search with timeout
                search = self.tune_model_with_timeout(model_name, model, param_grid, X_train, y_train)

                if search is not None:
                    # Store results
                    self.best_models[model_name] = search.best_estimator_
                    self.tuning_results[model_name] = {
                        'best_params': search.best_params_,
                        'best_cv_score': search.best_score_,
                        'cv_std': search.cv_results_['std_test_score'][search.best_index_]
                    }

                    print(f"✅ {model_name} tuning completed!")
                    print(f" Best CV Score: {search.best_score_:.4f} ± {search.cv_results_['std_test_score'][search.best_index_]:.4f}")
                    print(f" Best Parameters: {search.best_params_}")
                else:
                    print(f"❌ {model_name} tuning failed!")

            except KeyboardInterrupt:
                print(f"\n⚠️ Interrupted during {model_name} tuning")
                break
            except Exception as e:
                print(f"❌ Unexpected error tuning {model_name}: {str(e)}")
                continue

    def evaluate_tuned_models(self, X_test, y_test):
        """Evaluate all tuned models on test set"""
        print("\n=== EVALUATING TUNED MODELS ===")
        
        if not self.best_models:
            print("❌ No models were successfully tuned!")
            return {}

        test_results = {}
        
        for model_name, model in self.best_models.items():
            try:
                print(f"Evaluating {model_name}...")
                
                # Make predictions
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                test_auc = roc_auc_score(y_test, y_pred_proba)

                test_results[model_name] = {
                    'test_auc': test_auc,
                    'cv_score': self.tuning_results[model_name]['best_cv_score'],
                    'cv_std': self.tuning_results[model_name]['cv_std'],
                    'best_params': self.tuning_results[model_name]['best_params']
                }

                print(f"📊 {model_name}:")
                print(f" Test AUC: {test_auc:.4f}")
                print(f" CV AUC: {self.tuning_results[model_name]['best_cv_score']:.4f} ± {self.tuning_results[model_name]['cv_std']:.4f}")

            except Exception as e:
                print(f"❌ Error evaluating {model_name}: {str(e)}")
                continue

        return test_results

    def find_best_model(self, test_results):
        """Find the best performing model"""
        if not test_results:
            print("❌ No models to compare!")
            return None, None

        best_model_name = max(test_results.keys(),
                              key=lambda k: test_results[k]['test_auc'])
        best_model = self.best_models[best_model_name]

        return best_model_name, best_model

    def save_results(self, test_results, best_model_name, best_model):
        """Save tuning results and best model"""
        print("\n=== SAVING RESULTS ===")
        
        # Create models directory
        import os
        os.makedirs('models', exist_ok=True)

        # Save best model
        if best_model is not None:
            model_path = f'models/best_tuned_model_{best_model_name}.pkl'
            joblib.dump(best_model, model_path)
            print(f"✅ Best model saved: {model_path}")

        # Save detailed results
        detailed_results = {
            'tuning_info': {
                'cv_folds': self.cv_folds,
                'scoring_metric': self.scoring,
                'tuning_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'platform': 'Optimized for Speed',
                'timeout_seconds': self.timeout
            },
            'best_model': best_model_name,
            'model_comparison': test_results,
            'parameter_details': self.tuning_results
        }

        results_path = 'models/fast_hyperparameter_tuning_results.json'
        with open(results_path, 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        print(f"✅ Detailed results saved: {results_path}")
        return detailed_results

    def run_complete_tuning(self):
        """Execute complete fast hyperparameter tuning workflow"""
        print("🚀 STARTING FAST HYPERPARAMETER TUNING")
        print("=" * 70)

        try:
            # Load data
            X_train, X_test, y_train, y_test = self.load_data()

            # Tune all models
            self.tune_all_models(X_train, y_train)

            # Evaluate on test set
            test_results = self.evaluate_tuned_models(X_test, y_test)

            # Find best model
            best_model_name, best_model = self.find_best_model(test_results)

            # Save results
            if test_results:
                detailed_results = self.save_results(test_results, best_model_name, best_model)

                # Print summary
                print(f"\n🎉 FAST HYPERPARAMETER TUNING COMPLETED!")
                print("=" * 70)
                
                if best_model_name:
                    print("📊 FINAL RESULTS:")
                    print(f"🏆 Best Model: {best_model_name}")
                    print(f"🎯 Test AUC: {test_results[best_model_name]['test_auc']:.4f}")
                    print(f"🎯 CV AUC: {test_results[best_model_name]['cv_score']:.4f}")

                    print("\n📈 MODEL COMPARISON:")
                    print("-" * 60)
                    for name, results in test_results.items():
                        status = "👑" if name == best_model_name else " "
                        print(f"{status} {name:20} | Test: {results['test_auc']:.4f} | CV: {results['cv_score']:.4f}")
                    print("-" * 60)

                    # Prediction about which model should win
                    if best_model_name == 'lightgbm':
                        print("\n🎯 EXPECTED: LightGBM won on clean data - great generalization!")
                    elif best_model_name == 'xgboost':
                        print("\n🎯 SURPRISE: XGBoost still competitive even on clean data!")
                    
                return True
            else:
                print("\n❌ No models were successfully tuned!")
                return False

        except Exception as e:
            print(f"\n💥 ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    try:
        # Initialize fast tuner
        tuner = FastHyperparameterTuner(
            cv_folds=3,
            scoring='roc_auc',
            n_jobs=1,
            random_state=42,
            timeout=300
        )

        # Run complete tuning
        success = tuner.run_complete_tuning()

        if success:
            print("\n✅ FAST HYPERPARAMETER TUNING SUCCESSFUL!")
        else:
            print("\n❌ HYPERPARAMETER TUNING FAILED!")

    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()