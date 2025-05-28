import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
import joblib
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.model_results = {}
        
    def get_model_configs(self):
        return {
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'classifier__n_estimators': [100, 200],
                    'classifier__max_depth': [10, 20, None],
                    'classifier__min_samples_split': [2, 5],
                    'classifier__min_samples_leaf': [1, 2]
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'params': {
                    'classifier__n_estimators': [100, 200],
                    'classifier__max_depth': [3, 6, 10],
                    'classifier__learning_rate': [0.01, 0.1, 0.2],
                    'classifier__subsample': [0.8, 1.0]
                }
            },
            'lightgbm': {
                'model': lgb.LGBMClassifier(random_state=42, verbose=-1),
                'params': {
                    'classifier__n_estimators': [100, 200],
                    'classifier__max_depth': [3, 6, 10],
                    'classifier__learning_rate': [0.01, 0.1, 0.2],
                    'classifier__num_leaves': [31, 50, 100]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'classifier__C': [0.1, 1, 10],
                    'classifier__penalty': ['l1', 'l2'],
                    'classifier__solver': ['liblinear']
                }
            },
            'svm': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'classifier__C': [0.1, 1, 10],
                    'classifier__kernel': ['rbf', 'linear'],
                    'classifier__gamma': ['scale', 'auto']
                }
            }
        }
    
    def create_balanced_pipeline(self, model, sampling_strategy='auto'):
        return ImbPipeline([
            ('sampler', SMOTE(random_state=42, sampling_strategy=sampling_strategy)),
            ('classifier', model)
        ])
    
    def train_single_model(self, model_name, X_train, y_train, cv_folds=5):
        logger.info(f"Training {model_name}...")
        
        config = self.get_model_configs()[model_name]
        pipeline = self.create_balanced_pipeline(config['model'])
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            pipeline,
            config['params'],
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        cv_scores = cross_val_score(
            grid_search.best_estimator_,
            X_train, y_train,
            cv=cv,
            scoring='f1'
        )
        
        self.models[model_name] = grid_search.best_estimator_
        
        results = {
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'cv_scores_mean': cv_scores.mean(),
            'cv_scores_std': cv_scores.std(),
            'training_time': datetime.now().isoformat()
        }
        
        self.model_results[model_name] = results
        
        if grid_search.best_score_ > self.best_score:
            self.best_score = grid_search.best_score_
            self.best_model = grid_search.best_estimator_
        
        logger.info(f"{model_name} training completed. CV F1-score: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_
    
    def train_all_models(self, X_train, y_train, models_to_train=None):
        if models_to_train is None:
            models_to_train = list(self.get_model_configs().keys())
        
        logger.info(f"Training {len(models_to_train)} models...")
        
        for model_name in models_to_train:
            try:
                self.train_single_model(model_name, X_train, y_train)
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        logger.info("All models training completed")
        return self.models
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        evaluation = {
            'model_name': model_name,
            'f1_score': f1,
            'roc_auc_score': roc_auc,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        logger.info(f"{model_name} - F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
        return evaluation
    
    def evaluate_all_models(self, X_test, y_test):
        evaluations = {}
        
        for model_name, model in self.models.items():
            evaluations[model_name] = self.evaluate_model(model, X_test, y_test, model_name)
        
        return evaluations
    
    def get_feature_importance(self, model_name=None, top_n=20):
        if model_name is None:
            model = self.best_model
            name = "best_model"
        else:
            model = self.models.get(model_name)
            name = model_name
        
        if model is None:
            logger.warning(f"Model {name} not found")
            return None
        
        classifier = model.named_steps['classifier']
        
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
        elif hasattr(classifier, 'coef_'):
            importances = np.abs(classifier.coef_[0])
        else:
            logger.warning(f"Model {name} doesn't support feature importance")
            return None
        
        feature_importance = pd.DataFrame({
            'feature': range(len(importances)),
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance.head(top_n)
    
    def save_models(self, save_dir="models/"):
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = os.path.join(save_dir, f"{model_name}_model.pkl")
            joblib.dump(model, model_path)
            logger.info(f"Model {model_name} saved to {model_path}")
        
        if self.best_model is not None:
            best_model_path = os.path.join(save_dir, "best_model.pkl")
            joblib.dump(self.best_model, best_model_path)
            logger.info(f"Best model saved to {best_model_path}")
        
        results_path = os.path.join(save_dir, "model_comparison.json")
        with open(results_path, 'w') as f:
            json.dump(self.model_results, f, indent=2)
        logger.info(f"Model results saved to {results_path}")
    
    def load_model(self, model_path):
        try:
            model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
            return None
    
    def get_model_summary(self):
        summary = {
            'total_models_trained': len(self.models),
            'best_model_score': self.best_score,
            'model_results': self.model_results
        }
        
        if self.models:
            best_model_name = max(self.model_results.keys(), 
                                key=lambda x: self.model_results[x]['best_cv_score'])
            summary['best_model_name'] = best_model_name
        
        return summary


def main():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.7, 0.3],
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    trainer = ModelTrainer()
    
    models_to_train = ['random_forest', 'xgboost', 'logistic_regression']
    trainer.train_all_models(X_train, y_train, models_to_train)
    
    evaluations = trainer.evaluate_all_models(X_test, y_test)
    
    print("Model Evaluation Results:")
    for model_name, eval_results in evaluations.items():
        print(f"{model_name}: F1={eval_results['f1_score']:.4f}, "
              f"ROC-AUC={eval_results['roc_auc_score']:.4f}")
    
    summary = trainer.get_model_summary()
    print(f"\nBest model: {summary.get('best_model_name', 'Unknown')}")
    print(f"Best score: {summary['best_model_score']:.4f}")
    
    feature_importance = trainer.get_feature_importance(top_n=10)
    if feature_importance is not None:
        print("\nTop 10 Feature Importances:")
        print(feature_importance)


if __name__ == "__main__":
    main()