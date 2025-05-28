import os
import sys
import argparse
import logging
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_processor import DataProcessor
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.predictor import ChurnPredictor
from src.visualizer import ChurnVisualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_data(num_customers=7043):
    logger.info(f"Creating sample data with {num_customers} customers...")
    
    np.random.seed(42)
    
    data = {
        'customerID': [f'CUST_{i:04d}' for i in range(1, num_customers + 1)],
        'gender': np.random.choice(['Male', 'Female'], num_customers),
        'SeniorCitizen': np.random.choice([0, 1], num_customers, p=[0.84, 0.16]),
        'Partner': np.random.choice(['Yes', 'No'], num_customers, p=[0.52, 0.48]),
        'Dependents': np.random.choice(['Yes', 'No'], num_customers, p=[0.30, 0.70]),
        'tenure': np.random.randint(1, 73, num_customers),
        'PhoneService': np.random.choice(['Yes', 'No'], num_customers, p=[0.91, 0.09]),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], num_customers, p=[0.42, 0.49, 0.09]),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], num_customers, p=[0.34, 0.44, 0.22]),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], num_customers, p=[0.29, 0.49, 0.22]),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], num_customers, p=[0.34, 0.44, 0.22]),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], num_customers, p=[0.34, 0.44, 0.22]),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], num_customers, p=[0.29, 0.49, 0.22]),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], num_customers, p=[0.38, 0.40, 0.22]),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], num_customers, p=[0.38, 0.40, 0.22]),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], num_customers, p=[0.55, 0.21, 0.24]),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], num_customers, p=[0.59, 0.41]),
        'PaymentMethod': np.random.choice([
            'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
        ], num_customers, p=[0.34, 0.23, 0.22, 0.21]),
        'MonthlyCharges': np.random.normal(64.76, 30.09, num_customers),
        'TotalCharges': np.random.normal(2283.30, 2266.77, num_customers)
    }
    
    df = pd.DataFrame(data)
    
    df['MonthlyCharges'] = np.clip(df['MonthlyCharges'], 18.25, 118.75)
    df['TotalCharges'] = np.clip(df['TotalCharges'], 18.80, 8684.80)
    df['TotalCharges'] = np.where(df['tenure'] == 0, 0, df['TotalCharges'])
    
    churn_prob = 0.1 + 0.3 * (df['Contract'] == 'Month-to-month').astype(int)
    churn_prob += 0.15 * (df['PaymentMethod'] == 'Electronic check').astype(int)
    churn_prob += 0.1 * (df['InternetService'] == 'Fiber optic').astype(int)
    churn_prob -= 0.2 * (df['tenure'] > 24).astype(int)
    churn_prob -= 0.1 * (df['Partner'] == 'Yes').astype(int)
    
    churn_prob = np.clip(churn_prob, 0.05, 0.8)
    df['Churn'] = np.random.binomial(1, churn_prob, num_customers)
    df['Churn'] = df['Churn'].map({0: 'No', 1: 'Yes'})
    
    return df


def setup_directories():
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'reports',
        'notebooks'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def run_data_pipeline(create_sample=False):
    logger.info("Starting data processing pipeline...")
    
    setup_directories()
    
    data_file = "data/raw/telco_customer_churn.csv"
    
    if create_sample or not os.path.exists(data_file):
        sample_df = create_sample_data()
        sample_df.to_csv(data_file, index=False)
        logger.info(f"Sample data saved to {data_file}")
    
    processor = DataProcessor()
    df = processor.load_data(data_file)
    
    df_clean = processor.clean_data(df)
    
    engineer = FeatureEngineer()
    df_engineered = engineer.engineer_features(df_clean)
    
    df_encoded = processor.encode_categorical_features(df_engineered, fit=True)
    df_scaled = processor.scale_features(df_encoded, fit=True)
    
    X, y = processor.prepare_features_target(df_scaled)
    X_train, X_test, y_train, y_test = processor.split_data(X, y)
    
    processed_data_file = "data/processed/cleaned_churn_data.csv"
    df_scaled.to_csv(processed_data_file, index=False)
    
    logger.info("Data processing completed successfully")
    
    return X_train, X_test, y_train, y_test, processor


def run_model_training(X_train, y_train, X_test, y_test, models_to_train=None):
    logger.info("Starting model training pipeline...")
    
    trainer = ModelTrainer()
    
    if models_to_train is None:
        models_to_train = ['random_forest', 'xgboost', 'lightgbm']
    
    trainer.train_all_models(X_train, y_train, models_to_train)
    
    evaluations = trainer.evaluate_all_models(X_test, y_test)
    
    trainer.save_models()
    
    logger.info("Model training completed successfully")
    
    return trainer, evaluations


def run_visualization_pipeline(df, evaluations=None):
    logger.info("Starting visualization pipeline...")
    
    visualizer = ChurnVisualizer()
    
    saved_plots = visualizer.save_all_plots(df, target_col='Churn')
    
    if evaluations:
        for model_name, eval_data in evaluations.items():
            logger.info(f"{model_name}: F1={eval_data['f1_score']:.4f}, "
                       f"ROC-AUC={eval_data['roc_auc_score']:.4f}")
    
    logger.info("Visualization pipeline completed successfully")
    
    return saved_plots


def generate_business_report(evaluations, output_file="reports/model_performance_summary.txt"):
    with open(output_file, 'w') as f:
        f.write("TELECOM CHURN PREDICTION - MODEL PERFORMANCE SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if evaluations:
            f.write("MODEL PERFORMANCE METRICS:\n")
            f.write("-" * 30 + "\n")
            
            best_model = max(evaluations.keys(), 
                           key=lambda x: evaluations[x]['f1_score'])
            
            for model_name, metrics in evaluations.items():
                f.write(f"\n{model_name.upper()}:\n")
                f.write(f"  F1-Score: {metrics['f1_score']:.4f}\n")
                f.write(f"  ROC-AUC: {metrics['roc_auc_score']:.4f}\n")
                f.write(f"  Precision: {metrics['classification_report']['weighted avg']['precision']:.4f}\n")
                f.write(f"  Recall: {metrics['classification_report']['weighted avg']['recall']:.4f}\n")
            
            f.write(f"\nBEST PERFORMING MODEL: {best_model.upper()}\n")
            f.write(f"Best F1-Score: {evaluations[best_model]['f1_score']:.4f}\n")
        
        f.write("\nBUSINESS RECOMMENDATIONS:\n")
        f.write("-" * 30 + "\n")
        f.write("1. Focus retention efforts on month-to-month customers\n")
        f.write("2. Improve fiber optic service quality\n")
        f.write("3. Encourage automatic payment methods\n")
        f.write("4. Implement early intervention for new customers\n")
        f.write("5. Develop loyalty programs for long-term customers\n")
    
    logger.info(f"Business report generated: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Telecom Churn Prediction System')
    parser.add_argument('--create-sample', action='store_true',
                       help='Create sample data instead of using existing dataset')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip model training phase')
    parser.add_argument('--skip-viz', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--models', nargs='+', 
                       choices=['random_forest', 'xgboost', 'lightgbm', 'logistic_regression', 'svm'],
                       help='Specify which models to train')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("TELECOM CUSTOMER CHURN PREDICTION SYSTEM")
    logger.info("=" * 60)
    logger.info(f"Execution started at: {datetime.now()}")
    
    try:
        X_train, X_test, y_train, y_test, processor = run_data_pipeline(args.create_sample)
        
        evaluations = None
        if not args.skip_training:
            trainer, evaluations = run_model_training(X_train, y_train, X_test, y_test, args.models)
            generate_business_report(evaluations)
        else:
            logger.info("Skipping model training...")
        
        if not args.skip_viz:
            df = pd.read_csv("data/raw/telco_customer_churn.csv")
            run_visualization_pipeline(df, evaluations)
        else:
            logger.info("Skipping visualization generation...")
        
        logger.info("=" * 60)
        logger.info("EXECUTION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Review the generated reports in the 'reports/' directory")
        logger.info("2. Launch the Streamlit dashboard: streamlit run app/streamlit_dashboard.py")
        logger.info("3. Explore the Jupyter notebooks in the 'notebooks/' directory")
        logger.info("4. Check model artifacts in the 'models/' directory")
        
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()