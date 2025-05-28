import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = 'Churn'
        
    def load_data(self, file_path):
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def clean_data(self, df):
        df_clean = df.copy()
        
        if 'TotalCharges' in df_clean.columns:
            df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
            df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(0)
        
        if 'customerID' in df_clean.columns:
            df_clean = df_clean.drop('customerID', axis=1)
        
        missing_counts = df_clean.isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(f"Missing values found: {missing_counts[missing_counts > 0]}")
            
            categorical_cols = df_clean.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df_clean[col].isnull().sum() > 0:
                    mode_value = df_clean[col].mode()[0]
                    df_clean[col] = df_clean[col].fillna(mode_value)
            
            numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if df_clean[col].isnull().sum() > 0:
                    median_value = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(median_value)
        
        logger.info(f"Data cleaned. Final shape: {df_clean.shape}")
        return df_clean
    
    def encode_categorical_features(self, df, fit=True):
        df_encoded = df.copy()
        
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
        if self.target_column in categorical_cols:
            categorical_cols.remove(self.target_column)
        
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col])
            else:
                if col in self.label_encoders:
                    unique_values = set(df_encoded[col].unique())
                    known_values = set(self.label_encoders[col].classes_)
                    unseen_values = unique_values - known_values
                    
                    if unseen_values:
                        logger.warning(f"Unseen values in {col}: {unseen_values}")
                        most_frequent = self.label_encoders[col].classes_[0]
                        df_encoded[col] = df_encoded[col].replace(list(unseen_values), most_frequent)
                    
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col])
                else:
                    logger.warning(f"No encoder found for column {col}")
        
        if self.target_column in df_encoded.columns:
            if fit:
                self.label_encoders[self.target_column] = LabelEncoder()
                df_encoded[self.target_column] = self.label_encoders[self.target_column].fit_transform(df_encoded[self.target_column])
            else:
                df_encoded[self.target_column] = self.label_encoders[self.target_column].transform(df_encoded[self.target_column])
        
        logger.info(f"Categorical encoding completed. Encoded columns: {categorical_cols}")
        return df_encoded
    
    def scale_features(self, df, fit=True):
        df_scaled = df.copy()
        
        numerical_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_column in numerical_cols:
            numerical_cols.remove(self.target_column)
        
        if numerical_cols:
            if fit:
                df_scaled[numerical_cols] = self.scaler.fit_transform(df_scaled[numerical_cols])
            else:
                df_scaled[numerical_cols] = self.scaler.transform(df_scaled[numerical_cols])
            
            logger.info(f"Feature scaling completed. Scaled columns: {numerical_cols}")
        
        return df_scaled
    
    def prepare_features_target(self, df):
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset")
        
        X = df.drop(self.target_column, axis=1)
        y = df[self.target_column]
        
        self.feature_columns = X.columns.tolist()
        
        logger.info(f"Features and target prepared. Features: {X.shape[1]}, Samples: {X.shape[0]}")
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42, stratify=True):
        stratify_param = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=stratify_param
        )
        
        logger.info(f"Data split completed. Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        return X_train, X_test, y_train, y_test
    
    def process_pipeline(self, file_path, test_size=0.2, random_state=42):
        df = self.load_data(file_path)
        df_clean = self.clean_data(df)
        df_encoded = self.encode_categorical_features(df_clean, fit=True)
        df_scaled = self.scale_features(df_encoded, fit=True)
        X, y = self.prepare_features_target(df_scaled)
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size, random_state)
        
        logger.info("Data processing pipeline completed successfully")
        return X_train, X_test, y_train, y_test
    
    def get_feature_info(self):
        return {
            'feature_columns': self.feature_columns,
            'label_encoders': list(self.label_encoders.keys()),
            'scaler_fitted': hasattr(self.scaler, 'mean_')
        }


def main():
    processor = DataProcessor()
    file_path = "data/raw/telco_customer_churn.csv"
    
    try:
        X_train, X_test, y_train, y_test = processor.process_pipeline(file_path)
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Feature columns: {len(processor.feature_columns)}")
        
        feature_info = processor.get_feature_info()
        print(f"Feature info: {feature_info}")
        
    except FileNotFoundError:
        print(f"Data file not found at {file_path}")
        print("Please download the Telco Customer Churn dataset and place it in data/raw/")


if __name__ == "__main__":
    main()