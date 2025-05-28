"""
Unit tests for the DataProcessor class
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_processor import DataProcessor


class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor class"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.processor = DataProcessor()
        
        # Create sample data for testing
        self.sample_data = {
            'customerID': ['CUST_001', 'CUST_002', 'CUST_003', 'CUST_004'],
            'gender': ['Male', 'Female', 'Male', 'Female'],
            'SeniorCitizen': [0, 1, 0, 1],
            'Partner': ['Yes', 'No', 'Yes', 'No'],
            'Dependents': ['No', 'Yes', 'No', 'Yes'],
            'tenure': [12, 24, 6, 36],
            'PhoneService': ['Yes', 'Yes', 'No', 'Yes'],
            'InternetService': ['DSL', 'Fiber optic', 'No', 'DSL'],
            'Contract': ['Month-to-month', 'One year', 'Month-to-month', 'Two year'],
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Electronic check', 'Credit card (automatic)'],
            'MonthlyCharges': [29.85, 56.95, 20.25, 78.70],
            'TotalCharges': ['29.85', '1889.5', '20.25', '3046.05'],  # Some as strings to test conversion
            'Churn': ['No', 'Yes', 'No', 'Yes']
        }
        
        self.df = pd.DataFrame(self.sample_data)
    
    def test_init(self):
        """Test DataProcessor initialization"""
        processor = DataProcessor()
        self.assertIsNotNone(processor)
        self.assertEqual(processor.target_column, 'Churn')
        self.assertIsInstance(processor.label_encoders, dict)
    
    def test_clean_data(self):
        """Test data cleaning functionality"""
        # Add some missing values to test
        df_with_missing = self.df.copy()
        df_with_missing.loc[0, 'TotalCharges'] = ''  # Empty string
        df_with_missing.loc[1, 'Partner'] = None     # Missing value
        
        cleaned_df = self.processor.clean_data(df_with_missing)
        
        # Check that customerID is removed
        self.assertNotIn('customerID', cleaned_df.columns)
        
        # Check that TotalCharges is converted to numeric
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_df['TotalCharges']))
        
        # Check that missing values are handled
        self.assertEqual(cleaned_df.isnull().sum().sum(), 0)
    
    def test_encode_categorical_features(self):
        """Test categorical feature encoding"""
        df_clean = self.processor.clean_data(self.df)
        
        # Test fitting encoders
        df_encoded = self.processor.encode_categorical_features(df_clean, fit=True)
        
        # Check that categorical columns are encoded
        categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
        if 'Churn' in categorical_cols:
            categorical_cols.remove('Churn')
        
        for col in categorical_cols:
            self.assertTrue(pd.api.types.is_numeric_dtype(df_encoded[col]))
        
        # Check that encoders are stored
        self.assertGreater(len(self.processor.label_encoders), 0)
        
        # Test transforming without fitting
        df_encoded_2 = self.processor.encode_categorical_features(df_clean, fit=False)
        pd.testing.assert_frame_equal(df_encoded, df_encoded_2)
    
    def test_scale_features(self):
        """Test feature scaling"""
        df_clean = self.processor.clean_data(self.df)
        df_encoded = self.processor.encode_categorical_features(df_clean, fit=True)
        
        # Test fitting scaler
        df_scaled = self.processor.scale_features(df_encoded, fit=True)
        
        # Check that numerical features are scaled
        numerical_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
        if 'Churn' in numerical_cols:
            numerical_cols.remove('Churn')
        
        for col in numerical_cols:
            # Scaled features should have mean close to 0 and std close to 1
            self.assertAlmostEqual(df_scaled[col].mean(), 0, places=10)
            self.assertAlmostEqual(df_scaled[col].std(), 1, places=10)
    
    def test_prepare_features_target(self):
        """Test feature and target separation"""
        df_processed = self.df.copy()
        df_processed = self.processor.clean_data(df_processed)
        df_processed = self.processor.encode_categorical_features(df_processed, fit=True)
        
        X, y = self.processor.prepare_features_target(df_processed)
        
        # Check that target is separated correctly
        self.assertNotIn('Churn', X.columns)
        self.assertEqual(len(y), len(df_processed))
        self.assertEqual(y.name, 'Churn')
        
        # Check that feature columns are stored
        self.assertIsNotNone(self.processor.feature_columns)
        self.assertEqual(len(self.processor.feature_columns), X.shape[1])
    
    def test_split_data(self):
        """Test data splitting"""
        df_processed = self.df.copy()
        df_processed = self.processor.clean_data(df_processed)
        df_processed = self.processor.encode_categorical_features(df_processed, fit=True)
        X, y = self.processor.prepare_features_target(df_processed)
        
        X_train, X_test, y_train, y_test = self.processor.split_data(X, y, test_size=0.5, random_state=42)
        
        # Check split proportions
        total_samples = len(X)
        self.assertEqual(len(X_train), total_samples // 2)
        self.assertEqual(len(X_test), total_samples - total_samples // 2)
        self.assertEqual(len(y_train), len(X_train))
        self.assertEqual(len(y_test), len(X_test))
    
    def test_load_data_file_not_found(self):
        """Test loading data with non-existent file"""
        with self.assertRaises(Exception):
            self.processor.load_data('non_existent_file.csv')
    
    def test_load_data_success(self):
        """Test successful data loading"""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.df.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            loaded_df = self.processor.load_data(temp_file)
            self.assertEqual(loaded_df.shape, self.df.shape)
            self.assertListEqual(list(loaded_df.columns), list(self.df.columns))
        finally:
            os.unlink(temp_file)
    
    def test_get_feature_info(self):
        """Test feature information retrieval"""
        # Process some data first
        df_processed = self.df.copy()
        df_processed = self.processor.clean_data(df_processed)
        df_processed = self.processor.encode_categorical_features(df_processed, fit=True)
        df_processed = self.processor.scale_features(df_processed, fit=True)
        X, y = self.processor.prepare_features_target(df_processed)
        
        feature_info = self.processor.get_feature_info()
        
        self.assertIn('feature_columns', feature_info)
        self.assertIn('label_encoders', feature_info)
        self.assertIn('scaler_fitted', feature_info)
        self.assertTrue(feature_info['scaler_fitted'])
        self.assertGreater(len(feature_info['label_encoders']), 0)
    
    def test_process_pipeline(self):
        """Test complete processing pipeline"""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.df.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            X_train, X_test, y_train, y_test = self.processor.process_pipeline(
                temp_file, test_size=0.5, random_state=42
            )
            
            # Check that all outputs are correct
            self.assertIsInstance(X_train, pd.DataFrame)
            self.assertIsInstance(X_test, pd.DataFrame)
            self.assertIsInstance(y_train, pd.Series)
            self.assertIsInstance(y_test, pd.Series)
            
            # Check shapes
            total_samples = len(self.df)
            self.assertEqual(len(X_train) + len(X_test), total_samples)
            self.assertEqual(len(y_train) + len(y_test), total_samples)
            
            # Check that features don't contain target
            self.assertNotIn('Churn', X_train.columns)
            self.assertNotIn('Churn', X_test.columns)
            
        finally:
            os.unlink(temp_file)


class TestDataProcessorEdgeCases(unittest.TestCase):
    """Test edge cases for DataProcessor"""
    
    def setUp(self):
        self.processor = DataProcessor()
    
    def test_empty_dataframe(self):
        """Test processing empty dataframe"""
        empty_df = pd.DataFrame()
        
        with self.assertRaises(Exception):
            self.processor.clean_data(empty_df)
    
    def test_missing_target_column(self):
        """Test processing dataframe without target column"""
        df_no_target = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': ['A', 'B', 'C']
        })
        
        with self.assertRaises(ValueError):
            self.processor.prepare_features_target(df_no_target)
    
    def test_single_row_dataframe(self):
        """Test processing dataframe with single row"""
        single_row_df = pd.DataFrame({
            'feature1': [1],
            'feature2': ['A'],
            'Churn': ['Yes']
        })
        
        cleaned_df = self.processor.clean_data(single_row_df)
        self.assertEqual(len(cleaned_df), 1)


if __name__ == '__main__':
    unittest.main()