import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    def __init__(self):
        self.polynomial_features = None
        self.feature_names = []
        
    def create_tenure_features(self, df):
        df_new = df.copy()
        
        if 'tenure' in df_new.columns:
            df_new['tenure_group'] = pd.cut(df_new['tenure'],
                                          bins=[0, 12, 24, 48, 72, float('inf')],
                                          labels=[0, 1, 2, 3, 4])
            
            df_new['tenure_years'] = df_new['tenure'] / 12
            df_new['is_new_customer'] = (df_new['tenure'] <= 6).astype(int)
            df_new['is_long_term_customer'] = (df_new['tenure'] > 60).astype(int)
            
            logger.info("Tenure features created successfully")
        
        return df_new
    
    def create_charge_features(self, df):
        df_new = df.copy()
        
        if 'MonthlyCharges' in df_new.columns:
            df_new['monthly_charges_group'] = pd.cut(df_new['MonthlyCharges'],
                                                   bins=[0, 35, 65, 95, float('inf')],
                                                   labels=[0, 1, 2, 3])
        
        if 'TotalCharges' in df_new.columns and 'tenure' in df_new.columns:
            df_new['avg_monthly_charges'] = np.where(
                df_new['tenure'] > 0,
                df_new['TotalCharges'] / df_new['tenure'],
                df_new['MonthlyCharges']
            )
            
            df_new['charge_increase'] = (
                df_new['MonthlyCharges'] > df_new['avg_monthly_charges']
            ).astype(int)
        
        if 'MonthlyCharges' in df_new.columns:
            monthly_charges_75th = df_new['MonthlyCharges'].quantile(0.75)
            df_new['is_high_value_customer'] = (
                df_new['MonthlyCharges'] >= monthly_charges_75th
            ).astype(int)
        
        logger.info("Charge features created successfully")
        return df_new
    
    def create_service_features(self, df):
        df_new = df.copy()
        
        service_cols = [
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
        available_service_cols = [col for col in service_cols if col in df_new.columns]
        if available_service_cols:
            service_df = df_new[available_service_cols].copy()
            for col in available_service_cols:
                if service_df[col].dtype == 'object':
                    service_df[col] = (service_df[col] == 'Yes').astype(int)
            
            df_new['total_services'] = service_df.sum(axis=1)
            df_new['service_adoption_rate'] = df_new['total_services'] / len(available_service_cols)
        
        if 'InternetService' in df_new.columns:
            df_new['has_internet'] = (df_new['InternetService'] != 'No').astype(int)
            df_new['has_fiber_optic'] = (df_new['InternetService'] == 'Fiber optic').astype(int)
        
        premium_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
        available_premium = [col for col in premium_services if col in df_new.columns]
        if available_premium:
            premium_df = df_new[available_premium].copy()
            for col in available_premium:
                if premium_df[col].dtype == 'object':
                    premium_df[col] = (premium_df[col] == 'Yes').astype(int)
            
            df_new['premium_services_count'] = premium_df.sum(axis=1)
            df_new['has_premium_services'] = (df_new['premium_services_count'] > 0).astype(int)
        
        logger.info("Service features created successfully")
        return df_new
    
    def create_contract_features(self, df):
        df_new = df.copy()
        
        if 'Contract' in df_new.columns:
            df_new['is_month_to_month'] = (df_new['Contract'] == 'Month-to-month').astype(int)
            df_new['is_long_term_contract'] = (
                df_new['Contract'].isin(['One year', 'Two year'])
            ).astype(int)
        
        if 'PaymentMethod' in df_new.columns:
            df_new['is_automatic_payment'] = (
                df_new['PaymentMethod'].str.contains('automatic', case=False, na=False)
            ).astype(int)
            
            df_new['is_electronic_check'] = (
                df_new['PaymentMethod'] == 'Electronic check'
            ).astype(int)
        
        if 'PaperlessBilling' in df_new.columns:
            if df_new['PaperlessBilling'].dtype == 'object':
                df_new['paperless_billing'] = (df_new['PaperlessBilling'] == 'Yes').astype(int)
        
        logger.info("Contract features created successfully")
        return df_new
    
    def create_demographic_features(self, df):
        df_new = df.copy()
        
        if 'SeniorCitizen' in df_new.columns:
            df_new['is_senior_citizen'] = df_new['SeniorCitizen'].astype(int)
        
        if 'gender' in df_new.columns:
            df_new['is_male'] = (df_new['gender'] == 'Male').astype(int)
        
        if 'Partner' in df_new.columns:
            if df_new['Partner'].dtype == 'object':
                df_new['has_partner'] = (df_new['Partner'] == 'Yes').astype(int)
        
        if 'Dependents' in df_new.columns:
            if df_new['Dependents'].dtype == 'object':
                df_new['has_dependents'] = (df_new['Dependents'] == 'Yes').astype(int)
        
        if 'has_partner' in df_new.columns and 'has_dependents' in df_new.columns:
            df_new['family_size'] = df_new['has_partner'] + df_new['has_dependents']
            df_new['is_single'] = ((df_new['has_partner'] == 0) & 
                                 (df_new['has_dependents'] == 0)).astype(int)
        
        logger.info("Demographic features created successfully")
        return df_new
    
    def create_interaction_features(self, df, max_features=50):
        df_new = df.copy()
        
        interaction_pairs = [
            ('tenure', 'MonthlyCharges'),
            ('tenure', 'TotalCharges'),
            ('is_senior_citizen', 'MonthlyCharges'),
            ('has_internet', 'total_services'),
            ('is_month_to_month', 'MonthlyCharges'),
            ('has_fiber_optic', 'MonthlyCharges'),
            ('family_size', 'total_services'),
            ('is_automatic_payment', 'is_long_term_contract')
        ]
        
        interaction_count = 0
        for col1, col2 in interaction_pairs:
            if col1 in df_new.columns and col2 in df_new.columns and interaction_count < max_features:
                feature_name = f"{col1}_x_{col2}"
                df_new[feature_name] = df_new[col1] * df_new[col2]
                interaction_count += 1
        
        logger.info(f"Created {interaction_count} interaction features")
        return df_new
    
    def create_ratio_features(self, df):
        df_new = df.copy()
        
        if 'MonthlyCharges' in df_new.columns and 'total_services' in df_new.columns:
            df_new['charges_per_service'] = np.where(
                df_new['total_services'] > 0,
                df_new['MonthlyCharges'] / df_new['total_services'],
                df_new['MonthlyCharges']
            )
        
        if 'tenure' in df_new.columns and 'MonthlyCharges' in df_new.columns:
            df_new['tenure_to_charges_ratio'] = np.where(
                df_new['MonthlyCharges'] > 0,
                df_new['tenure'] / df_new['MonthlyCharges'],
                0
            )
        
        logger.info("Ratio features created successfully")
        return df_new
    
    def engineer_features(self, df):
        logger.info("Starting feature engineering process")
        
        df_engineered = df.copy()
        
        df_engineered = self.create_tenure_features(df_engineered)
        df_engineered = self.create_charge_features(df_engineered)
        df_engineered = self.create_service_features(df_engineered)
        df_engineered = self.create_contract_features(df_engineered)
        df_engineered = self.create_demographic_features(df_engineered)
        df_engineered = self.create_ratio_features(df_engineered)
        df_engineered = self.create_interaction_features(df_engineered)
        
        self.feature_names = df_engineered.columns.tolist()
        
        logger.info(f"Feature engineering completed. Total features: {len(self.feature_names)}")
        return df_engineered
    
    def get_feature_importance_groups(self):
        feature_groups = {
            'original': [],
            'tenure_based': [],
            'charge_based': [],
            'service_based': [],
            'contract_based': [],
            'demographic_based': [],
            'interaction_based': [],
            'ratio_based': []
        }
        
        for feature in self.feature_names:
            if 'tenure' in feature.lower():
                feature_groups['tenure_based'].append(feature)
            elif any(word in feature.lower() for word in ['charge', 'monthly', 'total']):
                feature_groups['charge_based'].append(feature)
            elif any(word in feature.lower() for word in ['service', 'internet', 'phone', 'streaming']):
                feature_groups['service_based'].append(feature)
            elif any(word in feature.lower() for word in ['contract', 'payment', 'paperless']):
                feature_groups['contract_based'].append(feature)
            elif any(word in feature.lower() for word in ['senior', 'gender', 'partner', 'dependent', 'family']):
                feature_groups['demographic_based'].append(feature)
            elif '_x_' in feature:
                feature_groups['interaction_based'].append(feature)
            elif 'ratio' in feature.lower() or 'per' in feature.lower():
                feature_groups['ratio_based'].append(feature)
            else:
                feature_groups['original'].append(feature)
        
        return feature_groups


def main():
    sample_data = {
        'tenure': [1, 12, 24, 36, 60],
        'MonthlyCharges': [29.85, 56.95, 53.85, 42.30, 78.70],
        'TotalCharges': [29.85, 1889.5, 108.15, 1840.75, 3046.05],
        'Contract': ['Month-to-month', 'One year', 'Month-to-month', 'One year', 'Two year'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Electronic check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
        'InternetService': ['DSL', 'DSL', 'DSL', 'DSL', 'Fiber optic'],
        'SeniorCitizen': [0, 0, 0, 0, 0],
        'Partner': ['Yes', 'No', 'No', 'No', 'Yes'],
        'Dependents': ['No', 'No', 'No', 'No', 'Yes']
    }
    
    df = pd.DataFrame(sample_data)
    
    engineer = FeatureEngineer()
    df_engineered = engineer.engineer_features(df)
    
    print(f"Original features: {len(df.columns)}")
    print(f"Engineered features: {len(df_engineered.columns)}")
    print(f"New features added: {len(df_engineered.columns) - len(df.columns)}")
    
    feature_groups = engineer.get_feature_importance_groups()
    for group, features in feature_groups.items():
        if features:
            print(f"\n{group.upper()} features ({len(features)}): {features[:5]}...")


if __name__ == "__main__":
    main()