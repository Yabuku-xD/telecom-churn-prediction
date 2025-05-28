import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnPredictor:
    def __init__(self, model_path=None):
        self.model = None
        self.feature_columns = None
        self.model_loaded = False
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        try:
            self.model = joblib.load(model_path)
            self.model_loaded = True
            logger.info(f"Model loaded successfully from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {str(e)}")
            return False
    
    def set_feature_columns(self, feature_columns):
        self.feature_columns = feature_columns
        logger.info(f"Feature columns set: {len(feature_columns)} features")
    
    def predict_single_customer(self, customer_data):
        if not self.model_loaded:
            raise ValueError("No model loaded. Please load a model first.")
        
        if isinstance(customer_data, dict):
            customer_df = pd.DataFrame([customer_data])
        else:
            customer_df = customer_data.copy()
        
        if self.feature_columns:
            missing_features = set(self.feature_columns) - set(customer_df.columns)
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                for feature in missing_features:
                    customer_df[feature] = 0
            
            customer_df = customer_df[self.feature_columns]
        
        try:
            churn_probability = self.model.predict_proba(customer_df)[0, 1]
            churn_prediction = self.model.predict(customer_df)[0]
            
            risk_level = self.get_risk_level(churn_probability)
            recommendations = self.get_recommendations(customer_data, churn_probability)
            
            result = {
                'customer_id': customer_data.get('customer_id', 'Unknown'),
                'churn_probability': float(churn_probability),
                'churn_prediction': bool(churn_prediction),
                'risk_level': risk_level,
                'recommendations': recommendations,
                'prediction_date': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def predict_batch(self, customers_data):
        if not self.model_loaded:
            raise ValueError("No model loaded. Please load a model first.")
        
        if isinstance(customers_data, list):
            customers_df = pd.DataFrame(customers_data)
        else:
            customers_df = customers_data.copy()
        
        if self.feature_columns:
            missing_features = set(self.feature_columns) - set(customers_df.columns)
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                for feature in missing_features:
                    customers_df[feature] = 0
            
            customers_df = customers_df[self.feature_columns]
        
        try:
            churn_probabilities = self.model.predict_proba(customers_df)[:, 1]
            churn_predictions = self.model.predict(customers_df)
            
            results = []
            for i, (prob, pred) in enumerate(zip(churn_probabilities, churn_predictions)):
                customer_data = customers_data.iloc[i] if hasattr(customers_data, 'iloc') else customers_data[i]
                
                result = {
                    'customer_id': customer_data.get('customer_id', f'Customer_{i}'),
                    'churn_probability': float(prob),
                    'churn_prediction': bool(pred),
                    'risk_level': self.get_risk_level(prob),
                    'recommendations': self.get_recommendations(customer_data, prob)
                }
                results.append(result)
            
            logger.info(f"Batch prediction completed for {len(results)} customers")
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            raise
    
    def get_risk_level(self, churn_probability):
        if churn_probability >= 0.8:
            return "Very High"
        elif churn_probability >= 0.6:
            return "High"
        elif churn_probability >= 0.4:
            return "Medium"
        elif churn_probability >= 0.2:
            return "Low"
        else:
            return "Very Low"
    
    def get_recommendations(self, customer_data, churn_probability):
        recommendations = []
        
        if churn_probability >= 0.6:
            recommendations.append("Immediate intervention required")
            recommendations.append("Assign dedicated account manager")
            recommendations.append("Offer retention discount (15-25%)")
        
        if customer_data.get('Contract') == 'Month-to-month':
            recommendations.append("Offer annual contract with discount")
            recommendations.append("Highlight benefits of long-term commitment")
        
        if customer_data.get('PaymentMethod') == 'Electronic check':
            recommendations.append("Encourage automatic payment setup")
            recommendations.append("Offer payment method incentives")
        
        if customer_data.get('InternetService') == 'Fiber optic' and churn_probability > 0.5:
            recommendations.append("Review service quality and speed")
            recommendations.append("Offer technical support consultation")
        
        if customer_data.get('TechSupport') == 'No' and churn_probability > 0.4:
            recommendations.append("Offer complimentary tech support trial")
            recommendations.append("Provide self-service resources")
        
        monthly_charges = customer_data.get('MonthlyCharges', 0)
        if monthly_charges > 80 and churn_probability > 0.5:
            recommendations.append("Review pricing and offer value packages")
            recommendations.append("Highlight premium service benefits")
        
        tenure = customer_data.get('tenure', 0)
        if tenure < 12 and churn_probability > 0.4:
            recommendations.append("Implement new customer onboarding program")
            recommendations.append("Provide welcome package and tutorials")
        
        if customer_data.get('SeniorCitizen') == 1:
            recommendations.append("Offer senior citizen discounts")
            recommendations.append("Provide simplified service options")
        
        if not recommendations:
            recommendations.append("Continue standard customer care")
            recommendations.append("Monitor satisfaction regularly")
        
        return recommendations[:5]
    
    def analyze_churn_factors(self, customer_data, churn_probability):
        factors = {
            'high_risk_factors': [],
            'protective_factors': [],
            'neutral_factors': []
        }
        
        if customer_data.get('Contract') == 'Month-to-month':
            factors['high_risk_factors'].append('Month-to-month contract')
        elif customer_data.get('Contract') in ['One year', 'Two year']:
            factors['protective_factors'].append('Long-term contract')
        
        if customer_data.get('PaymentMethod') == 'Electronic check':
            factors['high_risk_factors'].append('Electronic check payment')
        elif 'automatic' in str(customer_data.get('PaymentMethod', '')).lower():
            factors['protective_factors'].append('Automatic payment')
        
        tenure = customer_data.get('tenure', 0)
        if tenure < 12:
            factors['high_risk_factors'].append('New customer (< 1 year)')
        elif tenure > 60:
            factors['protective_factors'].append('Long-term customer (5+ years)')
        
        monthly_charges = customer_data.get('MonthlyCharges', 0)
        if monthly_charges > 80:
            factors['high_risk_factors'].append('High monthly charges')
        elif monthly_charges < 30:
            factors['protective_factors'].append('Low monthly charges')
        
        if customer_data.get('InternetService') == 'Fiber optic':
            factors['high_risk_factors'].append('Fiber optic service')
        
        if customer_data.get('TechSupport') == 'Yes':
            factors['protective_factors'].append('Has tech support')
        
        if customer_data.get('Partner') == 'Yes' or customer_data.get('Dependents') == 'Yes':
            factors['protective_factors'].append('Has family connections')
        
        return factors
    
    def get_retention_strategy(self, customer_data, churn_probability):
        risk_level = self.get_risk_level(churn_probability)
        
        strategy = {
            'priority': risk_level,
            'timeline': 'Immediate' if churn_probability >= 0.7 else 'Within 30 days',
            'budget_allocation': 'High' if churn_probability >= 0.6 else 'Medium',
            'success_probability': max(0.1, 0.9 - churn_probability)
        }
        
        if churn_probability >= 0.8:
            strategy['actions'] = [
                'Executive escalation',
                'Retention specialist assignment',
                'Customized retention offer',
                'Service quality review'
            ]
        elif churn_probability >= 0.6:
            strategy['actions'] = [
                'Account manager outreach',
                'Retention discount offer',
                'Service upgrade consultation',
                'Payment plan adjustment'
            ]
        elif churn_probability >= 0.4:
            strategy['actions'] = [
                'Proactive customer care call',
                'Service satisfaction survey',
                'Loyalty program enrollment',
                'Usage optimization tips'
            ]
        else:
            strategy['actions'] = [
                'Regular satisfaction monitoring',
                'Cross-sell opportunities',
                'Referral program invitation',
                'Service enhancement notifications'
            ]
        
        return strategy


def main():
    sample_customer = {
        'customer_id': 'CUST_001',
        'tenure': 12,
        'MonthlyCharges': 75.50,
        'TotalCharges': 906.00,
        'Contract': 'Month-to-month',
        'PaymentMethod': 'Electronic check',
        'InternetService': 'Fiber optic',
        'TechSupport': 'No',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No'
    }
    
    predictor = ChurnPredictor()
    
    print("ChurnPredictor Demo")
    print("==================")
    print("Note: This demo runs without a trained model")
    print(f"Sample customer: {sample_customer['customer_id']}")
    
    risk_level = predictor.get_risk_level(0.75)
    print(f"Risk level for 75% churn probability: {risk_level}")
    
    recommendations = predictor.get_recommendations(sample_customer, 0.75)
    print(f"Recommendations: {recommendations[:3]}")
    
    factors = predictor.analyze_churn_factors(sample_customer, 0.75)
    print(f"High risk factors: {factors['high_risk_factors']}")
    
    strategy = predictor.get_retention_strategy(sample_customer, 0.75)
    print(f"Retention strategy priority: {strategy['priority']}")
    print(f"Recommended actions: {strategy['actions'][:2]}")


if __name__ == "__main__":
    main()