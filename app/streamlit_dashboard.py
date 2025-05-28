"""
Streamlit Dashboard for Telecom Churn Prediction

Interactive web application for business stakeholders to analyze customer churn,
make predictions, and get actionable insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processor import DataProcessor
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.predictor import ChurnPredictor
from src.visualizer import ChurnVisualizer

# Page configuration
st.set_page_config(
    page_title="Telecom Churn Prediction Dashboard",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-high {
        color: #d62728;
        font-weight: bold;
    }
    .risk-medium {
        color: #ff7f0e;
        font-weight: bold;
    }
    .risk-low {
        color: #2ca02c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_sample_data():
    """Load sample data for demonstration."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'customerID': [f'CUST_{i:04d}' for i in range(n_samples)],
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.84, 0.16]),
        'Partner': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        'tenure': np.random.randint(1, 73, n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.4, 0.2]),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.25, 0.2]),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4]),
        'PaymentMethod': np.random.choice([
            'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
        ], n_samples),
        'MonthlyCharges': np.random.normal(65, 20, n_samples),
        'TotalCharges': np.random.normal(2000, 1500, n_samples),
        'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.27, 0.73])
    }
    
    df = pd.DataFrame(data)
    df['MonthlyCharges'] = np.maximum(df['MonthlyCharges'], 20)
    df['TotalCharges'] = np.maximum(df['TotalCharges'], df['MonthlyCharges'])
    
    return df


def create_risk_gauge(probability):
    """Create a risk gauge chart."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Churn Risk (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 50], 'color': "yellow"},
                {'range': [50, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig


def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<h1 class="main-header">📱 Telecom Churn Prediction Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Data Analysis", "Customer Prediction", "Batch Analysis", "Business Insights"]
    )
    
    # Load data
    df = load_sample_data()
    
    if page == "Overview":
        show_overview(df)
    elif page == "Data Analysis":
        show_data_analysis(df)
    elif page == "Customer Prediction":
        show_customer_prediction()
    elif page == "Batch Analysis":
        show_batch_analysis(df)
    elif page == "Business Insights":
        show_business_insights(df)


def show_overview(df):
    """Show overview page."""
    st.header("📊 Business Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_customers = len(df)
    churned_customers = len(df[df['Churn'] == 'Yes'])
    churn_rate = (churned_customers / total_customers) * 100
    avg_monthly_revenue = df['MonthlyCharges'].mean()
    
    with col1:
        st.metric("Total Customers", f"{total_customers:,}")
    
    with col2:
        st.metric("Churned Customers", f"{churned_customers:,}")
    
    with col3:
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    
    with col4:
        st.metric("Avg Monthly Revenue", f"${avg_monthly_revenue:.2f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Churn Distribution")
        churn_counts = df['Churn'].value_counts()
        fig = px.pie(values=churn_counts.values, names=churn_counts.index,
                    title="Customer Churn Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Monthly Charges vs Tenure")
        fig = px.scatter(df, x='tenure', y='MonthlyCharges', color='Churn',
                        title="Customer Tenure vs Monthly Charges")
        st.plotly_chart(fig, use_container_width=True)
    
    # Revenue impact
    st.subheader("💰 Revenue Impact Analysis")
    
    churned_revenue = df[df['Churn'] == 'Yes']['MonthlyCharges'].sum()
    total_revenue = df['MonthlyCharges'].sum()
    revenue_loss_pct = (churned_revenue / total_revenue) * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Monthly Revenue Loss", f"${churned_revenue:,.2f}")
    
    with col2:
        st.metric("Revenue Loss %", f"{revenue_loss_pct:.1f}%")
    
    with col3:
        annual_loss = churned_revenue * 12
        st.metric("Estimated Annual Loss", f"${annual_loss:,.2f}")


def show_data_analysis(df):
    """Show data analysis page."""
    st.header("📈 Data Analysis")
    
    # Dataset overview
    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Shape:**", df.shape)
        st.write("**Missing Values:**", df.isnull().sum().sum())
    
    with col2:
        st.write("**Data Types:**")
        st.write(df.dtypes.value_counts())
    
    # Feature analysis
    st.subheader("Feature Analysis")
    
    # Select features to analyze
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    categorical_features.remove('customerID')
    if 'Churn' in categorical_features:
        categorical_features.remove('Churn')
    
    selected_feature = st.selectbox("Select feature to analyze:", categorical_features)
    
    if selected_feature:
        col1, col2 = st.columns(2)
        
        with col1:
            # Feature distribution
            feature_counts = df[selected_feature].value_counts()
            fig = px.bar(x=feature_counts.index, y=feature_counts.values,
                        title=f"{selected_feature} Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Churn rate by feature
            churn_by_feature = df.groupby(selected_feature)['Churn'].apply(
                lambda x: (x == 'Yes').sum() / len(x) * 100
            ).reset_index()
            churn_by_feature.columns = [selected_feature, 'Churn_Rate']
            
            fig = px.bar(churn_by_feature, x=selected_feature, y='Churn_Rate',
                        title=f"Churn Rate by {selected_feature}")
            st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.subheader("Correlation Analysis")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numerical_cols].corr()
    
    fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto",
                   title="Feature Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)


def show_customer_prediction():
    """Show customer prediction page."""
    st.header("🎯 Individual Customer Prediction")
    
    st.write("Enter customer details to predict churn probability:")
    
    # Customer input form
    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
        
        with col2:
            tenure = st.slider("Tenure (months)", 1, 72, 12)
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        
        with col3:
            monthly_charges = st.number_input("Monthly Charges ($)", 20.0, 120.0, 65.0)
            total_charges = st.number_input("Total Charges ($)", 20.0, 10000.0, 1000.0)
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", 
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        
        submitted = st.form_submit_button("Predict Churn Risk")
        
        if submitted:
            # Create customer data
            customer_data = {
                'gender': gender,
                'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'InternetService': internet_service,
                'Contract': contract,
                'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }
            
            # Simulate prediction (in real app, would use trained model)
            # For demo, create a simple rule-based prediction
            risk_factors = 0
            if contract == "Month-to-month":
                risk_factors += 0.3
            if payment_method == "Electronic check":
                risk_factors += 0.2
            if tenure < 12:
                risk_factors += 0.25
            if monthly_charges > 70:
                risk_factors += 0.15
            if senior_citizen == "Yes":
                risk_factors += 0.1
            
            churn_probability = min(risk_factors, 0.95)
            
            # Display results
            st.subheader("Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk gauge
                fig = create_risk_gauge(churn_probability)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Risk level and recommendations
                if churn_probability > 0.7:
                    risk_level = "Critical"
                    risk_color = "risk-high"
                elif churn_probability > 0.5:
                    risk_level = "High"
                    risk_color = "risk-high"
                elif churn_probability > 0.3:
                    risk_level = "Medium"
                    risk_color = "risk-medium"
                else:
                    risk_level = "Low"
                    risk_color = "risk-low"
                
                st.markdown(f"**Risk Level:** <span class='{risk_color}'>{risk_level}</span>", 
                           unsafe_allow_html=True)
                st.write(f"**Churn Probability:** {churn_probability:.1%}")
                
                # Recommendations
                st.subheader("Recommendations")
                if churn_probability > 0.5:
                    st.write("🚨 **Immediate Action Required:**")
                    st.write("- Offer contract extension with discount")
                    st.write("- Provide premium customer support")
                    st.write("- Consider loyalty rewards program")
                elif churn_probability > 0.3:
                    st.write("⚠️ **Monitor Closely:**")
                    st.write("- Schedule satisfaction survey")
                    st.write("- Offer service upgrades")
                    st.write("- Improve payment experience")
                else:
                    st.write("✅ **Maintain Current Service:**")
                    st.write("- Continue excellent service")
                    st.write("- Consider upselling opportunities")


def show_batch_analysis(df):
    """Show batch analysis page."""
    st.header("📋 Batch Customer Analysis")
    
    # File upload
    st.subheader("Upload Customer Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load uploaded data
        uploaded_df = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:")
        st.dataframe(uploaded_df.head())
    else:
        # Use sample data
        st.write("Using sample data for demonstration:")
        uploaded_df = df.copy()
    
    # Simulate batch predictions
    np.random.seed(42)
    uploaded_df['churn_probability'] = np.random.beta(2, 5, len(uploaded_df))
    uploaded_df['risk_level'] = uploaded_df['churn_probability'].apply(
        lambda x: 'Critical' if x > 0.7 else 'High' if x > 0.5 else 'Medium' if x > 0.3 else 'Low'
    )
    
    # Summary statistics
    st.subheader("Batch Analysis Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_analyzed = len(uploaded_df)
    high_risk = len(uploaded_df[uploaded_df['churn_probability'] > 0.5])
    medium_risk = len(uploaded_df[(uploaded_df['churn_probability'] > 0.3) & 
                                 (uploaded_df['churn_probability'] <= 0.5)])
    low_risk = len(uploaded_df[uploaded_df['churn_probability'] <= 0.3])
    
    with col1:
        st.metric("Total Customers", total_analyzed)
    with col2:
        st.metric("High Risk", high_risk, f"{(high_risk/total_analyzed)*100:.1f}%")
    with col3:
        st.metric("Medium Risk", medium_risk, f"{(medium_risk/total_analyzed)*100:.1f}%")
    with col4:
        st.metric("Low Risk", low_risk, f"{(low_risk/total_analyzed)*100:.1f}%")
    
    # Risk distribution
    col1, col2 = st.columns(2)
    
    with col1:
        risk_counts = uploaded_df['risk_level'].value_counts()
        fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                    title="Risk Level Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(uploaded_df, x='churn_probability', nbins=20,
                          title="Churn Probability Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # High-risk customers table
    st.subheader("High-Risk Customers (Probability > 50%)")
    high_risk_customers = uploaded_df[uploaded_df['churn_probability'] > 0.5].copy()
    
    if len(high_risk_customers) > 0:
        # Select relevant columns for display
        display_cols = ['customerID', 'Contract', 'tenure', 'MonthlyCharges', 
                       'churn_probability', 'risk_level']
        available_cols = [col for col in display_cols if col in high_risk_customers.columns]
        
        st.dataframe(
            high_risk_customers[available_cols].sort_values('churn_probability', ascending=False),
            use_container_width=True
        )
        
        # Download button
        csv = high_risk_customers.to_csv(index=False)
        st.download_button(
            label="Download High-Risk Customers CSV",
            data=csv,
            file_name="high_risk_customers.csv",
            mime="text/csv"
        )
    else:
        st.write("No high-risk customers found in the current dataset.")


def show_business_insights(df):
    """Show business insights page."""
    st.header("💼 Business Insights & ROI Analysis")
    
    # Calculate business metrics
    total_customers = len(df)
    churned_customers = len(df[df['Churn'] == 'Yes'])
    monthly_revenue_loss = df[df['Churn'] == 'Yes']['MonthlyCharges'].sum()
    annual_revenue_loss = monthly_revenue_loss * 12
    
    # ROI Analysis
    st.subheader("💰 ROI Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Current Situation:**")
        st.metric("Annual Revenue Loss", f"${annual_revenue_loss:,.2f}")
        st.metric("Customers Lost Annually", f"{churned_customers * 12:,}")
        
        # Intervention scenarios
        st.write("**Intervention Scenarios:**")
        
        # Scenario 1: 15% reduction in churn
        scenario_1_reduction = 0.15
        scenario_1_savings = annual_revenue_loss * scenario_1_reduction
        st.write(f"**15% Churn Reduction:** ${scenario_1_savings:,.2f} annual savings")
        
        # Scenario 2: 25% reduction in churn
        scenario_2_reduction = 0.25
        scenario_2_savings = annual_revenue_loss * scenario_2_reduction
        st.write(f"**25% Churn Reduction:** ${scenario_2_savings:,.2f} annual savings")
    
    with col2:
        # Implementation costs and ROI
        st.write("**Implementation Costs:**")
        
        implementation_cost = st.number_input("Implementation Cost ($)", 
                                            value=150000, step=10000)
        annual_maintenance = st.number_input("Annual Maintenance ($)", 
                                           value=50000, step=5000)
        
        # Calculate ROI for different scenarios
        roi_1_year = ((scenario_2_savings - annual_maintenance) / implementation_cost) * 100
        roi_3_year = ((scenario_2_savings * 3 - annual_maintenance * 3) / implementation_cost) * 100
        
        st.write("**ROI Analysis (25% churn reduction):**")
        st.metric("1-Year ROI", f"{roi_1_year:.1f}%")
        st.metric("3-Year ROI", f"{roi_3_year:.1f}%")
        st.metric("Payback Period", f"{implementation_cost / (scenario_2_savings - annual_maintenance):.1f} years")
    
    # Customer segment analysis
    st.subheader("📊 Customer Segment Analysis")
    
    # Churn by contract type
    contract_analysis = df.groupby('Contract').agg({
        'Churn': lambda x: (x == 'Yes').sum() / len(x) * 100,
        'MonthlyCharges': 'mean',
        'customerID': 'count'
    }).round(2)
    contract_analysis.columns = ['Churn_Rate_%', 'Avg_Monthly_Charges', 'Customer_Count']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Churn Rate by Contract Type:**")
        st.dataframe(contract_analysis)
    
    with col2:
        fig = px.bar(contract_analysis.reset_index(), 
                    x='Contract', y='Churn_Rate_%',
                    title="Churn Rate by Contract Type")
        st.plotly_chart(fig, use_container_width=True)
    
    # Action plan
    st.subheader("🎯 Recommended Action Plan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Immediate Actions (0-30 days):**")
        st.write("1. 🎯 Target month-to-month customers with contract incentives")
        st.write("2. 📞 Proactive outreach to high-risk customers")
        st.write("3. 💳 Improve electronic check payment experience")
        st.write("4. 🛠️ Enhance technical support for fiber optic customers")
    
    with col2:
        st.write("**Medium-term Actions (1-6 months):**")
        st.write("1. 📈 Implement predictive churn model")
        st.write("2. 🎁 Develop loyalty rewards program")
        st.write("3. 📊 Create customer health score dashboard")
        st.write("4. 🔄 Establish retention team and processes")
    
    # Success metrics
    st.subheader("📈 Success Metrics to Track")
    
    metrics_df = pd.DataFrame({
        'Metric': [
            'Monthly Churn Rate',
            'Customer Lifetime Value',
            'Revenue Retention Rate',
            'Customer Satisfaction Score',
            'Contract Conversion Rate'
        ],
        'Current': ['27%', '$2,400', '73%', '3.2/5', '15%'],
        'Target': ['20%', '$3,200', '80%', '4.0/5', '25%'],
        'Timeline': ['6 months', '12 months', '6 months', '3 months', '6 months']
    })
    
    st.dataframe(metrics_df, use_container_width=True)


if __name__ == "__main__":
    main()