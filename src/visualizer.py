import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

plt.style.use('default')
sns.set_palette("husl")


class ChurnVisualizer:
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        self.color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
    def plot_churn_distribution(self, df, target_col='Churn'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        churn_counts = df[target_col].value_counts()
        
        ax1.pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%',
                colors=self.color_palette[:len(churn_counts)])
        ax1.set_title('Churn Distribution')
        
        sns.countplot(data=df, x=target_col, ax=ax2, hue=target_col, palette=self.color_palette, legend=False)
        ax2.set_title('Churn Count')
        ax2.set_xlabel('Churn Status')
        ax2.set_ylabel('Number of Customers')
        
        plt.tight_layout()
        return fig
    
    def plot_feature_correlation(self, df, target_col='Churn', top_n=15):
        numeric_df = df.select_dtypes(include=[np.number])
        
        if target_col in numeric_df.columns:
            correlations = numeric_df.corr()[target_col].abs().sort_values(ascending=False)
            top_features = correlations.head(top_n + 1).drop(target_col)
        else:
            correlations = numeric_df.corr().abs()
            top_features = correlations.mean().sort_values(ascending=False).head(top_n)
        
        plt.figure(figsize=self.figsize)
        sns.barplot(x=top_features.values, y=top_features.index, hue=top_features.index, palette='viridis', legend=False)
        plt.title(f'Top {top_n} Features Correlation with {target_col}')
        plt.xlabel('Absolute Correlation')
        plt.tight_layout()
        return plt.gcf()
    
    def plot_categorical_analysis(self, df, categorical_cols, target_col='Churn'):
        n_cols = min(3, len(categorical_cols))
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(categorical_cols):
            if i < len(axes):
                cross_tab = pd.crosstab(df[col], df[target_col], normalize='index') * 100
                cross_tab.plot(kind='bar', ax=axes[i], color=self.color_palette[:2])
                axes[i].set_title(f'{col} vs {target_col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Percentage')
                axes[i].legend(title=target_col)
                axes[i].tick_params(axis='x', rotation=45)
        
        for i in range(len(categorical_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_numerical_distributions(self, df, numerical_cols, target_col='Churn'):
        n_cols = min(3, len(numerical_cols))
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numerical_cols):
            if i < len(axes) and col in df.columns:
                for j, churn_value in enumerate(df[target_col].unique()):
                    subset = df[df[target_col] == churn_value][col]
                    axes[i].hist(subset, alpha=0.7, label=f'{target_col}={churn_value}',
                               color=self.color_palette[j], bins=30)
                
                axes[i].set_title(f'{col} Distribution')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
                axes[i].legend()
        
        for i in range(len(numerical_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_tenure_analysis(self, df, target_col='Churn'):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        sns.histplot(data=df, x='tenure', hue=target_col, bins=30, ax=axes[0,0])
        axes[0,0].set_title('Tenure Distribution by Churn')
        
        tenure_churn = df.groupby('tenure')[target_col].mean()
        axes[0,1].plot(tenure_churn.index, tenure_churn.values, marker='o')
        axes[0,1].set_title('Churn Rate by Tenure')
        axes[0,1].set_xlabel('Tenure (months)')
        axes[0,1].set_ylabel('Churn Rate')
        
        df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72, float('inf')],
                                   labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr', '6+yr'])
        
        tenure_group_churn = df.groupby('tenure_group')[target_col].agg(['mean', 'count'])
        axes[1,0].bar(range(len(tenure_group_churn)), tenure_group_churn['mean'],
                     color=self.color_palette[0])
        axes[1,0].set_title('Churn Rate by Tenure Group')
        axes[1,0].set_xlabel('Tenure Group')
        axes[1,0].set_ylabel('Churn Rate')
        axes[1,0].set_xticks(range(len(tenure_group_churn)))
        axes[1,0].set_xticklabels(tenure_group_churn.index, rotation=45)
        
        axes[1,1].bar(range(len(tenure_group_churn)), tenure_group_churn['count'],
                     color=self.color_palette[1])
        axes[1,1].set_title('Customer Count by Tenure Group')
        axes[1,1].set_xlabel('Tenure Group')
        axes[1,1].set_ylabel('Customer Count')
        axes[1,1].set_xticks(range(len(tenure_group_churn)))
        axes[1,1].set_xticklabels(tenure_group_churn.index, rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_charges_analysis(self, df, target_col='Churn'):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        sns.boxplot(data=df, x=target_col, y='MonthlyCharges', ax=axes[0,0])
        axes[0,0].set_title('Monthly Charges by Churn')
        
        sns.boxplot(data=df, x=target_col, y='TotalCharges', ax=axes[0,1])
        axes[0,1].set_title('Total Charges by Churn')
        
        sns.scatterplot(data=df, x='tenure', y='MonthlyCharges', hue=target_col, ax=axes[1,0])
        axes[1,0].set_title('Tenure vs Monthly Charges')
        
        sns.scatterplot(data=df, x='MonthlyCharges', y='TotalCharges', hue=target_col, ax=axes[1,1])
        axes[1,1].set_title('Monthly vs Total Charges')
        
        plt.tight_layout()
        return fig
    
    def create_interactive_dashboard(self, df, target_col='Churn'):
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Churn Distribution', 'Monthly Charges Distribution',
                          'Tenure vs Monthly Charges', 'Service Usage'),
            specs=[[{"type": "pie"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        churn_counts = df[target_col].value_counts()
        fig.add_trace(
            go.Pie(labels=churn_counts.index, values=churn_counts.values,
                   name="Churn Distribution"),
            row=1, col=1
        )
        
        for churn_value in df[target_col].unique():
            subset = df[df[target_col] == churn_value]
            fig.add_trace(
                go.Histogram(x=subset['MonthlyCharges'], name=f'Churn={churn_value}',
                           opacity=0.7),
                row=1, col=2
            )
        
        for churn_value in df[target_col].unique():
            subset = df[df[target_col] == churn_value]
            fig.add_trace(
                go.Scatter(x=subset['tenure'], y=subset['MonthlyCharges'],
                          mode='markers', name=f'Churn={churn_value}',
                          opacity=0.6),
                row=2, col=1
            )
        
        service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 'TechSupport']
        available_services = [col for col in service_cols if col in df.columns]
        
        if available_services:
            service_usage = df[available_services].apply(lambda x: (x == 'Yes').sum() if x.dtype == 'object' else x.sum())
            fig.add_trace(
                go.Bar(x=service_usage.index, y=service_usage.values,
                       name="Service Usage"),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=True, title_text="Customer Churn Analysis Dashboard")
        return fig
    
    def plot_model_performance(self, y_true, y_pred, y_pred_proba=None):
        from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        if y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            axes[1].plot(fpr, tpr, color='darkorange', lw=2,
                        label=f'ROC curve (AUC = {roc_auc:.2f})')
            axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[1].set_xlim([0.0, 1.0])
            axes[1].set_ylim([0.0, 1.05])
            axes[1].set_xlabel('False Positive Rate')
            axes[1].set_ylabel('True Positive Rate')
            axes[1].set_title('ROC Curve')
            axes[1].legend(loc="lower right")
            
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            axes[2].plot(recall, precision, color='blue', lw=2)
            axes[2].set_xlabel('Recall')
            axes[2].set_ylabel('Precision')
            axes[2].set_title('Precision-Recall Curve')
            axes[2].set_xlim([0.0, 1.0])
            axes[2].set_ylim([0.0, 1.05])
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, feature_importance_df, top_n=20):
        top_features = feature_importance_df.head(top_n)
        
        plt.figure(figsize=(10, max(6, top_n * 0.3)))
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        return plt.gcf()
    
    def save_all_plots(self, df, target_col='Churn', save_dir='reports/'):
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        plots = {
            'churn_distribution': self.plot_churn_distribution(df, target_col),
            'feature_correlation': self.plot_feature_correlation(df, target_col),
            'tenure_analysis': self.plot_tenure_analysis(df, target_col),
            'charges_analysis': self.plot_charges_analysis(df, target_col)
        }
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
        
        if categorical_cols:
            plots['categorical_analysis'] = self.plot_categorical_analysis(df, categorical_cols[:6], target_col)
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numerical_cols:
            numerical_cols.remove(target_col)
        
        if numerical_cols:
            plots['numerical_distributions'] = self.plot_numerical_distributions(df, numerical_cols[:6], target_col)
        
        for plot_name, fig in plots.items():
            save_path = os.path.join(save_dir, f'{plot_name}.png')
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved: {save_path}")
        
        plt.close('all')
        return list(plots.keys())


def main():
    np.random.seed(42)
    
    sample_data = {
        'tenure': np.random.randint(1, 73, 1000),
        'MonthlyCharges': np.random.normal(65, 20, 1000),
        'TotalCharges': np.random.normal(2000, 1000, 1000),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], 1000),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], 1000),
        'Churn': np.random.choice([0, 1], 1000, p=[0.73, 0.27])
    }
    
    df = pd.DataFrame(sample_data)
    df['TotalCharges'] = np.abs(df['TotalCharges'])
    df['MonthlyCharges'] = np.abs(df['MonthlyCharges'])
    
    visualizer = ChurnVisualizer()
    
    print("Creating sample visualizations...")
    
    churn_fig = visualizer.plot_churn_distribution(df)
    print("✓ Churn distribution plot created")
    
    correlation_fig = visualizer.plot_feature_correlation(df)
    print("✓ Feature correlation plot created")
    
    tenure_fig = visualizer.plot_tenure_analysis(df)
    print("✓ Tenure analysis plot created")
    
    charges_fig = visualizer.plot_charges_analysis(df)
    print("✓ Charges analysis plot created")
    
    interactive_fig = visualizer.create_interactive_dashboard(df)
    print("✓ Interactive dashboard created")
    
    plt.show()
    print("\nAll visualizations created successfully!")


if __name__ == "__main__":
    main()