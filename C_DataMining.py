import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

class RetailAnalytics:
    def __init__(self):
        # Initialize the database connection
        self.engine = create_engine('postgresql://postgres:postgres@localhost/data_warehouse')

    def load_data(self):
        """Load data from PostgreSQL database."""
        query = """
        SELECT 
            sf.customer_id, 
            SUM(sf.total_price) AS total_price,
            AVG(sf.total_price) AS avg_sales,
            COUNT(sf.invoice_no) AS transaction_count,
            MIN(td.invoice_date) AS first_purchase,
            MAX(td.invoice_date) AS last_purchase,
            EXTRACT(MONTH FROM MIN(td.invoice_date)) AS month,
            EXTRACT(YEAR FROM MIN(td.invoice_date)) AS year
        FROM sales_fact sf
        JOIN time_dimension td ON sf.time_id = td.time_id
        GROUP BY sf.customer_id
        """
        df = pd.read_sql(query, self.engine)
        
        # Calculate purchase frequency and handle division issues
        df['customer_lifetime'] = (
            pd.to_datetime(df['last_purchase']) - 
            pd.to_datetime(df['first_purchase'])
        ).dt.days
        df['customer_lifetime'] = df['customer_lifetime'].replace(0, 1)  # Avoid zero division
        df['purchase_frequency'] = df['transaction_count'] / df['customer_lifetime']
        
        return df

    def optimal_clusters(self, X_scaled):
        """Determine the optimal number of clusters using the Elbow Method."""
        inertia = []
        for k in range(1, 10):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            inertia.append(kmeans.inertia_)

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, 10), inertia, marker='o')
        plt.title("Elbow Method for Optimal Clusters")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Inertia")
        plt.show()

    def perform_customer_segmentation(self, df, n_clusters=3):
        """Perform customer segmentation using K-means clustering."""
        # Select features for clustering
        features = ['total_price', 'transaction_count']
        X = df[features]

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply K-Means clustering
        self.optimal_clusters(X_scaled)  # Visualize Elbow Method
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['segment'] = kmeans.fit_predict(X_scaled)

        # Cluster summary
        cluster_summary = df.groupby('segment')[['total_price', 'transaction_count', 'purchase_frequency']].mean()
        cluster_summary['count'] = df['segment'].value_counts()
        print("\nCluster Summary:")
        print(cluster_summary)

        # Visualize clusters with centers
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='total_price', y='transaction_count', hue='segment', palette='viridis')
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        # for idx, center in enumerate(cluster_centers):
        #     plt.scatter(center[0], center[1], s=200, c='red', label=f'Cluster {idx} Center', marker='X')
        plt.title('Customer Segmentation Using K-Means Clustering')
        plt.xlabel('Total Sales')
        plt.ylabel('Transaction Count')
        plt.legend()
        plt.show()

        return df, kmeans, cluster_summary

    def perform_sales_prediction(self, df):
        """Perform sales prediction using Linear Regression."""
        # Combine month and year into a single feature
        df['time'] = df['year'] + (df['month'] - 1) / 12

        # Add seasonal features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Features and target
        X = df[['avg_sales', 'transaction_count', 'purchase_frequency', 'time', 'month_sin', 'month_cos']]
        y = df['total_price']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Performance metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R-squared: {r2:.2f}")

        # Visualize Actual vs Predicted
        plt.figure(figsize=(10, 6))
        
        # Plot actual sales
        sns.scatterplot(x=y_test, y=y_test, alpha=0.6, label='Actual Sales', color='blue')
        
        # Plot predicted sales
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, label='Predicted Sales', color='orange')
        
        # Plot perfect prediction line
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Perfect Prediction')
        
        plt.xlabel('Actual Total Sales')
        plt.ylabel('Predicted Total Sales')
        plt.title('Actual vs. Predicted Sales')
        plt.legend()
        plt.show()

        return {'model': model, 'mse': mse, 'r2': r2, 'y_test': y_test, 'y_pred': y_pred}

    def perform_future_forecasting(self, df, model):
        """Perform future sales forecasting."""
        # Generate future time periods (e.g., next 12 months)
        last_year = df['year'].max()
        last_month = df['month'].max()
        last_date = f"{int(last_year)}-{int(last_month)}-01"
        
        future_months = pd.date_range(start=last_date, periods=12, freq='M')
        future_df = pd.DataFrame({
            'month': future_months.month,
            'year': future_months.year,
        })
        future_df['time'] = future_df['year'] + (future_df['month'] - 1) / 12

        # Use historical averages for other features
        future_df['avg_sales'] = df['avg_sales'].mean()
        future_df['transaction_count'] = df['transaction_count'].mean()
        future_df['purchase_frequency'] = df['purchase_frequency'].mean()

        # Add seasonal features
        future_df['month_sin'] = np.sin(2 * np.pi * future_df['month'] / 12)
        future_df['month_cos'] = np.cos(2 * np.pi * future_df['month'] / 12)

        # Predict future sales
        future_df['predicted_sales'] = model.predict(future_df[['avg_sales', 'transaction_count', 'purchase_frequency', 'time', 'month_sin', 'month_cos']])

        # Visualize future sales forecast
        plt.figure(figsize=(10, 6))
        plt.plot(future_months, future_df['predicted_sales'], marker='o', label='Predicted Sales')
        plt.title('Future Sales Forecast')
        plt.xlabel('Date')
        plt.ylabel('Predicted Sales')
        plt.legend()
        plt.grid()
        plt.show()

        return future_df

    def generate_report(self, segmentation_results, cluster_summary, prediction_results):
        """Generate a summary report of findings."""
        report = f"""
        ### Customer Segmentation Insights:
        - Total clusters: {segmentation_results['segment'].nunique()}
        - Cluster distribution:
        {segmentation_results['segment'].value_counts().to_string()}

        ### Cluster Summary:
        {cluster_summary.to_string()}

        ### Sales Prediction Insights:
        - Mean Squared Error (MSE): {prediction_results['mse']:.2f}
        - R-squared: {prediction_results['r2']:.2f}
        """
        print(report)

def main():
    analytics = RetailAnalytics()
    df = analytics.load_data()

    # Perform customer segmentation
    segmented_df, kmeans, cluster_summary = analytics.perform_customer_segmentation(df)

    # Perform sales prediction
    prediction_results = analytics.perform_sales_prediction(df)

    # Perform future forecasting
    future_forecast = analytics.perform_future_forecasting(df, prediction_results['model'])

    # Generate report
    analytics.generate_report(segmented_df, cluster_summary, prediction_results)

if __name__ == "__main__":
    main()
