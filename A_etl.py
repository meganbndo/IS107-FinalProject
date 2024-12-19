import pandas as pd

# Step 1: Extract
file_path = 'Online Retail.xlsx'
data = pd.read_excel(file_path)

# Step 2: Transform
try:
    # Initial Row Count
    uncleaned_row_count = data.shape[0]
    print(f"Number of rows in uncleaned data: {uncleaned_row_count}")

    # Clean missing values
    data = data.dropna(subset=['CustomerID', 'InvoiceDate'])
    data['CustomerID'] = data['CustomerID'].astype(int)

    # Fill missing descriptions based on existing descriptions for the same StockCode
    description_map = data.dropna(subset=['Description']).groupby('StockCode')['Description'].first()
    data['Description'] = data.apply(
        lambda row: description_map[row['StockCode']] if pd.isna(row['Description']) and row['StockCode'] in description_map else row['Description'],
        axis=1
    )

    # Identify unique StockCodes with missing descriptions
    unique_stockcodes = data[data['Description'].isna()]['StockCode'].unique()

    # Fill missing descriptions for unique StockCodes
    data.loc[data['StockCode'].isin(unique_stockcodes) & data['Description'].isna(), 'Description'] = 'Unknown Product'

    # Remove rows with zero and negative quantities or non-positive UnitPrice
    data = data[(data['Quantity'] > 0) & (data['UnitPrice'] > 0)]

    # Remove outliers based on IQR for Quantity and UnitPrice
    for col in ['Quantity', 'UnitPrice']:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[data[col].between(lower_bound, upper_bound)]

    # Calculate total price
    data['TotalPrice'] = data['Quantity'] * data['UnitPrice']

    # Clean and standardize text fields
    data['Description'] = data['Description'].str.strip().str.title()

    # Handle date inconsistencies
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], errors='coerce')
    data = data[data['InvoiceDate'].notna()]  # Drop rows with invalid dates

    # Remove rows with suspicious dates
    min_date = '2010-01-01'
    max_date = pd.Timestamp.today()
    data = data[(data['InvoiceDate'] >= min_date) & (data['InvoiceDate'] <= max_date)]

    # Create new columns
    data['InvoiceYear'] = data['InvoiceDate'].dt.year
    data['InvoiceMonth'] = data['InvoiceDate'].dt.month
    data['InvoiceDay'] = data['InvoiceDate'].dt.day
    data['DayOfWeek'] = data['InvoiceDate'].dt.dayofweek
    data['WeekOfYear'] = data['InvoiceDate'].dt.isocalendar().week

    # Deduplicate rows before proceeding further
    data = data.drop_duplicates()

    # Enrich data: Order size classification
    data['OrderSize'] = pd.cut(data['TotalPrice'], bins=[0, 50, 500, float('inf')], labels=['Small', 'Medium', 'Large'])

    # Final Row Count
    cleaned_row_count = data.shape[0]
    print(f"Number of rows in cleaned data: {cleaned_row_count}")

    # Pre-insertion check: Remove any duplicates from the data
    # This ensures that no duplicate records exist before saving to CSV or inserting into a database
    data = data.drop_duplicates(subset=['CustomerID', 'InvoiceDate', 'StockCode'])  # Deduplicate based on key columns

    # Save cleaned data
    output_csv_path = 'cleaned_data.csv'
    data.to_csv(output_csv_path, index=False)
    print(f"Cleaned data saved to {output_csv_path}")

except Exception as e:
    print(f"Error during data transformation: {e}")
    exit()

# Step 3: Load
# If you were loading to a database, the upsert technique would be implemented here (example below for PostgreSQL):