import pandas as pd
from sqlalchemy import create_engine

# Database connection setup
db_name = "data_warehouse"
db_user = "postgres"
db_password = "postgres"
db_host = "localhost"
db_port = 5432
engine = create_engine(f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

# Load cleaned data
csv_path = "cleaned_data.csv"
cleaned_data = pd.read_csv(csv_path)
cleaned_data['InvoiceDate'] = pd.to_datetime(cleaned_data['InvoiceDate'])


# Load country_dimension with deduplication
try:
    existing_countries = pd.read_sql("SELECT country FROM country_dimension", engine)
    new_countries = pd.DataFrame(cleaned_data['Country'].unique(), columns=['country'])
    new_countries = new_countries[~new_countries['country'].isin(existing_countries['country'])]
    if not new_countries.empty:
        new_countries.to_sql('country_dimension', engine, if_exists='append', index=False)
    print("Countries successfully loaded into country_dimension.")
except Exception as e:
    print(f"Error during loading data into country_dimension: {e}")

# Load customer_dimension with conflict handling
try:
    # Map country to country_id
    country_map = pd.read_sql("SELECT country, country_id FROM country_dimension", engine)
    customer_dim = cleaned_data[['CustomerID', 'Country']].drop_duplicates().rename(columns={'CustomerID': 'customer_id', 'Country': 'country'})
    customer_dim = customer_dim.merge(country_map, how='left', on='country')  # Map country_id
    customer_dim = customer_dim[['customer_id', 'country_id']]

    # Retrieve existing customers
    existing_customers = pd.read_sql("SELECT customer_id FROM customer_dimension", engine)

    # Filter out existing customers
    new_customers = customer_dim[~customer_dim['customer_id'].isin(existing_customers['customer_id'])]

    # Check if new_customers is empty before inserting
    if new_customers.empty:
        print("No new customers to load into customer_dimension.")
    else:
        # Insert new customers into the database
        for _, row in new_customers.iterrows():
            try:
                row.to_frame().T.to_sql('customer_dimension', engine, if_exists='append', index=False)
            except Exception as e:
                print(f"Error inserting customer_id {row['customer_id']}: {e}")
        print("New customers successfully loaded into customer_dimension.")
except Exception as e:
    print(f"Error during loading data into customer_dimension: {e}")

# Load product_dimension with deduplication
try:
    existing_products = pd.read_sql("SELECT stockcode FROM product_dimension", engine)
    product_dim = cleaned_data[['StockCode', 'Description', 'UnitPrice']].drop_duplicates().rename(columns={
        'StockCode': 'stockcode',
        'Description': 'description',
        'UnitPrice': 'unit_price'
    })
    product_dim = product_dim[~product_dim['stockcode'].isin(existing_products['stockcode'])]
    if not product_dim.empty:
        product_dim.to_sql('product_dimension', engine, if_exists='append', index=False)
        print("Products successfully loaded into product_dimension.")
    else:
        print("No new products to load into product_dimension.")
except Exception as e:
    print(f"Error during loading data into product_dimension: {e}")

# Load time_dimension with deduplication and filtering
try:
    existing_dates = pd.read_sql("SELECT DISTINCT invoice_date FROM time_dimension", engine)
    existing_dates['invoice_date'] = pd.to_datetime(existing_dates['invoice_date'])
    time_dim = cleaned_data[['InvoiceDate']].drop_duplicates().rename(columns={'InvoiceDate': 'invoice_date'})
    time_dim['invoice_date'] = pd.to_datetime(time_dim['invoice_date'])
    time_dim = time_dim[~time_dim['invoice_date'].isin(existing_dates['invoice_date'])]
    time_dim['year'] = time_dim['invoice_date'].dt.year
    time_dim['month'] = time_dim['invoice_date'].dt.month
    time_dim['day'] = time_dim['invoice_date'].dt.day
    if not time_dim.empty:
        time_dim.to_sql('time_dimension', engine, if_exists='append', index=False)
        print("Time data successfully loaded into time_dimension.")
    else:
        print("No new time data to load into time_dimension.")
except Exception as e:
    print(f"Error during loading data into time_dimension: {e}")



# Calculate total_price
cleaned_data['total_price'] = (cleaned_data['Quantity'] * cleaned_data['UnitPrice']).round(2)

# Load sales_fact with proper foreign key mappings
try:
    # Map product_id
    product_map = pd.read_sql("SELECT stockcode, product_id FROM product_dimension", engine)
    cleaned_data = cleaned_data.merge(product_map, how='left', left_on='StockCode', right_on='stockcode')

    # Map time_id
    time_map = pd.read_sql("SELECT invoice_date, time_id FROM time_dimension", engine)
    cleaned_data = cleaned_data.merge(time_map, how='left', left_on='InvoiceDate', right_on='invoice_date')

    # Map country_id
    country_map = pd.read_sql("SELECT country, country_id FROM country_dimension", engine)
    cleaned_data = cleaned_data.merge(country_map, how='left', left_on='Country', right_on='country')

    # Ensure all customer_id values exist in customer_dimension
    existing_customers = pd.read_sql("SELECT customer_id FROM customer_dimension", engine)
    cleaned_data = cleaned_data[cleaned_data['CustomerID'].isin(existing_customers['customer_id'])]

    # Prepare sales_fact by grouping data for the same customer purchasing different items
    sales_fact = cleaned_data.groupby(['CustomerID', 'country_id', 'time_id', 'InvoiceNo', 'product_id']) \
        .agg({
            'Quantity': 'sum',
            'total_price': 'sum'
        }).reset_index()

    sales_fact = sales_fact.rename(columns={
        'InvoiceNo': 'invoice_no',
        'CustomerID': 'customer_id',
        'Quantity': 'quantity',
        'total_price': 'total_price'
    })

    if not sales_fact.empty:
        sales_fact.to_sql('sales_fact', engine, if_exists='append', index=False)
        print("Sales data successfully loaded into sales_fact.")
    else:
        print("No new sales data to load into sales_fact.")
except Exception as e:
    print(f"Error during loading data into sales_fact: {e}")