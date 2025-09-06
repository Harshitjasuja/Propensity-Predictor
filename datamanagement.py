import pandas as pd
import os

# Path to your data folder (all CSV files should be inside 'data/')
DATA_FOLDER = "data"

def get_file_paths():
    """
    Returns a dictionary mapping dataset names to their file paths inside the data folder.
    """
    return {
        'orders': os.path.join(DATA_FOLDER, 'olist_orders_dataset.csv'),
        'customers': os.path.join(DATA_FOLDER, 'olist_customers_dataset.csv'),
        'order_items': os.path.join(DATA_FOLDER, 'olist_order_items_dataset.csv'),
        'order_payments': os.path.join(DATA_FOLDER, 'olist_order_payments_dataset.csv'),
        'products': os.path.join(DATA_FOLDER, 'olist_products_dataset.csv'),
        'reviews': os.path.join(DATA_FOLDER, 'olist_order_reviews_dataset.csv'),
        'sellers': os.path.join(DATA_FOLDER, 'olist_sellers_dataset.csv'),
        'geolocation': os.path.join(DATA_FOLDER, 'olist_geolocation_dataset.csv'),
        'translation': os.path.join(DATA_FOLDER, 'product_category_name_translation.csv')
    }

def load_data(path_dict):
    """
    Loads all necessary datasets as pandas DataFrames.
    Returns a dictionary {dataset_name: DataFrame}.
    """
    data = {}
    for name, path in path_dict.items():
        try:
            data[name] = pd.read_csv(path)
        except FileNotFoundError:
            print(f"Error: Could not find '{path}'. Check that the file exists in the 'data' folder.")
    return data

def merge_order_data(data):
    """
    Merges key datasets for modeling.
    Returns a merged DataFrame.
    """
    # Merge orders with customers (on 'customer_id')
    df = pd.merge(data['orders'], data['customers'], on='customer_id', how='left')

    # Merge order_items (on 'order_id')
    df = pd.merge(df, data['order_items'], on='order_id', how='left')

    # Merge order_payments (on 'order_id')
    df = pd.merge(df, data['order_payments'], on='order_id', how='left')

    # Merge products (on 'product_id')
    df = pd.merge(df, data['products'], on='product_id', how='left')

    return df

if __name__ == "__main__":
    # Get file paths
    paths = get_file_paths()
    # Load all data
    data = load_data(paths)
    # Merge datasets for modeling
    merged_df = merge_order_data(data)
    # Save merged DataFrame for later use (optional)
    merged_df.to_csv(os.path.join(DATA_FOLDER, "merged_orders.csv"), index=False)
    print("Merged data shape:", merged_df.shape)
    print(merged_df.head())