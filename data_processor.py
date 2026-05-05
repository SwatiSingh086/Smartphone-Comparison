import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def load_data():
    """
    Load data from CSV files and perform initial preprocessing
    """
    # Load data from CSV files
    flipkart_df = pd.read_csv('attached_assets/flipkart_phones.csv')
    amazon_df = pd.read_csv('attached_assets/amazon_phones.csv')
    cashify_df = pd.read_csv('attached_assets/cashify_phones.csv')
    
    # Add source column to each dataframe
    flipkart_df['Source'] = 'Flipkart'
    amazon_df['Source'] = 'Amazon'
    cashify_df['Source'] = 'Cashify'
    
    # Basic data cleaning
    for df in [flipkart_df, amazon_df, cashify_df]:
        # Remove any duplicate entries
        df.drop_duplicates(subset=['Brand', 'Model', 'RAM', 'Storage', 'Color'], inplace=True)
        
        # Convert price columns to numeric
        df['Original_Price'] = pd.to_numeric(df['Original_Price'], errors='coerce')
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df['Discount_Percentage'] = pd.to_numeric(df['Discount_Percentage'], errors='coerce')
        
        # Extract battery capacity as numeric (remove mAh)
        df['Battery_Value'] = df['Battery'].str.extract(r'(\d+)').astype(float)
        
        # Extract screen size as numeric
        df['Screen_Size_Value'] = pd.to_numeric(df['Screen_Size'], errors='coerce')
        
        # Extract camera megapixels as numeric
        df['Camera_MP'] = df['Main_Camera'].str.extract(r'(\d+)').astype(float)
        
        # Extract charging power as numeric (remove W)
        df['Charging_W'] = df['Charging'].str.extract(r'(\d+)').astype(float)
        
        # Add unique product identifier
        df['Product_ID'] = df['Brand'] + '_' + df['Model'] + '_' + df['RAM'] + '_' + df['Storage']

    return flipkart_df, amazon_df, cashify_df


def get_combined_data(flipkart_df, amazon_df, cashify_df):
    """
    Combine data from all sources for analysis
    """
    combined_df = pd.concat([flipkart_df, amazon_df, cashify_df], ignore_index=True)
    return combined_df


def get_unique_phones(combined_df):
    """
    Get a list of unique phone models across all platforms
    """
    unique_phones = combined_df[['Brand', 'Model', 'RAM', 'Storage', 'Product_ID']].drop_duplicates()
    return unique_phones


def get_phone_details(phone_id, combined_df):
    """
    Get detailed information about a specific phone
    """
    phone_data = combined_df[combined_df['Product_ID'] == phone_id]
    return phone_data


def get_price_comparison(phone_id, flipkart_df, amazon_df, cashify_df):
    """
    Compare prices across platforms for a specific phone
    """
    prices = []
    
    flipkart_match = flipkart_df[flipkart_df['Product_ID'] == phone_id]
    if not flipkart_match.empty:
        prices.append({
            'platform': 'Flipkart',
            'original_price': flipkart_match['Original_Price'].values[0],
            'current_price': flipkart_match['Price'].values[0],
            'discount': flipkart_match['Discount_Percentage'].values[0]
        })
    
    amazon_match = amazon_df[amazon_df['Product_ID'] == phone_id]
    if not amazon_match.empty:
        prices.append({
            'platform': 'Amazon',
            'original_price': amazon_match['Original_Price'].values[0],
            'current_price': amazon_match['Price'].values[0],
            'discount': amazon_match['Discount_Percentage'].values[0]
        })
    
    cashify_match = cashify_df[cashify_df['Product_ID'] == phone_id]
    if not cashify_match.empty:
        prices.append({
            'platform': 'Cashify',
            'original_price': cashify_match['Original_Price'].values[0],
            'current_price': cashify_match['Price'].values[0],
            'discount': cashify_match['Discount_Percentage'].values[0]
        })
    
    return prices


def generate_price_trends(phone_id, combined_df):
    """
    Generate simulated price trends based on current prices and discounts
    """
    phone_data = combined_df[combined_df['Product_ID'] == phone_id]
    if phone_data.empty:
        return None
    
    # Get average price and discount percentage
    avg_price = phone_data['Price'].mean()
    avg_discount = phone_data['Discount_Percentage'].mean()
    max_discount = phone_data['Discount_Percentage'].max()
    
    # Generate simulated price trend for the last 30 days
    today = datetime.now()
    dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30, 0, -1)]
    
    # Create price variations (we're simulating this based on the discount data we have)
    prices = []
    for i in range(30):
        # Create some variation in the price trend
        variation = np.sin(i/5) * (avg_discount/100) * avg_price * 0.4
        price = avg_price + variation
        prices.append(price)
    
    # Predict future prices for next 7 days
    future_dates = [(today + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 8)]
    
    # Simple prediction based on trend
    future_trend = []
    for i in range(7):
        # Assume prices might drop further based on max discount
        potential_drop = (max_discount/100) * avg_price * 0.8
        future_variation = np.sin((i+30)/5) * (avg_discount/100) * avg_price * 0.4
        future_price = avg_price + future_variation - (i * potential_drop / 14)
        future_trend.append(future_price)
    
    # Determine best time to buy (simplistic approach)
    min_future_price_index = future_trend.index(min(future_trend))
    best_time_to_buy = future_dates[min_future_price_index]
    
    return {
        'historical': list(zip(dates, prices)),
        'future': list(zip(future_dates, future_trend)),
        'best_time': best_time_to_buy,
        'expected_price': min(future_trend)
    }


def filter_phones(combined_df, brands=None, min_price=None, max_price=None, 
                 min_ram=None, max_ram=None, min_storage=None, max_storage=None,
                 min_camera=None, min_battery=None):
    """
    Filter phones based on user preferences
    """
    filtered_df = combined_df.copy()
    
    # Apply filters
    if brands:
        filtered_df = filtered_df[filtered_df['Brand'].isin(brands)]
    
    if min_price is not None:
        filtered_df = filtered_df[filtered_df['Price'] >= min_price]
    
    if max_price is not None:
        filtered_df = filtered_df[filtered_df['Price'] <= max_price]
    
    # Extract RAM as numeric (remove GB)
    filtered_df['RAM_GB'] = filtered_df['RAM'].str.extract(r'(\d+)').astype(float)
    
    if min_ram is not None:
        filtered_df = filtered_df[filtered_df['RAM_GB'] >= min_ram]
    
    if max_ram is not None:
        filtered_df = filtered_df[filtered_df['RAM_GB'] <= max_ram]
    
    # Extract Storage as numeric (remove GB/TB)
    filtered_df['Storage_Value'] = filtered_df['Storage'].apply(
        lambda x: float(x.replace('GB', '')) if 'GB' in x 
        else float(x.replace('TB', '')) * 1024 if 'TB' in x else 0
    )
    
    if min_storage is not None:
        filtered_df = filtered_df[filtered_df['Storage_Value'] >= min_storage]
    
    if max_storage is not None:
        filtered_df = filtered_df[filtered_df['Storage_Value'] <= max_storage]
    
    if min_camera is not None:
        filtered_df = filtered_df[filtered_df['Camera_MP'] >= min_camera]
    
    if min_battery is not None:
        filtered_df = filtered_df[filtered_df['Battery_Value'] >= min_battery]
    
    return filtered_df
