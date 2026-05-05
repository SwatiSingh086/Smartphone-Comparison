import pandas as pd
import numpy as np
import re


def create_feature_importance():
    """
    Create a dictionary of feature importance weights
    These weights determine how much each feature influences recommendations
    """
    return {
        'camera_sentiment': 0.15,
        'battery_sentiment': 0.15,
        'performance_sentiment': 0.2,
        'display_sentiment': 0.1,
        'build_quality_sentiment': 0.1,
        'value_sentiment': 0.2,
        'price': 0.1  # Lower price is better
    }


def calculate_recommendation_score(analyzed_df, preferences=None):
    """
    Calculate recommendation scores for all phones based on sentiment analysis and price
    
    Parameters:
    - analyzed_df: DataFrame with sentiment analysis
    - preferences: Dictionary of user preferences for feature importance
    
    Returns:
    - DataFrame with recommendation scores
    """
    # Create a copy of the DataFrame for scoring
    df = analyzed_df.copy()
    
    # Normalize sentiment scores to 0-1 range (they range from -1 to 1 in VADER)
    for feature in ['camera_sentiment', 'battery_sentiment', 'performance_sentiment', 
                   'display_sentiment', 'build_quality_sentiment', 'value_sentiment']:
        df[f'{feature}_norm'] = (df[feature] + 1) / 2
    
    # Normalize price (lower is better)
    if 'Price' in df.columns:
        max_price = df['Price'].max()
        min_price = df['Price'].min()
        if max_price > min_price:
            df['price_norm'] = 1 - ((df['Price'] - min_price) / (max_price - min_price))
        else:
            df['price_norm'] = 1
    
    # Use default or custom feature importance weights
    weights = create_feature_importance()
    if preferences:
        for key, value in preferences.items():
            if key in weights:
                weights[key] = value
    
    # Calculate weighted score
    df['recommendation_score'] = (
        weights['camera_sentiment'] * df['camera_sentiment_norm'].fillna(0.5) + 
        weights['battery_sentiment'] * df['battery_sentiment_norm'].fillna(0.5) + 
        weights['performance_sentiment'] * df['performance_sentiment_norm'].fillna(0.5) + 
        weights['display_sentiment'] * df['display_sentiment_norm'].fillna(0.5) + 
        weights['build_quality_sentiment'] * df['build_quality_sentiment_norm'].fillna(0.5) + 
        weights['value_sentiment'] * df['value_sentiment_norm'].fillna(0.5) + 
        weights['price'] * df['price_norm'].fillna(0.5)
    )
    
    return df


def get_recommendations(analyzed_df, feature_preferences=None, num_recommendations=5, 
                       filters=None, exclude_phone_id=None):
    """
    Get personalized phone recommendations based on user preferences
    
    Parameters:
    - analyzed_df: DataFrame with sentiment analysis
    - feature_preferences: Dictionary of feature importance weights
    - num_recommendations: Number of recommendations to return
    - filters: Dictionary of filters to apply (price range, brand, etc.)
    - exclude_phone_id: Phone ID to exclude from recommendations (e.g., current phone)
    
    Returns:
    - DataFrame with recommended phones
    """
    # Apply any filters first
    filtered_df = analyzed_df.copy()
    
    if filters:
        if 'brands' in filters and filters['brands']:
            filtered_df = filtered_df[filtered_df['Brand'].isin(filters['brands'])]
        
        if 'min_price' in filters and filters['min_price'] is not None:
            filtered_df = filtered_df[filtered_df['Price'] >= filters['min_price']]
        
        if 'max_price' in filters and filters['max_price'] is not None:
            filtered_df = filtered_df[filtered_df['Price'] <= filters['max_price']]
        
        if 'min_ram' in filters and filters['min_ram'] is not None:
            # Extract numeric value from both the dataframe RAM values and the min_ram filter
            ram_values = filtered_df['RAM'].str.extract(r'(\d+)').astype(float)
            
            # Handle min_ram parsing safely
            min_ram_value = 0
            if isinstance(filters['min_ram'], str):
                match = re.search(r'(\d+)', filters['min_ram'])
                if match:
                    min_ram_value = float(match.group(1))
            else:
                min_ram_value = filters['min_ram']
                
            filtered_df = filtered_df[ram_values >= min_ram_value]
        
        if 'min_storage' in filters and filters['min_storage'] is not None:
            # Convert storage to numeric (TB to GB)
            storage_values = filtered_df['Storage'].apply(
                lambda x: float(x.replace('GB', '')) if isinstance(x, str) and 'GB' in x else x 
                #else float(x.replace('TB', '')) * 1024 if 'TB' in x else 0
            )
            # Extract numeric value from min_storage if it's a string
            min_storage_value = 0
            if isinstance(filters['min_storage'], str):
                if 'GB' in filters['min_storage']:
                    min_storage_value = float(filters['min_storage'].replace('GB', ''))
                elif 'TB' in filters['min_storage']:
                    min_storage_value = float(filters['min_storage'].replace('TB', '')) * 1024
                else:
                    # Try to extract a number if string doesn't have GB or TB
                    match = re.search(r'(\d+)', filters['min_storage'])
                    if match:
                        min_storage_value = float(match.group(1))
            else:
                min_storage_value = filters['min_storage']
                
            filtered_df = filtered_df[storage_values >= min_storage_value]
    
    # Exclude current phone if specified
    if exclude_phone_id:
        filtered_df = filtered_df[filtered_df['Product_ID'] != exclude_phone_id]
    
    # If no phones left after filtering, return empty DataFrame
    if filtered_df.empty:
        return pd.DataFrame()
    
    # Calculate recommendation scores
    scored_df = calculate_recommendation_score(filtered_df, feature_preferences)
    
    # Get unique phones with the highest score
    # First get unique product IDs
    unique_phones = scored_df.drop_duplicates('Product_ID')
    
    # Sort by recommendation score
    ranked_phones = unique_phones.sort_values('recommendation_score', ascending=False)
    
    # Return top N recommendations
    return ranked_phones.head(num_recommendations)


def get_similar_phones(phone_id, analyzed_df, num_recommendations=5):
    """
    Find phones similar to a given phone
    
    Parameters:
    - phone_id: ID of the phone to find similar phones for
    - analyzed_df: DataFrame with sentiment analysis
    - num_recommendations: Number of similar phones to return
    
    Returns:
    - DataFrame with similar phones
    """
    # Get the target phone data
    target_phone = analyzed_df[analyzed_df['Product_ID'] == phone_id]
    
    if target_phone.empty:
        return pd.DataFrame()
    
    # Extract key features of the target phone
    target_features = target_phone.iloc[0]
    target_brand = target_features['Brand']
    target_price = target_features['Price']
    
    # Get RAM as numeric
    target_ram = float(target_features['RAM'].replace('GB', ''))
    
    # Get storage as numeric (handle TB vs GB)
    if 'TB' in target_features['Storage']:
        target_storage = float(target_features['Storage'].replace('TB', '')) * 1024
    else:
        target_storage = float(target_features['Storage'].replace('GB', ''))
    
    # Create a copy of the DataFrame for similarity calculation
    df = analyzed_df.copy()
    
    # Exclude the target phone
    df = df[df['Product_ID'] != phone_id]
    
    # Calculate RAM and Storage numeric values
    df['RAM_Value'] = df['RAM'].str.extract(r'(\d+)').astype(float)
    df['Storage_Value'] = df['Storage'].apply(
        lambda x: float(x.replace('GB', '')) if 'GB' in x 
        else float(x.replace('TB', '')) * 1024 if 'TB' in x else 0
    )
    
    # Normalize price difference (lower is better)
    max_price = df['Price'].max()
    min_price = df['Price'].min()
    price_range = max_price - min_price
    
    if price_range > 0:
        df['price_diff'] = 1 - (abs(df['Price'] - target_price) / price_range)
    else:
        df['price_diff'] = 1
    
    # Normalize RAM and Storage differences
    max_ram = df['RAM_Value'].max()
    min_ram = df['RAM_Value'].min()
    ram_range = max_ram - min_ram
    
    if ram_range > 0:
        df['ram_diff'] = 1 - (abs(df['RAM_Value'] - target_ram) / ram_range)
    else:
        df['ram_diff'] = 1
    
    max_storage = df['Storage_Value'].max()
    min_storage = df['Storage_Value'].min()
    storage_range = max_storage - min_storage
    
    if storage_range > 0:
        df['storage_diff'] = 1 - (abs(df['Storage_Value'] - target_storage) / storage_range)
    else:
        df['storage_diff'] = 1
    
    # Calculate brand similarity (1 if same brand, 0.5 otherwise)
    df['brand_similarity'] = df['Brand'].apply(lambda x: 1 if x == target_brand else 0.5)
    
    # Calculate overall similarity score
    df['similarity_score'] = (
        0.3 * df['price_diff'] +
        0.25 * df['brand_similarity'] +
        0.2 * df['ram_diff'] +
        0.25 * df['storage_diff']
    )
    
    # Get unique phones with the highest similarity score
    unique_phones = df.drop_duplicates('Product_ID')
    
    # Sort by similarity score
    ranked_phones = unique_phones.sort_values('similarity_score', ascending=False)
    
    # Return top N similar phones
    return ranked_phones.head(num_recommendations)


def get_best_value_phones(analyzed_df, num_recommendations=5, min_sentiment_score=0.3):
    """
    Find phones with the best value (highest sentiment-to-price ratio)
    
    Parameters:
    - analyzed_df: DataFrame with sentiment analysis
    - num_recommendations: Number of best value phones to return
    - min_sentiment_score: Minimum overall sentiment score to consider
    
    Returns:
    - DataFrame with best value phones
    """
    # Create a copy of the DataFrame for value calculation
    df = analyzed_df.copy()
    
    # Filter by minimum sentiment score
    df = df[df['Sentiment_Score'] >= min_sentiment_score]
    
    # Calculate value score (sentiment per price unit)
    # Normalize price (lower is better)
    max_price = df['Price'].max()
    min_price = df['Price'].min()
    if max_price > min_price:
        df['price_norm'] = (max_price - df['Price']) / (max_price - min_price)
    else:
        df['price_norm'] = 1
    
    # Normalize sentiment (higher is better)
    df['sentiment_norm'] = (df['Sentiment_Score'] + 1) / 2
    
    # Calculate value score (weighted combination of price and sentiment)
    df['value_score'] = (0.5 * df['sentiment_norm']) + (0.5 * df['price_norm'])
    
    # Get unique phones with the highest value score
    unique_phones = df.drop_duplicates('Product_ID')
    
    # Sort by value score
    ranked_phones = unique_phones.sort_values('value_score', ascending=False)
    
    # Return top N best value phones
    return ranked_phones.head(num_recommendations)
