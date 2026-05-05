import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

# Import custom modules
import data_processor
import sentiment_analyzer
import recommendation_engine
import visualization

# Set page config
st.set_page_config(
    page_title="Smartphone Analysis Platform",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS (minimal and using Streamlit's native styling)
st.markdown("""
<style>
    .header-text {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .subheader-text {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading to improve performance
@st.cache_data
def load_and_process_data():
    """Load and process data with caching"""
    # Load data from CSV files
    flipkart_df, amazon_df, cashify_df = data_processor.load_data()
    
    # Combine data for analysis
    combined_df = data_processor.get_combined_data(flipkart_df, amazon_df, cashify_df)
    
    # Analyze sentiment in reviews
    analyzed_df = sentiment_analyzer.analyze_reviews(combined_df)
    
    return flipkart_df, amazon_df, cashify_df, combined_df, analyzed_df

# Main app layout
def main():
    st.markdown("<div class='header-text'>📱 SmartPhone Analysis Platform</div>", unsafe_allow_html=True)
    st.markdown("Analyze sentiment, compare prices, and get personalized smartphone recommendations")
    
    # Load and process data
    with st.spinner("Loading data..."):
        flipkart_df, amazon_df, cashify_df, combined_df, analyzed_df = load_and_process_data()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Select a page:", 
                               ["Product Comparison", "Sentiment Analysis", 
                                "Price Analysis", "Recommendations"])
    
    # PRODUCT COMPARISON PAGE
    if app_mode == "Product Comparison":
        st.markdown("<div class='subheader-text'>📊 Product Comparison</div>", unsafe_allow_html=True)
        st.write("Compare smartphones across different e-commerce platforms")
        
        # Create a unique list of brands
        all_brands = sorted(combined_df['Brand'].unique())
        
        # Sidebar filters
        st.sidebar.title("Filters")
        selected_brands = st.sidebar.multiselect("Select Brands", all_brands, default=all_brands[:5])
        
        # Price range slider
        min_price = int(combined_df['Price'].min())
        max_price = int(combined_df['Price'].max())
        price_range = st.sidebar.slider("Price Range (₹)", min_price, max_price, (min_price, max_price))
        
        # Filter data based on selection
        filtered_df = combined_df[
            (combined_df['Brand'].isin(selected_brands)) &
            (combined_df['Price'] >= price_range[0]) &
            (combined_df['Price'] <= price_range[1])
        ]
        
        # Get unique phone models after filtering
        unique_phones = data_processor.get_unique_phones(filtered_df)
        
        if unique_phones.empty:
            st.warning("No phones match your filter criteria. Please adjust the filters.")
        else:
            # Create a dropdown to select phones
            selected_phone_option = st.selectbox(
                "Select a phone to compare:",
                unique_phones['Brand'] + ' ' + unique_phones['Model'] + ' (' + unique_phones['RAM'] + ', ' + unique_phones['Storage'] + ')'
            )
            
            # Get the product ID from the selected option
            selected_index = unique_phones.index[
                (unique_phones['Brand'] + ' ' + unique_phones['Model'] + ' (' + unique_phones['RAM'] + ', ' + unique_phones['Storage'] + ')') == selected_phone_option
            ].tolist()[0]
            selected_phone_id = unique_phones.loc[selected_index, 'Product_ID']
            
            # Get price comparison data
            price_data = data_processor.get_price_comparison(
                selected_phone_id, flipkart_df, amazon_df, cashify_df
            )
            
            # Display comparison results
            if price_data:
                # Create and display price comparison chart
                price_chart = visualization.create_price_comparison_chart(price_data)
                if price_chart:
                    st.plotly_chart(price_chart, use_container_width=True)
                
                # Display platform with lowest price
                min_price_platform = min(price_data, key=lambda x: x['current_price'])
                
                st.success(f"**Best Deal:** {min_price_platform['platform']} offers the lowest price at "
                          f"₹{min_price_platform['current_price']:,.2f} "
                          f"({min_price_platform['discount']:.1f}% off)")
                
                # Create columns for detailed comparison
                cols = st.columns(len(price_data))
                
                for i, platform_data in enumerate(price_data):
                    with cols[i]:
                        st.subheader(platform_data['platform'])
                        st.write(f"**Original Price:** ₹{platform_data['original_price']:,.2f}")
                        st.write(f"**Current Price:** ₹{platform_data['current_price']:,.2f}")
                        st.write(f"**Discount:** {platform_data['discount']:.1f}%")
                
                # Display detailed phone information
                st.subheader("Phone Specifications")
                phone_details = data_processor.get_phone_details(selected_phone_id, combined_df)
                
                if not phone_details.empty:
                    specs = phone_details.iloc[0]
                    
                    spec_cols = st.columns(3)
                    with spec_cols[0]:
                        st.write(f"**Brand:** {specs['Brand']}")
                        st.write(f"**Model:** {specs['Model']}")
                        st.write(f"**RAM:** {specs['RAM']}")
                        st.write(f"**Storage:** {specs['Storage']}")
                    
                    with spec_cols[1]:
                        st.write(f"**Screen Size:** {specs['Screen_Size']}")
                        st.write(f"**Battery:** {specs['Battery']}")
                        st.write(f"**Color:** {specs['Color']}")
                    
                    with spec_cols[2]:
                        st.write(f"**Main Camera:** {specs['Main_Camera']}")
                        st.write(f"**Charging:** {specs['Charging']}")
                
                # Show similar phones
                st.subheader("Similar Phones")
                similar_phones = recommendation_engine.get_similar_phones(
                    selected_phone_id, analyzed_df, num_recommendations=5
                )
                
                if not similar_phones.empty:
                    similar_cols = st.columns(len(similar_phones))
                    
                    for i, (_, phone) in enumerate(similar_phones.iterrows()):
                        with similar_cols[i]:
                            # Using Brand and Model instead of Product_Name
                            st.write(f"**{phone['Brand']} {phone['Model']}**")
                            st.write(f"RAM: {phone['RAM']}")
                            st.write(f"Storage: {phone['Storage']}")
                            st.write(f"Price: ₹{phone['Price']:,.2f}")
                else:
                    st.info("No similar phones found.")
            else:
                st.warning("Price comparison data not available for this phone.")
    
    # SENTIMENT ANALYSIS PAGE
    elif app_mode == "Sentiment Analysis":
        st.markdown("<div class='subheader-text'>🔍 Sentiment Analysis</div>", unsafe_allow_html=True)
        st.write("Analyze customer reviews and sentiment for smartphone features")
        
        # Create a unique list of brands
        all_brands = sorted(combined_df['Brand'].unique())
        
        # Sidebar filters
        st.sidebar.title("Filters")
        selected_brands = st.sidebar.multiselect("Select Brands", all_brands, default=all_brands[:5])
        
        # Filter data based on selection
        filtered_df = combined_df[combined_df['Brand'].isin(selected_brands)]
        
        # Get unique phone models after filtering
        unique_phones = data_processor.get_unique_phones(filtered_df)
        
        if unique_phones.empty:
            st.warning("No phones match your filter criteria. Please adjust the filters.")
        else:
            # Create a dropdown to select phones
            selected_phone_option = st.selectbox(
                "Select a phone to analyze:",
                unique_phones['Brand'] + ' ' + unique_phones['Model'] + ' (' + unique_phones['RAM'] + ', ' + unique_phones['Storage'] + ')'
            )
            
            # Get the product ID from the selected option
            selected_index = unique_phones.index[
                (unique_phones['Brand'] + ' ' + unique_phones['Model'] + ' (' + unique_phones['RAM'] + ', ' + unique_phones['Storage'] + ')') == selected_phone_option
            ].tolist()[0]
            selected_phone_id = unique_phones.loc[selected_index, 'Product_ID']
            
            # Get sentiment summary data
            sentiment_data = sentiment_analyzer.get_phone_sentiment_summary(
                selected_phone_id, analyzed_df
            )
            
            if sentiment_data:
                # Create dashboard layout with columns
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Display sentiment distribution chart
                    sentiment_dist_chart = visualization.create_sentiment_distribution_chart(sentiment_data)
                    if sentiment_dist_chart:
                        st.plotly_chart(sentiment_dist_chart, use_container_width=True)
                
                with col2:
                    # Display feature sentiment radar chart
                    radar_chart = visualization.create_sentiment_radar_chart(sentiment_data)
                    if radar_chart:
                        st.plotly_chart(radar_chart, use_container_width=True)
                
                # Display feature analysis in detail
                st.subheader("Feature Analysis")
                feature_cols = st.columns(3)
                
                features = sentiment_data['features']
                feature_names = {
                    'camera': '📷 Camera',
                    'battery': '🔋 Battery',
                    'performance': '⚡ Performance',
                    'display': '📱 Display',
                    'build_quality': '🛠️ Build Quality',
                    'value': '💰 Value for Money'
                }
                
                for i, (feature, score) in enumerate(features.items()):
                    with feature_cols[i % 3]:
                        sentiment_category = "Positive" if score > 0.05 else ("Negative" if score < -0.05 else "Neutral")
                        sentiment_color = "#3CB371" if score > 0.05 else ("#FF6347" if score < -0.05 else "#FFD700")
                        
                        st.markdown(f"**{feature_names.get(feature, feature)}**")
                        st.progress((score + 1) / 2)  # Convert -1 to 1 scale to 0 to 1 for progress bar
                        st.markdown(f"<span style='color:{sentiment_color}'>{sentiment_category}</span> (Score: {score:.2f})", unsafe_allow_html=True)
                
                # Display sample reviews
                st.subheader("Sample Reviews")
                review_tabs = st.tabs(["Positive Reviews", "Negative Reviews"])
                
                with review_tabs[0]:
                    if sentiment_data['positive_reviews']:
                        for i, review in enumerate(sentiment_data['positive_reviews'][:5], 1):
                            st.markdown(f"**Review {i}:**")
                            st.info(review)
                    else:
                        st.info("No positive reviews found.")
                
                with review_tabs[1]:
                    if sentiment_data['negative_reviews']:
                        for i, review in enumerate(sentiment_data['negative_reviews'][:5], 1):
                            st.markdown(f"**Review {i}:**")
                            st.error(review)
                    else:
                        st.info("No negative reviews found.")
            else:
                st.warning("Sentiment analysis data not available for this phone.")
    
    # PRICE ANALYSIS PAGE
    elif app_mode == "Price Analysis":
        st.markdown("<div class='subheader-text'>💰 Price Analysis</div>", unsafe_allow_html=True)
        st.write("Analyze price trends and find the best time to buy")
        
        # Create a unique list of brands
        all_brands = sorted(combined_df['Brand'].unique())
        
        # Sidebar filters
        st.sidebar.title("Filters")
        selected_brands = st.sidebar.multiselect("Select Brands", all_brands, default=all_brands[:5])
        
        # Price range slider
        min_price = int(combined_df['Price'].min())
        max_price = int(combined_df['Price'].max())
        price_range = st.sidebar.slider("Price Range (₹)", min_price, max_price, (min_price, max_price))
        
        # Filter data based on selection
        filtered_df = combined_df[
            (combined_df['Brand'].isin(selected_brands)) &
            (combined_df['Price'] >= price_range[0]) &
            (combined_df['Price'] <= price_range[1])
        ]
        
        # Platform discount comparison
        st.subheader("Discount Comparison Across Platforms")
        platform_discount_chart = visualization.create_platform_discount_comparison(combined_df)
        st.plotly_chart(platform_discount_chart, use_container_width=True)
        
        # Get unique phone models after filtering
        unique_phones = data_processor.get_unique_phones(filtered_df)
        
        if unique_phones.empty:
            st.warning("No phones match your filter criteria. Please adjust the filters.")
        else:
            # Create a dropdown to select phones
            selected_phone_option = st.selectbox(
                "Select a phone to analyze price trends:",
                unique_phones['Brand'] + ' ' + unique_phones['Model'] + ' (' + unique_phones['RAM'] + ', ' + unique_phones['Storage'] + ')'
            )
            
            # Get the product ID from the selected option
            selected_index = unique_phones.index[
                (unique_phones['Brand'] + ' ' + unique_phones['Model'] + ' (' + unique_phones['RAM'] + ', ' + unique_phones['Storage'] + ')') == selected_phone_option
            ].tolist()[0]
            selected_phone_id = unique_phones.loc[selected_index, 'Product_ID']
            
            # Get price trend data
            trend_data = data_processor.generate_price_trends(selected_phone_id, combined_df)
            
            if trend_data:
                # Display price trend chart
                price_trend_chart = visualization.create_price_trend_chart(trend_data)
                if price_trend_chart:
                    st.plotly_chart(price_trend_chart, use_container_width=True)
                
                # Show best time to buy recommendation
                st.success(f"**Best Time to Buy:** {trend_data['best_time']} (Expected Price: ₹{trend_data['expected_price']:,.2f})")
                
                # Add price comparison across platforms
                price_data = data_processor.get_price_comparison(
                    selected_phone_id, flipkart_df, amazon_df, cashify_df
                )
                
                if price_data:
                    # Create columns for current prices
                    st.subheader("Current Prices")
                    price_cols = st.columns(len(price_data))
                    
                    for i, platform_data in enumerate(price_data):
                        with price_cols[i]:
                            st.metric(
                                platform_data['platform'],
                                f"₹{platform_data['current_price']:,.2f}",
                                f"-{platform_data['discount']:.1f}%",
                                delta_color="normal"
                            )
                
                # Add buying advice
                st.subheader("Buying Advice")
                
                # Simple logic for buying advice based on price trend
                historical = trend_data['historical']
                current_price = historical[-1][1]  # Last price in historical data
                min_future_price = min([price for _, price in trend_data['future']])
                price_difference = ((current_price - min_future_price) / current_price) * 100
                
                if price_difference > 5:
                    st.info(f"📉 **Wait for Price Drop:** Prices are expected to drop by {price_difference:.1f}% in the coming days. Consider waiting until {trend_data['best_time']} for the best deal.")
                elif price_difference > 2:
                    st.info(f"⏳ **Consider Waiting:** A slight price drop of {price_difference:.1f}% is expected. If not urgent, wait until {trend_data['best_time']}.")
                else:
                    st.info("✅ **Good Time to Buy:** Prices are stable or may slightly increase. Current offers are good value.")
            else:
                st.warning("Price trend data not available for this phone.")
    
    # RECOMMENDATIONS PAGE
    elif app_mode == "Recommendations":
        st.markdown("<div class='subheader-text'>🎯 Personalized Recommendations</div>", unsafe_allow_html=True)
        st.write("Get smartphone recommendations based on your preferences")
        
        # Sidebar for preferences
        st.sidebar.title("Your Preferences")
        
        # User preferences for features
        st.sidebar.subheader("Feature Importance")
        camera_weight = st.sidebar.slider("Camera", 0.0, 1.0, 0.15, 0.05)
        battery_weight = st.sidebar.slider("Battery Life", 0.0, 1.0, 0.15, 0.05)
        performance_weight = st.sidebar.slider("Performance", 0.0, 1.0, 0.2, 0.05)
        display_weight = st.sidebar.slider("Display", 0.0, 1.0, 0.1, 0.05)
        build_quality_weight = st.sidebar.slider("Build Quality", 0.0, 1.0, 0.1, 0.05)
        value_weight = st.sidebar.slider("Value for Money", 0.0, 1.0, 0.2, 0.05)
        price_weight = st.sidebar.slider("Price (Lower is Better)", 0.0, 1.0, 0.1, 0.05)
        
        # Normalize weights to sum to 1
        total_weight = camera_weight + battery_weight + performance_weight + display_weight + build_quality_weight + value_weight + price_weight
        
        camera_weight = camera_weight / total_weight
        battery_weight = battery_weight / total_weight
        performance_weight = performance_weight / total_weight
        display_weight = display_weight / total_weight
        build_quality_weight = build_quality_weight / total_weight
        value_weight = value_weight / total_weight
        price_weight = price_weight / total_weight
        
        # Create feature preferences dictionary
        feature_preferences = {
            'camera': camera_weight,
            'battery': battery_weight,
            'performance': performance_weight,
            'display': display_weight, 
            'build_quality': build_quality_weight,
            'value': value_weight,
            'price': price_weight
        }
        
        # Feature importance chart
        feature_chart = visualization.create_feature_importance_chart(feature_preferences)
        st.plotly_chart(feature_chart, use_container_width=True)
        
        # Additional filters
        st.sidebar.subheader("Additional Filters")
        
        # Brand preference
        all_brands = sorted(combined_df['Brand'].unique())
        selected_brands = st.sidebar.multiselect("Preferred Brands", all_brands, default=all_brands[:3])
        
        # Price range
        min_price = int(combined_df['Price'].min())
        max_price = int(combined_df['Price'].max())
        price_range = st.sidebar.slider("Price Range (₹)", min_price, max_price, (min_price, max_price))
        
        # RAM and Storage preference
        all_ram = sorted(combined_df['RAM'].unique())
        selected_ram = st.sidebar.multiselect("Minimum RAM", all_ram, default=all_ram[len(all_ram)//2:])
        
        all_storage = sorted(combined_df['Storage'].unique())
        selected_storage = st.sidebar.multiselect("Minimum Storage", all_storage, default=all_storage[len(all_storage)//2:])
        
        # Create filter dictionary
        filters = {
            'brands': selected_brands,
            'min_price': price_range[0],
            'max_price': price_range[1],
            'min_ram': selected_ram[0] if selected_ram else None,
            'min_storage': selected_storage[0] if selected_storage else None
        }
        
        # Get personalized recommendations
        recommended_phones = recommendation_engine.get_recommendations(
            analyzed_df, feature_preferences, num_recommendations=10, filters=filters
        )
        
        # Display recommendations
        if not recommended_phones.empty:
            st.subheader("Your Personalized Recommendations")
            
            # Display recommendation chart
            rec_chart = visualization.create_recommendation_bar_chart(recommended_phones)
            if rec_chart:
                st.plotly_chart(rec_chart, use_container_width=True)
            
            # Display detailed recommendations
            for i, (_, phone) in enumerate(recommended_phones.iterrows()):
                with st.expander(f"{i+1}. {phone['Brand']} {phone['Model']} ({phone['RAM']}, {phone['Storage']})"):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.write(f"**Price:** ₹{phone['Price']:,.2f}")
                        st.write(f"**Overall Score:** {phone['recommendation_score']:.2f}")
                        st.write(f"**Battery:** {phone['Battery']}")
                        st.write(f"**Camera:** {phone['Main_Camera']}")
                    
                    with col2:
                        # Feature scores
                        st.write("**Feature Sentiment Scores:**")
                        feature_scores = {
                            'Camera': phone.get('camera_sentiment', 0),
                            'Battery': phone.get('battery_sentiment', 0),
                            'Performance': phone.get('performance_sentiment', 0),
                            'Display': phone.get('display_sentiment', 0),
                            'Build Quality': phone.get('build_quality_sentiment', 0),
                            'Value': phone.get('value_sentiment', 0)
                        }
                        
                        feature_cols = st.columns(3)
                        for j, (feature, score) in enumerate(feature_scores.items()):
                            with feature_cols[j % 3]:
                                sentiment_color = "#3CB371" if score > 0.05 else ("#FF6347" if score < -0.05 else "#FFD700")
                                st.markdown(f"**{feature}:**")
                                st.progress((score + 1) / 2)  # Convert -1 to 1 scale to 0 to 1 for progress bar
                                st.markdown(f"<span style='color:{sentiment_color}'>{score:.2f}</span>", unsafe_allow_html=True)
        else:
            st.warning("No recommendations match your criteria. Try adjusting your filters.")
        
        # Show best value phones
        st.subheader("Best Value Phones")
        best_value_phones = recommendation_engine.get_best_value_phones(analyzed_df, num_recommendations=5)
        
        if not best_value_phones.empty:
            best_value_cols = st.columns(len(best_value_phones))
            
            for i, (_, phone) in enumerate(best_value_phones.iterrows()):
                with best_value_cols[i]:
                    # Using Brand and Model instead of Product_Name
                    st.subheader(f"{phone['Brand']} {phone['Model']}")
                    st.write(f"**Price:** ₹{phone['Price']:,.2f}")
                    st.write(f"**Value Score:** {phone['value_score']:.2f}")
                    st.write(f"**RAM:** {phone['RAM']}")
                    st.write(f"**Storage:** {phone['Storage']}")
        else:
            st.info("No best value phones found.")

# Run the app
if __name__ == "__main__":
    main()