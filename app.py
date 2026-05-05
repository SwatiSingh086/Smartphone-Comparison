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
    with st.spinner("Loading data and running sentiment analysis — "
                    "first run downloads DistilBERT (~67 MB) and builds a disk cache. "
                    "Subsequent startups load instantly from cache …"):
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
            # Create a radio option to choose comparison mode
            comparison_mode = st.radio(
                "Select comparison mode:",
                ["Single Phone Analysis", "Compare Two Phones"]
            )

            if comparison_mode == "Single Phone Analysis":
                # Create a dropdown to select phones
                selected_phone_option = st.selectbox(
                    "Select a phone to analyze:",
                    unique_phones['Brand'] + ' ' + unique_phones['Model'] + ' (' + unique_phones['RAM'] + ', ' +
                    unique_phones['Storage'] + ')'
                )

                # Get the product ID from the selected option
                selected_index = unique_phones.index[
                    (unique_phones['Brand'] + ' ' + unique_phones['Model'] + ' (' + unique_phones['RAM'] + ', ' +
                     unique_phones['Storage'] + ')') == selected_phone_option
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

            else:  # Compare Two Phones mode
                # Create two columns for phone selection
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Phone 1")
                    phone1_option = st.selectbox(
                        "Select first phone:",
                        unique_phones['Brand'] + ' ' + unique_phones['Model'] + ' (' + unique_phones['RAM'] + ', ' +
                        unique_phones['Storage'] + ')',
                        key="phone1"
                    )
                    # Get the product ID for phone 1
                    phone1_index = unique_phones.index[
                        (unique_phones['Brand'] + ' ' + unique_phones['Model'] + ' (' + unique_phones['RAM'] + ', ' +
                         unique_phones['Storage'] + ')') == phone1_option
                        ].tolist()[0]
                    phone1_id = unique_phones.loc[phone1_index, 'Product_ID']

                with col2:
                    st.subheader("Phone 2")
                    # Filter out the first phone from options for the second dropdown
                    remaining_phones = unique_phones[unique_phones['Product_ID'] != phone1_id]
                    phone2_option = st.selectbox(
                        "Select second phone:",
                        remaining_phones['Brand'] + ' ' + remaining_phones['Model'] + ' (' + remaining_phones[
                            'RAM'] + ', ' + remaining_phones['Storage'] + ')',
                        key="phone2"
                    )
                    # Get the product ID for phone 2
                    phone2_index = remaining_phones.index[
                        (remaining_phones['Brand'] + ' ' + remaining_phones['Model'] + ' (' + remaining_phones[
                            'RAM'] + ', ' + remaining_phones['Storage'] + ')') == phone2_option
                        ].tolist()[0]
                    phone2_id = remaining_phones.loc[phone2_index, 'Product_ID']

                # Get details for both phones
                phone1_details = data_processor.get_phone_details(phone1_id, combined_df).iloc[0]
                phone2_details = data_processor.get_phone_details(phone2_id, combined_df).iloc[0]

                # Get price data for both phones
                phone1_price_data = data_processor.get_price_comparison(phone1_id, flipkart_df, amazon_df, cashify_df)
                phone2_price_data = data_processor.get_price_comparison(phone2_id, flipkart_df, amazon_df, cashify_df)

                # Display side-by-side comparison
                st.subheader("Price Comparison")

                # Calculate the lowest prices for each phone
                phone1_min_price = min(phone1_price_data, key=lambda x: x['current_price'])
                phone2_min_price = min(phone2_price_data, key=lambda x: x['current_price'])

                price_cols = st.columns(2)
                with price_cols[0]:
                    st.write(f"**Best Price for {phone1_details['Brand']} {phone1_details['Model']}:**")
                    st.write(f"₹{phone1_min_price['current_price']:,.2f} on {phone1_min_price['platform']}")
                    st.write(f"Discount: {phone1_min_price['discount']:.1f}%")

                with price_cols[1]:
                    st.write(f"**Best Price for {phone2_details['Brand']} {phone2_details['Model']}:**")
                    st.write(f"₹{phone2_min_price['current_price']:,.2f} on {phone2_min_price['platform']}")
                    st.write(f"Discount: {phone2_min_price['discount']:.1f}%")

                # Specifications comparison
                st.subheader("Specifications Comparison")

                # Create a comparison table without styling first
                # Ensure we have pandas available within this scope
                import pandas as pd

                specs_comparison = pd.DataFrame({
                    'Specification': ['Brand', 'Model', 'RAM', 'Storage', 'Screen Size', 'Battery', 'Main Camera',
                                      'Charging', 'Price'],
                    'Phone 1': [
                        phone1_details['Brand'],
                        phone1_details['Model'],
                        phone1_details['RAM'],
                        phone1_details['Storage'],
                        phone1_details['Screen_Size'],
                        phone1_details['Battery'],
                        phone1_details['Main_Camera'],
                        phone1_details['Charging'],
                        f"₹{phone1_min_price['current_price']:,.2f}"
                    ],
                    'Phone 2': [
                        phone2_details['Brand'],
                        phone2_details['Model'],
                        phone2_details['RAM'],
                        phone2_details['Storage'],
                        phone2_details['Screen_Size'],
                        phone2_details['Battery'],
                        phone2_details['Main_Camera'],
                        phone2_details['Charging'],
                        f"₹{phone2_min_price['current_price']:,.2f}"
                    ]
                })

                # Display the comparison table
                st.dataframe(specs_comparison, use_container_width=True)

                # Value for money analysis
                price_diff = abs(phone1_min_price['current_price'] - phone2_min_price['current_price'])
                price_diff_percent = price_diff / max(phone1_min_price['current_price'],
                                                      phone2_min_price['current_price']) * 100

                if price_diff_percent > 5:
                    if phone1_min_price['current_price'] < phone2_min_price['current_price']:
                        st.info(
                            f"💡 **Value Analysis:** {phone1_details['Brand']} {phone1_details['Model']} is ₹{price_diff:,.2f} ({price_diff_percent:.1f}%) cheaper than {phone2_details['Brand']} {phone2_details['Model']}.")
                    else:
                        st.info(
                            f"💡 **Value Analysis:** {phone2_details['Brand']} {phone2_details['Model']} is ₹{price_diff:,.2f} ({price_diff_percent:.1f}%) cheaper than {phone1_details['Brand']} {phone1_details['Model']}.")
                else:
                    st.info("💡 **Value Analysis:** Both phones are priced similarly (within 5% difference).")

                # Get sentiment analysis for both phones if available
                phone1_sentiment = sentiment_analyzer.get_phone_sentiment_summary(phone1_id, analyzed_df)
                phone2_sentiment = sentiment_analyzer.get_phone_sentiment_summary(phone2_id, analyzed_df)

                if phone1_sentiment and phone2_sentiment:
                    st.subheader("Review Sentiment Comparison")

                    # Import pandas and plotly express if needed within this scope
                    import pandas as pd
                    import plotly.express as px

                    # Create feature comparison dataframe
                    feature_comparison = pd.DataFrame({
                        'Feature': list(phone1_sentiment['features'].keys()),
                        f"{phone1_details['Brand']} {phone1_details['Model']}": list(
                            phone1_sentiment['features'].values()),
                        f"{phone2_details['Brand']} {phone2_details['Model']}": list(
                            phone2_sentiment['features'].values())
                    })

                    # Convert sentiment scores to 0-10 scale for better visualization
                    feature_comparison[f"{phone1_details['Brand']} {phone1_details['Model']}"] = ((feature_comparison[
                                                                                                       f"{phone1_details['Brand']} {phone1_details['Model']}"] + 1) / 2) * 10
                    feature_comparison[f"{phone2_details['Brand']} {phone2_details['Model']}"] = ((feature_comparison[
                                                                                                       f"{phone2_details['Brand']} {phone2_details['Model']}"] + 1) / 2) * 10

                    # Create a bar chart comparing features
                    feature_chart = px.bar(
                        feature_comparison,
                        x='Feature',
                        y=[f"{phone1_details['Brand']} {phone1_details['Model']}",
                           f"{phone2_details['Brand']} {phone2_details['Model']}"],
                        barmode='group',
                        title="Feature Sentiment Comparison (Higher is Better)",
                        labels={'value': 'Sentiment Score (0-10)', 'variable': 'Phone'}
                    )

                    st.plotly_chart(feature_chart, use_container_width=True)

                    # Overall comparison text
                    phone1_positive = phone1_sentiment['overall']['positive']
                    phone2_positive = phone2_sentiment['overall']['positive']

                    if abs(phone1_positive - phone2_positive) > 10:
                        if phone1_positive > phone2_positive:
                            st.success(
                                f"🏆 **User Satisfaction:** {phone1_details['Brand']} {phone1_details['Model']} has {phone1_positive:.1f}% positive reviews compared to {phone2_positive:.1f}% for {phone2_details['Brand']} {phone2_details['Model']}.")
                        else:
                            st.success(
                                f"🏆 **User Satisfaction:** {phone2_details['Brand']} {phone2_details['Model']} has {phone2_positive:.1f}% positive reviews compared to {phone1_positive:.1f}% for {phone1_details['Brand']} {phone1_details['Model']}.")
                    else:
                        st.success("🏆 **User Satisfaction:** Both phones have similar user satisfaction ratings.")

                # Final recommendation
                st.subheader("Recommendation")
                if phone1_sentiment and phone2_sentiment:
                    # Calculate a simple score based on price and sentiment
                    phone1_score = (phone1_sentiment['overall']['positive'] / 100) / (
                                phone1_min_price['current_price'] / 10000)
                    phone2_score = (phone2_sentiment['overall']['positive'] / 100) / (
                                phone2_min_price['current_price'] / 10000)

                    if phone1_score > phone2_score * 1.2:
                        st.info(
                            f"Based on price and user reviews, **{phone1_details['Brand']} {phone1_details['Model']}** offers better overall value.")
                    elif phone2_score > phone1_score * 1.2:
                        st.info(
                            f"Based on price and user reviews, **{phone2_details['Brand']} {phone2_details['Model']}** offers better overall value.")
                    else:
                        st.info(
                            "Both phones offer similar value. Choose based on your preference for specific features.")
                else:
                    if phone1_min_price['current_price'] < phone2_min_price['current_price'] * 0.8:
                        st.info(
                            f"Based on price alone, **{phone1_details['Brand']} {phone1_details['Model']}** appears to offer better value.")
                    elif phone2_min_price['current_price'] < phone1_min_price['current_price'] * 0.8:
                        st.info(
                            f"Based on price alone, **{phone2_details['Brand']} {phone2_details['Model']}** appears to offer better value.")
                    else:
                        st.info(
                            "Both phones are priced similarly. Choose based on your preference for specific features.")

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
                unique_phones['Brand'] + ' ' + unique_phones['Model'] + ' (' + unique_phones['RAM'] + ', ' +
                unique_phones['Storage'] + ')'
            )

            # Get the product ID from the selected option
            selected_index = unique_phones.index[
                (unique_phones['Brand'] + ' ' + unique_phones['Model'] + ' (' + unique_phones['RAM'] + ', ' +
                 unique_phones['Storage'] + ')') == selected_phone_option
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
                        sentiment_category = "Positive" if score > 0.05 else (
                            "Negative" if score < -0.05 else "Neutral")
                        sentiment_color = "#3CB371" if score > 0.05 else ("#FF6347" if score < -0.05 else "#FFD700")

                        st.markdown(f"**{feature_names.get(feature, feature)}**")
                        st.progress((score + 1) / 2)  # Convert -1 to 1 scale to 0 to 1 for progress bar
                        st.markdown(
                            f"<span style='color:{sentiment_color}'>{sentiment_category}</span> (Score: {score:.2f})",
                            unsafe_allow_html=True)

                # Dynamic aspects discovered by KeyBERT
                dynamic_aspects = sentiment_data.get('dynamic_aspects', [])
                if dynamic_aspects:
                    import pandas as pd
                    st.subheader("Discovered Aspects (KeyBERT)")
                    st.write("Key phrases automatically extracted from reviews using BERT embeddings:")
                    da_df = pd.DataFrame(dynamic_aspects, columns=["Aspect Phrase", "Relevance Score"])
                    da_df["Relevance Score"] = da_df["Relevance Score"].round(3)
                    st.dataframe(da_df, use_container_width=True)

                # Display sample reviews with metadata
                st.subheader("Review Analysis")

                # Create three tabs for positive, negative, and sentiment breakdown
                review_tabs = st.tabs(["Positive Reviews", "Negative Reviews", "Sentiment Detail"])

                # Import sentiment_analyzer for the analyze_sentiment function if needed
                from sentiment_analyzer import analyze_sentiment

                # Tab 1: Positive Reviews
                with review_tabs[0]:
                    if sentiment_data['positive_reviews']:
                        st.markdown(
                            f"**Top Positive Comments** ({len(sentiment_data['positive_reviews'])} sentences found)")
                        for i, review in enumerate(sentiment_data['positive_reviews'][:5], 1):
                            # Get sentiment score
                            sentiment_result = analyze_sentiment(review)
                            score = sentiment_result['compound']
                            score_display = round(score * 100)

                            # Define feature keywords for tagging
                            feature_keywords = {
                                '📷 Camera': ['camera', 'photo', 'picture', 'image', 'lens', 'photography'],
                                '🔋 Battery': ['battery', 'charge', 'power', 'charging', 'life', 'runtime'],
                                '⚡ Performance': ['fast', 'speed', 'performance', 'processor', 'smooth', 'responsive'],
                                '📱 Display': ['display', 'screen', 'resolution', 'brightness', 'amoled', 'lcd',
                                              'color'],
                                '🛠️ Build': ['quality', 'build', 'design', 'durability', 'sturdy', 'premium', 'feel'],
                                '💰 Value': ['price', 'value', 'worth', 'cost', 'affordable']
                            }

                            # Find relevant feature tags
                            feature_tags = []
                            lower_review = review.lower()

                            for tag, keywords in feature_keywords.items():
                                if any(word in lower_review for word in keywords):
                                    feature_tags.append(tag)

                            feature_string = ", ".join(feature_tags) if feature_tags else "General"

                            # Display review with sentiment score and tags
                            st.info(f"{review}")
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                st.markdown(f"**Sentiment: +{score_display}%**")
                            with col2:
                                st.markdown(f"**Tags: {feature_string}**")
                            st.markdown("---")
                    else:
                        st.info("No positive reviews found.")

                # Tab 2: Negative Reviews
                with review_tabs[1]:
                    if sentiment_data['negative_reviews']:
                        st.markdown(f"**Top Concerns** ({len(sentiment_data['negative_reviews'])} sentences found)")
                        for i, review in enumerate(sentiment_data['negative_reviews'][:5], 1):
                            # Get sentiment score
                            sentiment_result = analyze_sentiment(review)
                            score = sentiment_result['compound']
                            score_display = round(abs(score) * 100)

                            # Define feature keywords for tagging
                            feature_keywords = {
                                '📷 Camera': ['camera', 'photo', 'picture', 'image', 'lens', 'photography'],
                                '🔋 Battery': ['battery', 'charge', 'power', 'charging', 'life', 'runtime'],
                                '⚡ Performance': ['slow', 'lag', 'speed', 'performance', 'processor', 'sluggish',
                                                  'freeze'],
                                '📱 Display': ['display', 'screen', 'resolution', 'brightness', 'amoled', 'lcd',
                                              'color'],
                                '🛠️ Build': ['quality', 'build', 'design', 'durability', 'flimsy', 'break', 'cheap'],
                                '💰 Value': ['price', 'value', 'worth', 'cost', 'expensive', 'overpriced']
                            }

                            # Find relevant feature tags
                            feature_tags = []
                            lower_review = review.lower()

                            for tag, keywords in feature_keywords.items():
                                if any(word in lower_review for word in keywords):
                                    feature_tags.append(tag)

                            feature_string = ", ".join(feature_tags) if feature_tags else "General"

                            # Display review with sentiment score and tags
                            st.error(f"{review}")
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                st.markdown(f"**Sentiment: -{score_display}%**")
                            with col2:
                                st.markdown(f"**Tags: {feature_string}**")
                            st.markdown("---")
                    else:
                        st.info("No negative reviews found.")

                # Tab 3: Sentiment Breakdown
                with review_tabs[2]:
                    # Show a bar chart of review count by feature
                    st.subheader("Review Distribution by Feature")

                    # Count how many reviews mention each feature
                    feature_counts = {
                        "Camera": 0,
                        "Battery": 0,
                        "Performance": 0,
                        "Display": 0,
                        "Build Quality": 0,
                        "Value": 0,
                        "General": 0
                    }

                    # Count feature mentions in positive reviews
                    for review in sentiment_data['positive_reviews'] + sentiment_data['negative_reviews']:
                        lower_review = review.lower()
                        feature_found = False
                        if any(word in lower_review for word in ['camera', 'photo', 'picture', 'image']):
                            feature_counts["Camera"] += 1
                            feature_found = True
                        if any(word in lower_review for word in ['battery', 'charge', 'power']):
                            feature_counts["Battery"] += 1
                            feature_found = True
                        if any(word in lower_review for word in ['fast', 'slow', 'speed', 'performance', 'processor']):
                            feature_counts["Performance"] += 1
                            feature_found = True
                        if any(word in lower_review for word in ['display', 'screen', 'resolution']):
                            feature_counts["Display"] += 1
                            feature_found = True
                        if any(word in lower_review for word in ['quality', 'build', 'design', 'durability']):
                            feature_counts["Build Quality"] += 1
                            feature_found = True
                        if any(word in lower_review for word in ['price', 'value', 'worth', 'cost', 'expensive']):
                            feature_counts["Value"] += 1
                            feature_found = True
                        if not feature_found:
                            feature_counts["General"] += 1

                    # Create a DataFrame for the chart
                    import pandas as pd
                    import plotly.express as px

                    feature_df = pd.DataFrame({
                        'Feature': list(feature_counts.keys()),
                        'Review Count': list(feature_counts.values())
                    })

                    # Filter out empty features
                    feature_df = feature_df[feature_df['Review Count'] > 0]

                    # Sort by count
                    feature_df = feature_df.sort_values('Review Count', ascending=False)

                    # Create a bar chart
                    if not feature_df.empty:
                        fig = px.bar(
                            feature_df,
                            x='Feature',
                            y='Review Count',
                            color='Feature',
                            title="Number of Review Sentences by Feature",
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No feature-specific review data available.")
            else:
                st.warning("Sentiment data not available for this phone.")

    # PRICE ANALYSIS PAGE
    elif app_mode == "Price Analysis":
        st.markdown("<div class='subheader-text'>💰 Price Analysis</div>", unsafe_allow_html=True)
        st.write("Analyze smartphone price trends and forecasts")

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
                "Select a phone for price analysis:",
                unique_phones['Brand'] + ' ' + unique_phones['Model'] + ' (' + unique_phones['RAM'] + ', ' +
                unique_phones['Storage'] + ')'
            )

            # Get the product ID from the selected option
            selected_index = unique_phones.index[
                (unique_phones['Brand'] + ' ' + unique_phones['Model'] + ' (' + unique_phones['RAM'] + ', ' +
                 unique_phones['Storage'] + ')') == selected_phone_option
                ].tolist()[0]
            selected_phone_id = unique_phones.loc[selected_index, 'Product_ID']

            # First show current price comparison
            st.subheader("Current Prices")
            price_data = data_processor.get_price_comparison(
                selected_phone_id, flipkart_df, amazon_df, cashify_df
            )

            if price_data:
                # Create and display price comparison chart
                price_chart = visualization.create_price_comparison_chart(price_data)
                if price_chart:
                    st.plotly_chart(price_chart, use_container_width=True)

                # Show price trend forecast
                st.subheader("Price Trend Analysis")
                price_trend_data = data_processor.generate_price_trends(selected_phone_id, combined_df)

                if price_trend_data:
                    trend_chart = visualization.create_price_trend_chart(price_trend_data)
                    if trend_chart:
                        st.plotly_chart(trend_chart, use_container_width=True)

                    # Price prediction insights
                    st.subheader("Price Prediction Insights")
                    st.info(f"**Best time to buy:** {price_trend_data['best_time']}")
                    st.info(f"**Expected price:** ₹{price_trend_data['expected_price']:,.2f}")

                    # Calculate potential savings
                    min_current_price = min(item['current_price'] for item in price_data)
                    potential_savings = min_current_price - price_trend_data['expected_price']

                    if potential_savings > 0:
                        st.success(
                            f"**Potential savings:** ₹{potential_savings:,.2f} ({(potential_savings / min_current_price) * 100:.1f}%)")
                    else:
                        st.warning("Prices are expected to rise. Consider buying now.")
                else:
                    st.warning("Price trend data not available for this phone.")
            else:
                st.warning("Price data not available for this phone.")

            # Show sentiment-based feature analysis for the selected phone
            st.subheader("Feature Analysis")

            # Get sentiment data for this phone
            sentiment_data = sentiment_analyzer.get_phone_sentiment_summary(
                selected_phone_id, analyzed_df
            )

            if sentiment_data and 'features' in sentiment_data:
                # Create feature analysis display
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
                        sentiment_category = "Positive" if score > 0.05 else (
                            "Negative" if score < -0.05 else "Neutral")
                        sentiment_color = "#3CB371" if score > 0.05 else ("#FF6347" if score < -0.05 else "#FFD700")

                        st.markdown(f"**{feature_names.get(feature, feature)}**")
                        st.progress((score + 1) / 2)  # Convert -1 to 1 scale to 0 to 1 for progress bar
                        st.markdown(
                            f"<span style='color:{sentiment_color}'>{sentiment_category}</span> (Score: {score:.2f})",
                            unsafe_allow_html=True)
            else:
                st.info("Feature sentiment analysis not available for this phone.")

            # Platform discount comparison
            st.subheader("Platform Discount Comparison")
            platform_discount_chart = visualization.create_platform_discount_comparison(combined_df)
            if platform_discount_chart:
                st.plotly_chart(platform_discount_chart, use_container_width=True)

    # RECOMMENDATIONS PAGE
    elif app_mode == "Recommendations":
        st.markdown("<div class='subheader-text'>🎯 Personalized Recommendations</div>", unsafe_allow_html=True)
        st.write("Get smartphone recommendations based on your preferences")

        # Sidebar filters
        st.sidebar.title("Preferences")

        # Brand preferences
        all_brands = sorted(combined_df['Brand'].unique())
        selected_brands = st.sidebar.multiselect("Preferred Brands", all_brands, default=all_brands[:5])

        # Price range slider
        min_price = int(combined_df['Price'].min())
        max_price = int(combined_df['Price'].max())
        price_range = st.sidebar.slider("Budget Range (₹)", min_price, max_price, (min_price, max_price))

        # Feature importance
        st.sidebar.subheader("Feature Importance")
        st.sidebar.write("Rate the importance of each feature (1-10):")

        camera_imp = st.sidebar.slider("📷 Camera", 1, 10, 5)
        battery_imp = st.sidebar.slider("🔋 Battery", 1, 10, 5)
        performance_imp = st.sidebar.slider("⚡ Performance", 1, 10, 5)
        display_imp = st.sidebar.slider("📱 Display", 1, 10, 5)
        build_quality_imp = st.sidebar.slider("🛠️ Build Quality", 1, 10, 5)
        value_imp = st.sidebar.slider("💰 Value for Money", 1, 10, 5)

        # Normalize weights to sum to 1
        total_weight = camera_imp + battery_imp + performance_imp + display_imp + build_quality_imp + value_imp

        if total_weight > 0:  # Avoid division by zero
            feature_preferences = {
                'camera_sentiment': (camera_imp / total_weight) * 0.9,
                'battery_sentiment': (battery_imp / total_weight) * 0.9,
                'performance_sentiment': (performance_imp / total_weight) * 0.9,
                'display_sentiment': (display_imp / total_weight) * 0.9,
                'build_quality_sentiment': (build_quality_imp / total_weight) * 0.9,
                'value_sentiment': (value_imp / total_weight) * 0.9,
                'price': 0.1  # Always keep price as a factor
            }
        else:
            # Default weights if all sliders are set to minimum
            feature_preferences = {
                'camera_sentiment': 0.15,
                'battery_sentiment': 0.15,
                'performance_sentiment': 0.15,
                'display_sentiment': 0.15,
                'build_quality_sentiment': 0.15,
                'value_sentiment': 0.15,
                'price': 0.1
            }

        # Other specifications
        st.sidebar.subheader("Minimum Specifications")
        min_ram_options = ["Any", "4GB", "6GB", "8GB", "12GB"]
        min_ram = st.sidebar.selectbox("Minimum RAM", min_ram_options)

        min_storage_options = ["Any", "64GB", "128GB", "256GB", "512GB", "1TB"]
        min_storage = st.sidebar.selectbox("Minimum Storage", min_storage_options)

        # Convert "Any" to None for filtering
        min_ram = None if min_ram == "Any" else min_ram
        min_storage = None if min_storage == "Any" else min_storage

        # Apply filters
        filters = {
            'brands': selected_brands,
            'min_price': price_range[0],
            'max_price': price_range[1],
            'min_ram': min_ram,
            'min_storage': min_storage
        }

        # Get recommendations
        with st.spinner("Generating recommendations..."):
            recommendations = recommendation_engine.get_recommendations(
                analyzed_df,
                feature_preferences=feature_preferences,
                num_recommendations=10,
                filters=filters
            )

        # Display feature importance visualization
        col1, col2 = st.columns([1, 2])

        with col1:
            # Display feature importance chart
            feature_importance_chart = visualization.create_feature_importance_chart(feature_preferences)
            if feature_importance_chart:
                st.plotly_chart(feature_importance_chart)

        with col2:
            # Display recommendations summary
            if recommendations.empty:
                st.warning("No phones match your preferences. Try adjusting your filters.")
            else:
                st.subheader(f"Top {len(recommendations)} Recommendations")
                st.write("Based on your preferences, here are your personalized smartphone recommendations:")

        # Display recommendations in a nice grid
        if not recommendations.empty:
            # Display each recommendation in a card
            for i in range(0, len(recommendations), 2):
                cols = st.columns(2)

                for j in range(2):
                    idx = i + j
                    if idx < len(recommendations):
                        phone = recommendations.iloc[idx]
                        with cols[j]:
                            st.markdown(f"### {idx + 1}. {phone['Brand']} {phone['Model']}")

                            # Specifications
                            specs_cols = st.columns(2)
                            with specs_cols[0]:
                                st.write(f"**RAM:** {phone['RAM']}")
                                st.write(f"**Storage:** {phone['Storage']}")
                                st.write(f"**Price:** ₹{phone['Price']:,.2f}")

                            with specs_cols[1]:
                                st.write(f"**Screen Size:** {phone['Screen_Size']}")
                                st.write(f"**Battery:** {phone['Battery']}")
                                st.write(f"**Camera:** {phone['Main_Camera']}")

                            # Sentiment score
                            sentiment_score = phone['Sentiment_Score']
                            sentiment_category = "Positive" if sentiment_score > 0.05 else (
                                "Negative" if sentiment_score < -0.05 else "Neutral")
                            sentiment_color = "#3CB371" if sentiment_score > 0.05 else (
                                "#FF6347" if sentiment_score < -0.05 else "#FFD700")

                            st.write("**User Sentiment:**")
                            st.progress((sentiment_score + 1) / 2)  # Convert -1 to 1 scale to 0 to 1 for progress bar
                            st.markdown(
                                f"<span style='color:{sentiment_color}'>{sentiment_category}</span> (Score: {sentiment_score:.2f})",
                                unsafe_allow_html=True)

                            st.markdown("---")


# Run the app
if __name__ == "__main__":
    main()
