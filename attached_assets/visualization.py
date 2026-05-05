import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st


def create_price_comparison_chart(price_data):
    """
    Create a bar chart comparing prices across platforms
    """
    if not price_data:
        return None
    
    # Create DataFrame from price data
    df = pd.DataFrame(price_data)
    
    # Create figure
    fig = go.Figure()
    
    # Add original price bars
    fig.add_trace(go.Bar(
        x=df['platform'],
        y=df['original_price'],
        name='Original Price',
        marker_color='lightgrey',
        opacity=0.7
    ))
    
    # Add current price bars
    fig.add_trace(go.Bar(
        x=df['platform'],
        y=df['current_price'],
        name='Current Price',
        marker_color='royalblue'
    ))
    
    # Add discount annotations
    for i, row in df.iterrows():
        fig.add_annotation(
            x=row['platform'],
            y=row['original_price'],
            text=f"{row['discount']:.1f}% off",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-30
        )
    
    # Update layout
    fig.update_layout(
        title="Price Comparison Across Platforms",
        xaxis_title="Platform",
        yaxis_title="Price (₹)",
        barmode='group',
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_sentiment_radar_chart(sentiment_data):
    """
    Create a radar chart showing sentiment across features
    """
    if not sentiment_data or 'features' not in sentiment_data:
        return None
    
    features = sentiment_data['features']
    
    # Convert sentiment scores to a 0-10 scale for better visualization
    # (VADER scores range from -1 to 1)
    feature_scores = {k: ((v + 1) / 2) * 10 for k, v in features.items()}
    
    # Create categories and values for the radar chart
    categories = [k.replace('_', ' ').title() for k in feature_scores.keys()]
    values = list(feature_scores.values())
    
    # Add the first value at the end to close the loop
    categories.append(categories[0])
    values.append(values[0])
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Sentiment Score',
        line_color='royalblue'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )
        ),
        showlegend=False,
        title="Feature Sentiment Analysis",
        height=400
    )
    
    return fig


def create_sentiment_distribution_chart(sentiment_data):
    """
    Create a pie chart showing the distribution of sentiment
    """
    if not sentiment_data or 'overall' not in sentiment_data:
        return None
    
    overall = sentiment_data['overall']
    
    # Create data for pie chart
    labels = ['Positive', 'Neutral', 'Negative']
    values = [overall['positive'], overall['neutral'], overall['negative']]
    colors = ['#3CB371', '#FFD700', '#FF6347']
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.3,
        marker_colors=colors
    )])
    
    fig.update_layout(
        title=f"Review Sentiment Distribution (Total: {overall['total_reviews']} reviews)",
        height=350
    )
    
    return fig


def create_price_trend_chart(trend_data):
    """
    Create a line chart showing historical and predicted price trends
    """
    if not trend_data:
        return None
    
    # Extract historical and future trend data
    historical = trend_data['historical']
    future = trend_data['future']
    best_time = trend_data['best_time']
    expected_price = trend_data['expected_price']
    
    # Create DataFrames
    hist_df = pd.DataFrame(historical, columns=['date', 'price'])
    future_df = pd.DataFrame(future, columns=['date', 'price'])
    
    # Convert date strings to datetime
    hist_df['date'] = pd.to_datetime(hist_df['date'])
    future_df['date'] = pd.to_datetime(future_df['date'])
    
    # Create figure
    fig = go.Figure()
    
    # Add historical price line
    fig.add_trace(go.Scatter(
        x=hist_df['date'],
        y=hist_df['price'],
        mode='lines',
        name='Historical Price',
        line=dict(color='royalblue', width=2)
    ))
    
    # Add predicted price line
    fig.add_trace(go.Scatter(
        x=future_df['date'],
        y=future_df['price'],
        mode='lines+markers',
        name='Predicted Price',
        line=dict(color='orange', width=2, dash='dash')
    ))
    
    # Add best time marker
    best_time_date = pd.to_datetime(best_time)
    fig.add_trace(go.Scatter(
        x=[best_time_date],
        y=[expected_price],
        mode='markers',
        marker=dict(size=12, color='red', symbol='star'),
        name='Best Time to Buy'
    ))
    
    # Add annotation for best time
    fig.add_annotation(
        x=best_time_date,
        y=expected_price,
        text=f"Best Time: {best_time}<br>Expected Price: ₹{expected_price:.2f}",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40
    )
    
    # Update layout
    fig.update_layout(
        title="Price Trend and Forecast",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_recommendation_bar_chart(recommended_phones):
    """
    Create a horizontal bar chart for recommended phones
    """
    if recommended_phones.empty:
        return None
    
    # Prepare data
    df = recommended_phones.sort_values('recommendation_score', ascending=True).tail(10)
    
    # Create the bar chart
    fig = px.bar(
        df,
        y='Product_Name',
        x='recommendation_score',
        color='recommendation_score',
        orientation='h',
        color_continuous_scale='blues',
        title="Top Recommended Phones",
        labels={'recommendation_score': 'Recommendation Score', 'Product_Name': 'Phone'}
    )
    
    fig.update_layout(height=400)
    
    return fig


def create_feature_importance_chart(feature_importance):
    """
    Create a pie chart showing feature importance weights
    """
    # Prepare data
    labels = [k.replace('_', ' ').replace('sentiment', '').title() for k in feature_importance.keys()]
    values = list(feature_importance.values())
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.3
    )])
    
    fig.update_layout(
        title="Feature Importance Weights",
        height=350
    )
    
    return fig


def create_platform_discount_comparison(combined_df):
    """
    Create a box plot comparing discount percentages across platforms
    """
    # Calculate average discount by platform
    platform_discounts = combined_df.groupby('Source')['Discount_Percentage'].agg(['mean', 'min', 'max']).reset_index()
    
    # Create figure
    fig = go.Figure()
    
    for platform in platform_discounts['Source']:
        platform_data = combined_df[combined_df['Source'] == platform]
        
        fig.add_trace(go.Box(
            y=platform_data['Discount_Percentage'],
            name=platform,
            boxmean=True
        ))
    
    # Update layout
    fig.update_layout(
        title="Discount Comparison Across Platforms",
        yaxis_title="Discount Percentage (%)",
        height=400
    )
    
    return fig
