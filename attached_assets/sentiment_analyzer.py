import pandas as pd
import re
import string
from collections import defaultdict

# Initialize sentiment analyzer with default values first
sia = None
stopwords = set()

try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    
    # Download necessary NLTK data
    print("Downloading NLTK resources...")
    nltk.download('punkt', quiet=False)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
    
    # Initialize proper sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    stopwords = set(nltk.corpus.stopwords.words('english'))
    print("NLTK resources loaded successfully")
except Exception as e:
    print(f"Warning: NLTK initialization error: {e}")
    # If NLTK fails, we'll use a simpler approach
    # We'll define a minimal sentiment analyzer as fallback
    
    class SimpleSentimentAnalyzer:
        def polarity_scores(self, text):
            # Simple polarity based on positive/negative word count
            positive_words = ['good', 'great', 'excellent', 'amazing', 'perfect', 'love', 'best']
            negative_words = ['bad', 'poor', 'terrible', 'awful', 'worst', 'hate', 'disappointing']
            
            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            total = pos_count + neg_count
            if total == 0:
                return {'compound': 0, 'pos': 0.5, 'neg': 0.5, 'neu': 0}
                
            pos_score = pos_count / (pos_count + neg_count) if (pos_count + neg_count) > 0 else 0.5
            neg_score = neg_count / (pos_count + neg_count) if (pos_count + neg_count) > 0 else 0.5
            compound = (pos_score - neg_score)  # Between -1 and 1
            
            return {'compound': compound, 'pos': pos_score, 'neg': neg_score, 'neu': 0}
    
    # Use our simple sentiment analyzer as fallback
    sia = SimpleSentimentAnalyzer()


def clean_text(text):
    """
    Clean the review text by removing punctuation, numbers, and stopwords
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def analyze_sentiment(review):
    """
    Analyze sentiment of a review using VADER
    """
    if not isinstance(review, str) or review == "":
        return {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0}
    
    sentiment = sia.polarity_scores(review)
    return sentiment


def get_feature_sentiments(review):
    """
    Extract feature-based sentiments from a review
    """
    if not isinstance(review, str) or review == "":
        return {}
    
    features = {
        'camera': r'\b(camera|photo|picture|image|selfie|video)\b',
        'battery': r'\b(battery|charge|charging|power|life)\b',
        'performance': r'\b(performance|speed|fast|slow|lag|processor|smooth)\b',
        'display': r'\b(display|screen|resolution|bright|color|touch)\b',
        'build_quality': r'\b(build|quality|material|design|look|feel|durability)\b',
        'value': r'\b(price|value|worth|money|expensive|cheap|cost)\b'
    }
    
    results = {}
    
    # Use a simple sentence tokenizer as a fallback for NLTK's tokenizer
    try:
        # Attempt to use NLTK's sentence tokenizer
        sentences = nltk.sent_tokenize(review)
        print("NLTK tokenization successful")
    except Exception as e:
        # Log the specific error for troubleshooting
        print(f"NLTK tokenization failed: {e}. Using fallback tokenizer.")
        # Fallback to simple period-based splitting if NLTK tokenizer fails
        sentences = [s.strip() for s in re.split(r'[.!?]+', review) if s.strip()]
        if not sentences:  # If splitting produced no sentences, treat the whole review as one sentence
            sentences = [review] if review.strip() else []
    
    for feature, pattern in features.items():
        feature_sentences = []
        for sentence in sentences:
            if re.search(pattern, sentence.lower()):
                feature_sentences.append(sentence)
        
        if feature_sentences:
            combined_text = ' '.join(feature_sentences)
            sentiment = analyze_sentiment(combined_text)
            results[feature] = sentiment['compound']
        else:
            results[feature] = None
    
    return results


def analyze_reviews(combined_df):
    """
    Analyze all reviews and generate sentiment data
    """
    # Create a copy to avoid modifying the original dataframe
    df = combined_df.copy()
    
    # Clean reviews and analyze overall sentiment
    df['Clean_Review'] = df['Reviews'].apply(clean_text)
    df['Sentiment'] = df['Clean_Review'].apply(analyze_sentiment)
    df['Sentiment_Score'] = df['Sentiment'].apply(lambda x: x['compound'])
    df['Sentiment_Category'] = df['Sentiment_Score'].apply(
        lambda x: 'Positive' if x >= 0.05 else ('Negative' if x <= -0.05 else 'Neutral')
    )
    
    # Extract feature-specific sentiments
    feature_sentiments = df['Clean_Review'].apply(get_feature_sentiments)
    
    # Add feature sentiments as separate columns
    for feature in ['camera', 'battery', 'performance', 'display', 'build_quality', 'value']:
        df[f'{feature}_sentiment'] = feature_sentiments.apply(lambda x: x.get(feature, None))
    
    return df


def get_phone_sentiment_summary(phone_id, analyzed_df):
    """
    Generate a sentiment summary for a specific phone
    """
    phone_data = analyzed_df[analyzed_df['Product_ID'] == phone_id]
    
    if phone_data.empty:
        return None
    
    # Overall sentiment statistics
    sentiment_counts = phone_data['Sentiment_Category'].value_counts().to_dict()
    total_reviews = phone_data.shape[0]
    positive_percentage = sentiment_counts.get('Positive', 0) / total_reviews * 100 if total_reviews > 0 else 0
    negative_percentage = sentiment_counts.get('Negative', 0) / total_reviews * 100 if total_reviews > 0 else 0
    neutral_percentage = sentiment_counts.get('Neutral', 0) / total_reviews * 100 if total_reviews > 0 else 0
    
    # Feature-specific sentiment
    features = ['camera', 'battery', 'performance', 'display', 'build_quality', 'value']
    feature_scores = {}
    
    for feature in features:
        feature_values = phone_data[f'{feature}_sentiment'].dropna()
        if len(feature_values) > 0:
            avg_score = feature_values.mean()
            feature_scores[feature] = avg_score
        else:
            feature_scores[feature] = 0
    
    # Extract positive and negative reviews for display
    positive_reviews = phone_data[phone_data['Sentiment_Score'] > 0.5]['Reviews'].tolist()
    negative_reviews = phone_data[phone_data['Sentiment_Score'] < -0.5]['Reviews'].tolist()
    
    # Limit to top 5 most extreme reviews
    if positive_reviews:
        positive_indices = phone_data[phone_data['Sentiment_Score'] > 0.5].sort_values('Sentiment_Score', ascending=False).index[:5]
        positive_reviews = phone_data.loc[positive_indices, 'Reviews'].tolist()
    
    if negative_reviews:
        negative_indices = phone_data[phone_data['Sentiment_Score'] < -0.5].sort_values('Sentiment_Score', ascending=True).index[:5]
        negative_reviews = phone_data.loc[negative_indices, 'Reviews'].tolist()
    
    return {
        'overall': {
            'positive': positive_percentage,
            'negative': negative_percentage,
            'neutral': neutral_percentage,
            'total_reviews': total_reviews
        },
        'features': feature_scores,
        'positive_reviews': positive_reviews,
        'negative_reviews': negative_reviews
    }
