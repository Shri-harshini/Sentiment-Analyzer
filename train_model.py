"""
Simple Sentiment Analysis Model Training Script
Uses sklearn and NLTK for text classification
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data (only needed once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')

def create_sample_dataset():
    """Create a simple sample dataset for sentiment analysis"""
    data = {
        'text': [
            # Positive samples
            "I love this product! It's amazing and works perfectly.",
            "This is the best thing I've ever bought. Highly recommended!",
            "Excellent quality and great customer service. Very satisfied!",
            "Absolutely fantastic! Exceeded all my expectations.",
            "I'm so happy with this purchase. Worth every penny!",
            "Great experience overall. Would definitely buy again.",
            "Outstanding performance and reliability. Love it!",
            "Perfect! Exactly what I was looking for.",
            "Amazing quality and fast shipping. Very impressed!",
            "This product changed my life. Thank you so much!",
            
            # Negative samples
            "This is terrible. I'm very disappointed with the service.",
            "Worst purchase ever. Complete waste of money.",
            "Horrible experience. Would not recommend to anyone.",
            "Product broke after one use. Very poor quality.",
            "Customer service was rude and unhelpful. Frustrating!",
            "I regret buying this. Total disappointment.",
            "Awful quality and overpriced. Stay away!",
            "Terrible experience from start to finish.",
            "Doesn't work at all. Completely useless.",
            "Very poor quality. Not worth the money.",
            
            # Neutral samples
            "The weather is okay today, nothing special.",
            "Just another regular day at the office.",
            "The product works as expected, nothing more.",
            "It's fine, I guess. Does the job.",
            "Average quality for the price. Nothing extraordinary.",
            "The service was acceptable. No complaints.",
            "It's neither good nor bad. Just okay.",
            "Standard performance. Meets basic requirements.",
            "Nothing impressive, but not terrible either.",
            "It works. That's about all I can say."
        ],
        'sentiment': (
            ['positive'] * 10 +  # 10 positive samples
            ['negative'] * 10 +  # 10 negative samples
            ['neutral'] * 10     # 10 neutral samples
        )
    }
    
    return pd.DataFrame(data)

def preprocess_text(text):
    """
    Preprocess text data:
    1. Convert to lowercase
    2. Remove punctuation and special characters
    3. Remove stopwords
    4. Tokenize
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # 3. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 4. Tokenize and remove stopwords
    try:
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        return ' '.join(tokens)
    except:
        return text

def main():
    """Main training function"""
    print("=== Sentiment Analysis Model Training ===\n")
    
    # 1. Create and save sample dataset
    print("1. Creating sample dataset...")
    df = create_sample_dataset()
    df.to_csv('sentiment_dataset.csv', index=False)
    print(f"   Dataset created with {len(df)} samples")
    print(f"   Distribution: {df['sentiment'].value_counts().to_dict()}\n")
    
    # 2. Preprocess the text data
    print("2. Preprocessing text data...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    print("   Text preprocessing completed\n")
    
    # 3. Split the data
    print("3. Splitting data into train and test sets...")
    X = df['processed_text']
    y = df['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}\n")
    
    # 4. Create and fit TF-IDF Vectorizer
    print("4. Creating TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=1000,  # Limit vocabulary size
        ngram_range=(1, 2), # Use unigrams and bigrams
        min_df=1,           # Minimum document frequency
        max_df=0.8          # Maximum document frequency
    )
    
    # Fit and transform training data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"   Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"   Feature matrix shape: {X_train_tfidf.shape}\n")
    
    # 5. Train the model (using Naive Bayes)
    print("5. Training Naive Bayes classifier...")
    model = MultinomialNB(alpha=1.0)  # Laplace smoothing
    model.fit(X_train_tfidf, y_train)
    print("   Model training completed\n")
    
    # 6. Evaluate the model
    print("6. Evaluating model performance...")
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # 7. Save the model and vectorizer
    print("\n7. Saving model and vectorizer...")
    
    # Save the trained model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("   Model saved as 'model.pkl'")
    
    # Save the vectorizer
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("   Vectorizer saved as 'vectorizer.pkl'")
    
    # 8. Test with sample predictions
    print("\n8. Testing with sample predictions...")
    test_texts = [
        "This product is absolutely wonderful!",
        "I hate this terrible service.",
        "The package arrived on time."
    ]
    
    for text in test_texts:
        # Preprocess
        processed = preprocess_text(text)
        # Vectorize
        text_tfidf = vectorizer.transform([processed])
        # Predict
        prediction = model.predict(text_tfidf)[0]
        # Get probability
        probabilities = model.predict_proba(text_tfidf)[0]
        confidence = max(probabilities)
        
        print(f"   Text: '{text}'")
        print(f"   Prediction: {prediction} (confidence: {confidence:.2f})")
        print()
    
    print("=== Training Complete! ===")
    print("\nFiles created:")
    print("- sentiment_dataset.csv (training data)")
    print("- model.pkl (trained model)")
    print("- vectorizer.pkl (TF-IDF vectorizer)")

def predict_sentiment(text, model_path='model.pkl', vectorizer_path='vectorizer.pkl'):
    """
    Function to predict sentiment for new text using saved model
    """
    try:
        # Load the saved model and vectorizer
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Preprocess the text
        processed_text = preprocess_text(text)
        
        # Vectorize the text
        text_tfidf = vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = model.predict(text_tfidf)[0]
        probabilities = model.predict_proba(text_tfidf)[0]
        confidence = max(probabilities)
        
        return {
            'sentiment': prediction,
            'confidence': confidence,
            'probabilities': {
                'positive': float(probabilities[0]) if len(probabilities) == 3 else 0.0,
                'negative': float(probabilities[1]) if len(probabilities) == 3 else 0.0,
                'neutral': float(probabilities[2]) if len(probabilities) == 3 else 0.0
            }
        }
        
    except FileNotFoundError:
        return {'error': 'Model or vectorizer file not found. Please run training first.'}
    except Exception as e:
        return {'error': f'Prediction error: {str(e)}'}

if __name__ == "__main__":
    # Run the training
    main()
    
    # Example of how to use the prediction function
    print("\n" + "="*50)
    print("Example Usage of Prediction Function:")
    print("="*50)
    
    result = predict_sentiment("I really love this amazing product!")
    if 'error' not in result:
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.2f}")
    else:
        print(result['error'])
