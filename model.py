import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data (only once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.model_type = 'naive_bayes'
        self.is_trained = False
        self.label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.reverse_mapping = {v: k for k, v in self.label_mapping.items()}
        
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove punctuation and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and remove stopwords
        try:
            tokens = word_tokenize(text)
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words]
            return ' '.join(tokens)
        except:
            return text
    
    def load_data(self, file_path):
        """Load and preprocess dataset"""
        try:
            df = pd.read_csv(file_path)
            
            # Check required columns
            if 'text' not in df.columns or 'sentiment' not in df.columns:
                raise ValueError("Dataset must contain 'text' and 'sentiment' columns")
            
            # Remove missing values
            df = df.dropna(subset=['text', 'sentiment'])
            
            # Preprocess text
            df['processed_text'] = df['text'].apply(self.preprocess_text)
            
            # Remove empty texts after preprocessing
            df = df[df['processed_text'].str.len() > 0]
            
            # Map sentiment labels to numbers
            df['sentiment_label'] = df['sentiment'].map(self.label_mapping)
            
            # Remove rows with invalid sentiment labels
            df = df.dropna(subset=['sentiment_label'])
            df['sentiment_label'] = df['sentiment_label'].astype(int)
            
            return df
            
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def train(self, file_path, model_type='naive_bayes', test_size=0.2):
        """Train the sentiment analysis model"""
        try:
            # Load and preprocess data
            df = self.load_data(file_path)
            
            if len(df) < 10:
                raise ValueError("Dataset too small for training")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                df['processed_text'], 
                df['sentiment_label'], 
                test_size=test_size, 
                random_state=42,
                stratify=df['sentiment_label']
            )
            
            # Create and fit vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)
            
            # Select and train model
            self.model_type = model_type
            
            if model_type == 'naive_bayes':
                self.model = MultinomialNB(alpha=1.0)
            elif model_type == 'logistic_regression':
                self.model = LogisticRegression(random_state=42, max_iter=1000)
            elif model_type == 'svm':
                self.model = SVC(random_state=42, probability=True)
            elif model_type == 'random_forest':
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Train model
            self.model.fit(X_train_vec, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Generate classification report
            report = classification_report(y_test, y_pred, target_names=list(self.label_mapping.keys()))
            
            self.is_trained = True
            
            # Save model
            self.save_model()
            
            return {
                'accuracy': accuracy,
                'classification_report': report,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'model_type': model_type
            }
            
        except Exception as e:
            raise Exception(f"Error training model: {str(e)}")
    
    def predict(self, text):
        """Predict sentiment for a single text"""
        if not self.is_trained:
            # Try to load existing model
            if not self.load_model():
                raise Exception("Model not trained. Please train the model first.")
        
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            if not processed_text:
                return {
                    'sentiment': 'neutral',
                    'confidence': 0.5,
                    'probabilities': {'negative': 0.33, 'neutral': 0.34, 'positive': 0.33}
                }
            
            # Vectorize text
            text_vec = self.vectorizer.transform([processed_text])
            
            # Get prediction
            prediction = self.model.predict(text_vec)[0]
            probabilities = self.model.predict_proba(text_vec)[0]
            
            # Handle both numeric and string predictions
            if isinstance(prediction, str):
                # Model returned string label directly
                sentiment = prediction.lower()
            else:
                # Model returned numeric index
                prediction = int(prediction)
                sentiment = self.reverse_mapping[prediction]
            
            # Create probability dictionary
            prob_dict = {}
            for i, label in enumerate(list(self.label_mapping.keys())):
                prob_dict[label] = float(probabilities[i])
            
            return {
                'sentiment': sentiment,
                'confidence': float(max(probabilities)),
                'probabilities': prob_dict
            }
            
        except Exception as e:
            raise Exception(f"Error making prediction: {str(e)}")
    
    def save_model(self):
        """Save the trained model and vectorizer"""
        try:
            if self.model and self.vectorizer:
                with open('sentiment_model.pkl', 'wb') as f:
                    pickle.dump({
                        'model': self.model,
                        'vectorizer': self.vectorizer,
                        'model_type': self.model_type,
                        'label_mapping': self.label_mapping,
                        'reverse_mapping': self.reverse_mapping
                    }, f)
        except Exception as e:
            print(f"Warning: Could not save model: {str(e)}")
    
    def load_model(self):
        """Load a trained model"""
        try:
            # Check for both possible model filenames
            model_file = None
            vectorizer_file = None
            
            if os.path.exists('sentiment_model.pkl'):
                model_file = 'sentiment_model.pkl'
            elif os.path.exists('model.pkl'):
                model_file = 'model.pkl'
            
            if os.path.exists('sentiment_vectorizer.pkl'):
                vectorizer_file = 'sentiment_vectorizer.pkl'
            elif os.path.exists('vectorizer.pkl'):
                vectorizer_file = 'vectorizer.pkl'
            
            if model_file and vectorizer_file:
                # Load model and vectorizer separately (from train_model.py)
                with open(model_file, 'rb') as f:
                    self.model = pickle.load(f)
                
                with open(vectorizer_file, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                
                # Set default values for compatibility
                self.model_type = 'naive_bayes'
                self.label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
                self.reverse_mapping = {v: k for k, v in self.label_mapping.items()}
                self.is_trained = True
                return True
                
            elif model_file and os.path.exists(model_file):
                # Load combined model file (from Flask training)
                with open(model_file, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.vectorizer = data['vectorizer']
                    self.model_type = data['model_type']
                    self.label_mapping = data['label_mapping']
                    self.reverse_mapping = data['reverse_mapping']
                    self.is_trained = True
                    return True
                    
        except Exception as e:
            print(f"Warning: Could not load model: {str(e)}")
        
        return False
    
    def get_model_info(self):
        """Get information about the current model"""
        if not self.is_trained:
            return {
                'is_trained': False,
                'message': 'Model not trained yet'
            }
        
        return {
            'is_trained': True,
            'model_type': self.model_type,
            'supported_sentiments': list(self.label_mapping.keys()),
            'label_mapping': self.label_mapping
        }
