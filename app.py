from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from model import SentimentAnalyzer
import os

app = Flask(__name__)

# Initialize the sentiment analyzer
sentiment_analyzer = SentimentAnalyzer()

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    """Analyze sentiment of the provided text"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Please provide some text to analyze'}), 400
        
        # Get sentiment prediction
        result = sentiment_analyzer.predict(text)
        
        return jsonify({
            'success': True,
            'sentiment': result['sentiment'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train_model():
    """Train the sentiment analysis model"""
    try:
        # Check if dataset exists
        if not os.path.exists('dataset.csv'):
            return jsonify({'error': 'Dataset not found'}), 404
        
        # Train the model
        metrics = sentiment_analyzer.train('dataset.csv')
        
        return jsonify({
            'success': True,
            'message': 'Model trained successfully',
            'metrics': metrics
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the current model"""
    try:
        info = sentiment_analyzer.get_model_info()
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
