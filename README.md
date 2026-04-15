# AI Sentiment Analyzer

A full-stack web application for analyzing text sentiment using machine learning. Built with Flask, Scikit-learn, and modern web technologies.

## Features

- **Real-time Sentiment Analysis**: Analyze text for positive, negative, or neutral sentiment
- **Multiple ML Models**: Support for Naive Bayes, Logistic Regression, SVM, and Random Forest
- **Interactive UI**: Modern, responsive interface with real-time results
- **Model Training**: Train custom models on your dataset
- **Probability Visualization**: See confidence scores and probability distributions
- **Sample Texts**: Quick-start examples for testing

## Project Structure

```
AI Sentiment Analyzer/
|-- app.py                
|-- model.py              
|-- dataset.csv            
|-- requirements.txt       
|-- README.md           
|-- templates/
|   |-- index.html      
|-- static/
|   |-- style.css         
|   |-- script.js         
```

## Installation

1. **Clone or download the project** to your local machine

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to `http://localhost:5000`

### Using the Application

1. **Train the Model** (first time only):
   - Click the "Train Model" button to train the sentiment analysis model
   - The model will be trained using the provided dataset.csv

2. **Analyze Text**:
   - Enter text in the input area or use sample texts
   - Click "Analyze Sentiment" to get results
   - View sentiment, confidence score, and probability distribution

3. **Model Management**:
   - Check model status in the Model Management section
   - Retrain the model with updated data if needed

## API Endpoints

### `GET /`
- Returns the main web interface

### `POST /analyze`
- Analyzes sentiment of provided text
- **Request**: `{"text": "your text here"}`
- **Response**: 
  ```json
  {
    "success": true,
    "sentiment": "positive",
    "confidence": 0.85,
    "probabilities": {
      "positive": 0.85,
      "neutral": 0.10,
      "negative": 0.05
    }
  }
  ```

### `POST /train`
- Trains the sentiment analysis model
- **Response**:
  ```json
  {
    "success": true,
    "message": "Model trained successfully",
    "metrics": {
      "accuracy": 0.92,
      "training_samples": 120,
      "test_samples": 30,
      "model_type": "naive_bayes"
    }
  }
  ```

### `GET /model_info`
- Returns information about the current model
- **Response**:
  ```json
  {
    "is_trained": true,
    "model_type": "naive_bayes",
    "supported_sentiments": ["negative", "neutral", "positive"],
    "label_mapping": {"negative": 0, "neutral": 1, "positive": 2}
  }
  ```

## Dataset Format

The training dataset should be a CSV file with two columns:

```csv
text,sentiment
"I love this product!",positive
"This is terrible.",negative
"The weather is okay.",neutral
```

**Supported sentiment labels**: `positive`, `negative`, `neutral`

## Model Details

### Supported Algorithms

- **Naive Bayes** (default): Fast and effective for text classification
- **Logistic Regression**: Good performance with interpretable results
- **SVM**: Excellent for high-dimensional data
- **Random Forest**: Robust ensemble method

### Text Preprocessing

- Lowercase conversion
- URL and mention removal
- Punctuation and number removal
- Stopword filtering
- Tokenization using NLTK

### Feature Extraction

- TF-IDF Vectorization
- N-gram support (1-2 grams)
- Maximum 5000 features
- Minimum document frequency of 2

## Customization

### Adding Your Own Dataset

1. Replace `dataset.csv` with your own data
2. Ensure your CSV has `text` and `sentiment` columns
3. Use sentiment labels: `positive`, `negative`, `neutral`
4. Click "Train Model" to retrain

### Changing the Model Type

Modify the `train_model()` function in `app.py` to use a different algorithm:

```python
metrics = sentiment_analyzer.train('dataset.csv', model_type='logistic_regression')
```

Available options: `'naive_bayes'`, `'logistic_regression'`, `'svm'`, `'random_forest'`

## Technologies Used

- **Backend**: Flask, Python
- **Machine Learning**: Scikit-learn, NLTK, Pandas, NumPy
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Styling**: CSS Grid, Flexbox, CSS Variables
- **Deployment**: Gunicorn (production-ready)

## Performance

- **Training Time**: ~2-5 seconds for 150 samples
- **Prediction Time**: <100ms per request
- **Memory Usage**: ~50-100MB
- **Accuracy**: 85-95% (depending on dataset quality)

## Troubleshooting

### Common Issues

1. **Model not trained error**:
   - Ensure you've clicked "Train Model" first
   - Check that `dataset.csv` exists and is properly formatted

2. **NLTK download errors**:
   - The application automatically downloads required NLTK data
   - Ensure you have an internet connection for the first run

3. **Port already in use**:
   - Change the port in `app.py`: `app.run(port=5001)`

4. **Memory issues**:
   - Reduce `max_features` in the TF-IDF vectorizer
   - Use a smaller dataset for training

### Debug Mode

Enable debug mode by setting:
```python
app.run(debug=True)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Future Enhancements

- [ ] Support for custom model upload
- [ ] Batch text analysis
- [ ] Sentiment trend analysis
- [ ] Multi-language support
- [ ] Export results functionality
- [ ] Advanced text preprocessing options
- [ ] Model performance comparison
- [ ] REST API documentation

<img width="1920" height="904" alt="Screenshot 2026-04-15 101646" src="https://github.com/user-attachments/assets/fd5a39e9-4d15-4f49-a776-41702bddb619" />

