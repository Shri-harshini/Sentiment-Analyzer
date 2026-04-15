# Sentiment Analyzer - Setup & Run Guide

## Quick Start

Follow these steps to get your AI Sentiment Analyzer running:

### Step 1: Install Dependencies

```bash
# Create and activate virtual environment (recommended)
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

# Install all required packages
pip install -r requirements.txt
```

### Step 2: Train the Model

```bash
# Train the sentiment analysis model
python train_model.py
```

This will:
- Create a sample dataset (sentiment_dataset.csv)
- Train a Naive Bayes classifier
- Save the trained model as `model.pkl`
- Save the TF-IDF vectorizer as `vectorizer.pkl`

### Step 3: Run the Flask Application

```bash
# Start the Flask development server
python app.py
```

The server will start on `http://localhost:5000`

### Step 4: Open in Browser

Navigate to `http://localhost:5000` in your web browser to use the sentiment analyzer.

## Requirements.txt

```txt
Flask==2.3.3
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
nltk==3.8.1
gunicorn==21.2.0
```

## Detailed Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Web browser (Chrome, Firefox, Safari, etc.)

### Step-by-Step Setup

#### 1. Project Setup
```bash
# Navigate to your project directory
cd "AI Sentiment Analyzer"

# Create virtual environment (optional but recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

#### 2. Install Dependencies
```bash
# Install all required Python packages
pip install -r requirements.txt

# Verify installation
pip list
```

#### 3. Train the Model
```bash
# Run the training script
python train_model.py
```

Expected output:
```
=== Sentiment Analysis Model Training ===

1. Creating sample dataset...
   Dataset created with 30 samples
   Distribution: {'positive': 10, 'negative': 10, 'neutral': 10}

2. Preprocessing text data...
   Text preprocessing completed

3. Splitting data into train and test sets...
   Training samples: 21
   Test samples: 9

4. Creating TF-IDF vectorizer...
   Vocabulary size: 85
   Feature matrix shape: (21, 85)

5. Training Naive Bayes classifier...
   Model training completed

6. Evaluating model performance...
   Accuracy: 0.8889 (88.89%)

   Classification Report:
              precision    recall  f1-score   support

    negative       1.00      0.67      0.80         3
     neutral       0.75      1.00      0.86         3
    positive       1.00      1.00      1.00         3

    accuracy                           0.89         9
   macro avg       0.92      0.89      0.89         9
weighted avg       0.92      0.89      0.89         9

7. Saving model and vectorizer...
   Model saved as 'model.pkl'
   Vectorizer saved as 'vectorizer.pkl'

=== Training Complete! ===
```

#### 4. Run the Web Application
```bash
# Start Flask server
python app.py
```

Expected output:
```
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment.
 * Running on http://127.0.0.1:5000
 * Running on http://localhost:5000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 123-456-789
```

#### 5. Access the Application
Open your web browser and navigate to:
- `http://localhost:5000`
- `http://127.0.0.1:5000`

## Using the Application

### First Time Setup
1. **Train Model**: Click the "Train Model" button if the model shows "Not Trained"
2. **Wait for Training**: The model will train using the sample dataset
3. **Ready to Use**: Once trained, you can start analyzing text

### Analyzing Text
1. **Enter Text**: Type or paste text in the input area
2. **Click Analyze**: Press "Analyze Sentiment" or use Ctrl+Enter
3. **View Results**: See sentiment, confidence, and probability distribution
4. **Try Examples**: Use the sample text buttons for quick testing

## Troubleshooting

### Common Issues

#### 1. Module Not Found Error
```bash
# Solution: Install missing dependencies
pip install flask pandas numpy scikit-learn nltk
```

#### 2. NLTK Data Download Error
The training script automatically downloads required NLTK data. If you encounter issues:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

#### 3. Port Already in Use
```bash
# Solution: Change port in app.py
# Find this line and change the port number:
app.run(debug=True, host='0.0.0.0', port=5001)
```

#### 4. Model Not Found
```bash
# Solution: Ensure you've trained the model first
python train_model.py
```

#### 5. Permission Denied (Windows)
```bash
# Solution: Run PowerShell as Administrator
# Or use:
python -m pip install -r requirements.txt --user
```

### Performance Tips

- **For Better Performance**: Use a larger dataset for training
- **Memory Issues**: Reduce `max_features` in the TF-IDF vectorizer
- **Slow Training**: Use a smaller dataset or fewer features

## Advanced Usage

### Using Custom Dataset
1. Replace `dataset.csv` with your own data
2. Format: `text,sentiment` columns
3. Sentiment labels: `positive`, `negative`, `neutral`
4. Retrain the model: `python train_model.py`

### Production Deployment
```bash
# Use Gunicorn for production
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### API Usage
```bash
# Analyze text via API
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'
```

## File Structure

```
AI Sentiment Analyzer/
|-- app.py                 # Main Flask application
|-- model.py               # Sentiment analysis model
|-- train_model.py         # Model training script
|-- dataset.csv            # Sample training data
|-- requirements.txt       # Python dependencies
|-- model.pkl             # Trained model (generated)
|-- vectorizer.pkl        # TF-IDF vectorizer (generated)
|-- templates/
|   |-- index.html        # Frontend interface
|-- static/
|   |-- style.css         # Styling
|   |-- script.js         # Frontend JavaScript
|   |-- predict.js        # Enhanced JavaScript
|   |-- enhanced-style.css # Premium CSS
|-- README_SETUP.md        # This setup guide
```

## Support

If you encounter any issues:
1. Check that all dependencies are installed
2. Ensure the model is trained before analyzing
3. Verify the server is running on the correct port
4. Check browser console for JavaScript errors

Enjoy using your AI Sentiment Analyzer!
