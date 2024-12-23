import pickle
from flask import Flask, render_template, request
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# Initialize Flask app
app = Flask(__name__)

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the PorterStemmer
ps = PorterStemmer()

# Text preprocessing function
def transform_text(message):
    # Lowercasing the message
    text = message.lower()
    
    # Word level tokenization
    text = nltk.word_tokenize(text)
    
    # Removing non-alphanumeric characters
    y = [i for i in text if i.isalnum()]
    text = y[:]
    
    # Removing stopwords and punctuation
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    # Stemming the words
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Spam prediction function
def predict_spam(message):
    # Preprocess the message
    transformed_sms = transform_text(message)
    # Vectorize the preprocessed message
    vector_input = tfidf.transform([transformed_sms])
    # Predict using the ML model
    result = model.predict(vector_input)[0]

    return result

# Homepage route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_sms = request.form['message']
        result = predict_spam(input_sms)
        # Pass the result to the HTML template
        return render_template('index.html', result=result)

# Load the vectorizer and model at the start
if __name__ == '__main__':
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    app.run(debug=True)
