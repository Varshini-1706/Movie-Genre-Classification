from flask import Flask, request, render_template
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the trained model, vectorizer, and encoder
with open('naive_bayes_model.pkl', 'rb') as model_file:
    naive_bayes_model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open('encoder.pkl', 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)

# Route for the homepage (index)
@app.route('/')
def index():
    return render_template('index.html')

# Route for the prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the confession text from the form
        confession_text = request.form['confession_text']
        
        # Vectorize the confession using the saved vectorizer
        confession_vectorized = vectorizer.transform([confession_text])
        
        # Predict the genre using the trained model
        prediction = naive_bayes_model.predict(confession_vectorized)
        
        # Decode the prediction using the encoder
        predicted_genre = encoder.inverse_transform(prediction)
        
        return render_template('result.html', genre=predicted_genre[0])

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
