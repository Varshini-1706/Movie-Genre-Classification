# save_models.py

import os
import joblib

# Check if the 'models' folder exists, if not, create it
if not os.path.exists('models'):
    os.makedirs('models')

# Assuming you have already trained your models (naive_bayes_model, vectorizer, encoder)
# Save models in the 'models' folder
joblib.dump(naive_bayes_model, 'models/naive_bayes_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')
joblib.dump(encoder, 'models/encoder.pkl')

print("Models saved successfully!")
