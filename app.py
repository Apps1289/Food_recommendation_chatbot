import os
import pickle
import time
import re
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from nltk.corpus import stopwords
import pandas as pd
import nltk
# Download stopwords if not already present (runs every time for safety)
nltk.download('stopwords', quiet=True) 

# --- Dependency Check and Sentence Transformer Setup ---
SENTENCE_TRANSFORMERS_AVAILABLE = False
LIBRARIES_AVAILABLE = True
try:
    from sentence_transformers import SentenceTransformer
    # NOTE: The model path/name must match what was used in the notebook
    # 'all-MiniLM-L6-v2' is assumed based on the notebook content
    SENTENCE_MODEL_NAME = 'all-MiniLM-L6-v2' 
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print(f"⚠️ Warning: sentence-transformers not found. Prediction will fail if model relies on it. Please install: pip install sentence-transformers")
    
# Initialize Flask app
app = Flask(__name__)
CORS(app) 

# --- File Paths ---
NLP_MODEL_PATH = 'nlp_model.pkl'
DATA_ENGINE_PATH = 'diet_engine.pkl'

# --- Placeholder/Global DataFrames ---
HEALTH_CONDITIONS_DF = None
FOOD_DF = None
ICMR_RDA_DF = None

# --- NutriCareNLP Class (Fixed Prediction Logic) ---

class NutriCareNLP:
    def __init__(self, nlp_model_path):
        self.symptom_classifier = None
        self.intent_classifier = None
        self.stop_words = set(stopwords.words('english'))
        self.sentence_model = None
        self.model_loaded = False
        
        # 1. Load the SentenceTransformer model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer(SENTENCE_MODEL_NAME) 
                print("✅ SentenceTransformer loaded for contextual embeddings.")
            except Exception as e:
                print(f"⚠️ Failed to initialize SentenceTransformer ({SENTENCE_MODEL_NAME}): {e}. Diagnostic prediction will fail.")
                self.sentence_model = None
        
        # 2. Load the trained classifiers
        try:
            with open(nlp_model_path, 'rb') as f:
                loaded_models = pickle.load(f)

            if isinstance(loaded_models, dict):
                # Retrieve the classifier that was trained on the embeddings
                self.symptom_classifier = loaded_models.get('symptom_classifier') or loaded_models.get('classifier')
                self.intent_classifier = loaded_models.get('intent_classifier')
            elif hasattr(loaded_models, 'symptom_classifier'):
                self.symptom_classifier = loaded_models.symptom_classifier
                self.intent_classifier = loaded_models.intent_classifier
            else:
                if hasattr(loaded_models, 'predict'):
                     self.symptom_classifier = loaded_models
                     print("ℹ️ Loaded object seems to be the classifier directly.")
            
            # Simple check if the core model loaded successfully
            if self.symptom_classifier is not None and hasattr(self.symptom_classifier, 'predict'):
                self.model_loaded = True
                print(f"✅ NutriCareNLP classifiers loaded from {nlp_model_path}.")
            else:
                 print(f"❌ Error: Could not find a valid symptom classifier in {nlp_model_path}.")

        except FileNotFoundError:
            print(f"❌ Error: NLP model file not found at {nlp_model_path}.")
        except Exception as e:
            print(f"❌ Error loading NLP model from pickle: {e}")

    def preprocess_text(self, text):
        """Preprocess text for NLP, matching the notebook's logic."""
        if pd.isna(text): return ""
        text = str(text).lower()
        # Remove non-alphabetic characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove stop words and short words (length > 2)
        return ' '.join([word for word in text.split() if word not in self.stop_words and len(word) > 2])

    def predict_condition(self, symptom_text):
        """
        Predict the simplified health category.
        
        CRITICAL FIX: This now relies entirely on generating the embedding 
        first, as the saved classifier is LogisticRegression trained on embeddings.
        """
        # Fallback if no model is available
        if not self.model_loaded or self.symptom_classifier is None: 
            return "general_health", 0.0
        
        processed_text = self.preprocess_text(symptom_text)
        if not processed_text: return "general_health", 0.0

        try:
            # 1. CRITICAL: Check for SentenceTransformer availability
            if self.sentence_model is None:
                # If the SentenceTransformer failed to load, we cannot generate embeddings.
                print("Warning: SentenceTransformer not available. Cannot generate embeddings for prediction.")
                return "general_health", 0.0

            # 2. Generate the embeddings for the input text
            # This is the expected input format for the saved LogisticRegression model.
            X_data = self.sentence_model.encode([processed_text])

            # 3. Predict using the LogisticRegression model
            prediction = self.symptom_classifier.predict(X_data)[0]
            
            # 4. Calculate confidence
            probabilities = self.symptom_classifier.predict_proba(X_data)[0] if hasattr(self.symptom_classifier, 'predict_proba') else []
            confidence = max(probabilities) if len(probabilities) > 0 else 0.0
            
            print(f"Prediction: {prediction}, Confidence: {round(confidence, 2)}")
            return prediction, confidence
            
        except Exception as e:
            # This catches any runtime failure during embedding or prediction.
            print(f"Error during NLP prediction process: {e}. Falling back to general_health.")
            return "general_health", 0.0

# --- ICMRDietEngine Class (Simplified for Backend API) ---

class ICMRDietEngine:
    def __init__(self, health_df):
        self.health_df = health_df
        self.health_map = self._create_lookup_map()
        print("✅ ICMRDietEngine initialized with health data.")

    def _create_lookup_map(self):
        """Convert the health DataFrame into a faster dictionary lookup."""
        health_map = {}
        for _, row in self.health_df.iterrows():
            condition = row['condition']
            health_map[condition] = {
                'avoid': ', '.join([s.strip() for s in row['avoid_keywords'].split(',')]),
                'recommend': ', '.join([s.strip() for s in row['recommend_keywords'].split(',')]),
                'nutrients': ', '.join([s.strip() for s in row['key_nutrients'].split(',')]),
                'advice': row['meal_advice'].strip()
            }
        return health_map

    def get_recommendations(self, predicted_category):
        """
        Maps the broad dietary category (predicted by NLP model) 
        to a specific recommendation set.
        """
        # Mapping from predicted NLP category to a representative condition 
        category_to_condition = {
            'metabolic_cardio': 'diabetes', 
            'low_sodium_renal': 'hypertension',
            'deficiency_hormonal': 'anemia',
            'digestive_gi': 'gastritis',
            'general_health': 'general_health',
            # Add mappings for the augmented categories (if present, use a general mapping)
            'deficiency_hormonal': 'anemia', # Mapping the general category to a representative disease
            'low_sodium_renal': 'kidney_disease',
            'digestive_gi': 'ibs'
        }
        
        target_condition = category_to_condition.get(predicted_category, 'general_health')

        # Retrieve the structured recommendation data
        # Fallback check needed if target_condition isn't a direct key in the final health_conditions_df
        return self.health_map.get(target_condition, self.health_map.get('general_health'))


# --- Global Model Loading ---

MODELS = {}

def load_models_and_data():
    """Loads the NLP model and all required DataFrames from pickle files."""
    global HEALTH_CONDITIONS_DF, FOOD_DF, ICMR_RDA_DF

    print("Attempting to load data and models...")

    # 1. Load DataFrames from diet_engine.pkl
    try:
        with open(DATA_ENGINE_PATH, 'rb') as f:
            loaded_data = pickle.load(f)
            
        # Robustly find the necessary DataFrames
        if isinstance(loaded_data, dict):
            HEALTH_CONDITIONS_DF = loaded_data.get('health_conditions_df') or loaded_data.get('health_df')
            FOOD_DF = loaded_data.get('food_df')
            ICMR_RDA_DF = loaded_data.get('icmr_rda_df')
        else:
            # Attempt to extract if the class instance was pickled directly
            HEALTH_CONDITIONS_DF = getattr(loaded_data, 'health_df', None)
            FOOD_DF = getattr(loaded_data, 'food_df', None)
            ICMR_RDA_DF = getattr(loaded_data, 'icmr_rda_df', None)

        if HEALTH_CONDITIONS_DF is None:
            raise KeyError("No health dataframe found in diet_engine.pkl")

        print(f"✅ DataFrames loaded successfully (Health rows: {getattr(HEALTH_CONDITIONS_DF, 'shape', 'unknown')[0]})")


    except Exception as e:
        print(f"❌ Error loading data from pickle: {e}. Using internal fallback health data.")
        # Fallback to minimal working data
        fallback = {
            'condition': ['general_health', 'diabetes', 'hypertension'],
            'avoid_keywords': ['junk,processed,excessive sugar,trans fats', 'sugar,sweet,refined,maida,white rice,fried', 'salt,pickle,papad,processed,namkeen,chips'],
            'recommend_keywords': ['balanced,variety,whole grains,vegetables,fruits', 'brown rice,oats,ragi,bajra,vegetables,dal,salad', 'fruits,vegetables,whole grains,low sodium,potassium rich'],
            'key_nutrients': ['balanced macro and micronutrients', 'fiber,chromium,magnesium,complex carbs', 'potassium,magnesium,calcium,low sodium'],
            'meal_advice': ['Balanced nutrition, variety, regular meal times', 'Small frequent meals, avoid skipping, consistent timing', 'Regular meal times, reduce portion size, avoid late night eating']
        }
        HEALTH_CONDITIONS_DF = pd.DataFrame(fallback)

    # 2. Initialize NLP Model (loads the nlp_model.pkl)
    MODELS['nlp_model'] = NutriCareNLP(NLP_MODEL_PATH)

    # 3. Initialize Diet Engine (uses the loaded data)
    MODELS['diet_engine'] = ICMRDietEngine(HEALTH_CONDITIONS_DF)

    print("✅ System ready for chat interaction.")

# Load models on startup
with app.app_context():
    load_models_and_data()


# --- API Endpoint ---
@app.route('/', methods=['GET'])
def serve_index():
    try:
        web_root = os.path.dirname(os.path.abspath(__file__))
        return send_from_directory(web_root, 'index.html')
    except Exception:
        return "Index not found. You can open index.html directly from the project folder.", 404

@app.route('/chat', methods=['POST'])
def chat():
    # Basic check for model readiness
    if 'nlp_model' not in MODELS or 'diet_engine' not in MODELS:
        return jsonify({
            'condition': 'Initialization Error',
            'recommendation': 'Server models failed to load. Check console for file path errors.',
            'disclaimer': 'Consult a professional before making major dietary changes.'
        }), 500

    data = request.get_json()
    query = data.get('query', 'General health check.')

    if not query:
        return jsonify({
            'condition': 'None Specified',
            'recommendation': 'Please provide a valid query describing your health condition.',
            'disclaimer': 'Consult a professional before making major dietary changes.'
        }), 400

    # Simulate typing delay for better UX
    time.sleep(0.7)

    try:
        nlp_model = MODELS['nlp_model']
        diet_engine = MODELS['diet_engine']

        # 2. Predict Condition using NLP Model
        predicted_category, confidence = nlp_model.predict_condition(query)
        
        # 3. Confidence Threshold Logic
        confidence_threshold = 0.60 # Set a reasonable threshold
        if not nlp_model.model_loaded or nlp_model.sentence_model is None:
            # If the necessary model components aren't loaded, force general health
            predicted_category = 'general_health'
            confidence = 0.0
            print("Forcing general_health due to missing NLP components.")
        elif confidence < confidence_threshold and predicted_category != 'general_health':
            print(f"Low confidence ({round(confidence, 2)}) for '{predicted_category}'. Falling back to general health.")
            predicted_category = 'general_health'

        # 4. Get Recommendations using Diet Engine
        recommendations = diet_engine.get_recommendations(predicted_category)
        
        # 5. Prepare Response
        response = {
            'condition': predicted_category.replace('_', ' ').title(),
            'recommendation': recommendations['recommend'],
            'avoid': recommendations['avoid'],
            'nutrients': recommendations['nutrients'],
            'advice': recommendations['advice'],
            'disclaimer': 'Disclaimer: This is a diet recommendation based on general guidelines. Always consult a healthcare professional before making significant changes to your diet, especially when managing a chronic disease.'
        }
        # For debugging/verification
        # response['confidence'] = round(confidence * 100, 1)

        return jsonify(response)

    except Exception as e:
        print(f"An unexpected error occurred during chat processing: {e}")
        return jsonify({
            'condition': 'System Error',
            'recommendation': f'An unexpected server error occurred. Please check the backend console for details. Error: {str(e)}',
            'disclaimer': 'Consult a professional before making major dietary changes.'
        }), 500

# --- Running the Server ---

if __name__ == '__main__':
    # Flask runs on 127.0.0.1:5000, matching the frontend's CHAT_API_URL
    # Note: Flask runs twice in debug mode, which is fine for this setup.
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)