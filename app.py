import os
import pickle
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from huggingface_hub import InferenceClient
import pandas as pd

# --- CONFIGURATION ---
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NutriCare")

# Setup Flask to serve index.html from the current directory
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

client = InferenceClient(api_key=HF_TOKEN)

# --- REQUIRED CLASS DEFINITIONS FOR PICKLE ---
# These MUST match the names used in your training notebook
class NutriCareNLP:
    def __init__(self):
        self.symptom_classifier = None
        self.intent_classifier = None

class ICMRDietEngine:
    def __init__(self, food_df, health_df, icmr_rda_df):
        self.food_df = food_df
        self.health_df = health_df
        self.icmr_rda_df = icmr_rda_df

# --- LOAD MODELS ---
def load_models():
    try:
        # Load NLP Model
        with open('nlp_model.pkl', 'rb') as f:
            nlp = pickle.load(f)
        
        # Load Diet Engine
        with open('diet_engine.pkl', 'rb') as f:
            engine = pickle.load(f)
            
        logger.info("✅ Models loaded successfully!")
        return nlp, engine
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        return None, None

nlp_model, diet_engine = load_models()

# --- SUPPORTED DISEASES LIST ---
SUPPORTED_DISEASES = [
    'diabetes', 'hypertension', 'anemia', 'obesity', 'heart_disease',
    'kidney_disease', 'thyroid', 'pcod', 'gastritis', 'osteoporosis',
    'high_cholesterol', 'fatty_liver', 'ibs'
]

# --- AI WRAPPER ---
def get_llm_response(system_prompt, user_prompt):
    try:
        response = client.chat.completions.create(
            model=HF_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=450,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"HF API Error: {e}")
        return "I'm having trouble connecting to my AI brain. Please try again later!"

# --- ROUTES ---

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(force=True)
    query = data.get('query', '').lower().strip()

    # 1. Intent/Disease Detection
    detected_condition = None
    for condition in SUPPORTED_DISEASES:
        if condition.replace('_', ' ') in query:
            detected_condition = condition
            break

    # 2. Routing logic
    if detected_condition:
        # PULL DATA FROM YOUR MODEL
        # Ensure your health_df is accessible from the engine
        row = diet_engine.health_df[diet_engine.health_df['condition'] == detected_condition].iloc[0]
        
        system_msg = "You are a warm and expert AI Dietitian named Nutri-Bot."
        user_msg = (
            f"The user is asking about {detected_condition}. "
            f"Our internal data suggests avoiding: {row['avoid_keywords']}. "
            f"Focus on: {row['recommend_keywords']}. "
            f"Meal advice: {row['meal_advice']}. "
            "Please turn this data into a friendly, human-like conversation. "
            "Use bullet points for food lists and include a medical disclaimer."
        )
        final_reply = get_llm_response(system_msg, user_msg)
    else:
        # PURE CONVERSATION / UNKNOWN DISEASE
        system_msg = "You are Nutri-Bot, a helpful AI Dietitian. Provide general dietary advice for health queries."
        final_reply = get_llm_response(system_msg, query)

    return jsonify({'recommendation': final_reply})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)