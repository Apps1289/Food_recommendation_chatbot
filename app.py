# app.py — Nutri-Bot backend (complete and robust)
# Save this as app.py

import os
import re
import json
import pickle
import logging
import random
from typing import Tuple, Any, Dict
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Optional heavy transformer dependency
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# NLTK (for stopwords / wordnet used in gibberish heuristics)
try:
    import nltk
    from nltk.corpus import stopwords, wordnet
    try:
        nltk.data.find('corpora/stopwords')
    except Exception:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except Exception:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
except Exception:
    stopwords = None
    wordnet = None

import pandas as pd

# -------------------------
# Config & Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NutriCare")

NLP_MODEL_PATH = 'nlp_model.pkl'
DIET_ENGINE_PATH = 'diet_engine.pkl'
SENTENCE_MODEL_NAME = 'all-MiniLM-L6-v2'
CONFIDENCE_THRESHOLD = 0.30
DEBUG_MODE = True

app = Flask(__name__, static_folder='.')
CORS(app)


# -------------------------
# Expanded disease synonyms (canonical_key -> many variants)
# -------------------------
DISEASE_SYNONYMS = {
    'diabetes': [
        'diabetes', 'type 2 diabetes', 'type2 diabetes', 'type-2 diabetes',
        'type ii diabetes', 't2d', 't2dm', 'diabetic', 'sugar disease',
        'high blood sugar','high blood glucose','hyperglycemia'
    ],
    'hypertension': [
        'hypertension', 'high blood pressure', 'high bp', 'bp', 'hypertensive',
        'elevated blood pressure','raised bp'
    ],
    'anemia': [
        'anemia', 'anaemia', 'low hb', 'low hemoglobin', 'iron deficiency',
        'ferritin low','low ferritin','pale','pallor'
    ],
    'gastritis': [
        'gastritis', 'acid reflux', 'heartburn', 'acidity', 'reflux', 'gastroesophageal reflux',
        'gerd', 'acid reflux disease'
    ],
    'ibs': [
        'ibs', 'irritable bowel', 'irritable bowel syndrome', 'ibd', 'bowel syndrome',
        'constipation and diarrhea','alternating diarrhea constipation'
    ],
    'obesity': [
        'obesity', 'overweight', 'obese', 'high bmi', 'bmi high', 'excess weight'
    ],
    'thyroid': [
        'thyroid', 'hypothyroid', 'hyperthyroid', 'tsh', 'thyroiditis', 'hashimotos',
        'graves'
    ],
    'kidney_disease': [
        'kidney disease', 'ckd', 'chronic kidney disease', 'kidney failure', 'renal failure',
        'reduced gfr', 'creatinine high', 'protein in urine','nephropathy'
    ],
    'fatty_liver': [
        'fatty liver', 'nafld', 'non alcoholic fatty liver', 'fatty liver disease',
        'elevated lft','high alt','high ast'
    ],
    'high_cholesterol': [
        'high cholesterol', 'cholesterol', 'ldl', 'hdl low', 'high ldl', 'hyperlipidemia',
        'high triglycerides','dyslipidemia'
    ],
    'pcod': [
        'pcod','pcos','polycystic ovary syndrome','polycystic ovarian syndrome'
    ],
    'heart_disease': [
        'heart disease','heart problem','cardiac','angina','coronary artery disease',
        'cad','heart attack risk','palpitations','arrhythmia'
    ],
    'osteoporosis': [
        'osteoporosis','bone loss','low bone density','dexa','fragile bones'
    ],
    'general_health': [
        'general health','healthy','wellbeing','overall health'
    ]
}

# build reverse lookup map
DISEASE_REVERSE = {}
for key, variants in DISEASE_SYNONYMS.items():
    for v in variants:
        DISEASE_REVERSE[v.lower()] = key

# We create the regex group for detection by prioritizing longer phrases first.
DISEASE_PHRASES = sorted(DISEASE_REVERSE.keys(), key=lambda s: -len(s))
DISEASE_REGEX_GROUP = r'|'.join([re.escape(k) for k in DISEASE_PHRASES])

# --- FIXED REGEX PATTERNS ---
EXPLICIT_PATTERNS = [
    re.compile(rf"\b(?:i have|i am|i'm|i was diagnosed with|diagnosed with|diagnosed)\s+(?P<disease>({DISEASE_REGEX_GROUP}))\b", re.I),
    re.compile(rf"^\s*(?P<disease>({DISEASE_REGEX_GROUP}))\s*$", re.I)
]
# ----------------------------

def extract_explicit_disease(text: str):
    """
    Attempts multiple strategies to explicitly identify a health condition.
    Returns (canonical_key, raw_found) or None
    """
    if not text:
        return None

    t = text.lower().strip()

    # 1) Try the explicit regex patterns (Pattern 1, Pattern 2)
    for pat in EXPLICIT_PATTERNS:
        m = pat.search(t)
        if m:
            found = m.group('disease').strip().lower()
            canon = DISEASE_REVERSE.get(found)
            if canon:
                return canon, found

    # 2) Fallback to substring matching for less explicit or complex inputs
    for phrase, canon in DISEASE_REVERSE.items():
        if phrase in t:
            if len(phrase.split()) > 1 or len(phrase) > 5:
                 return canon, phrase
            
    # 3) Check if the entire, cleaned input is a known synonym (e.g., 'ibs')
    if t in DISEASE_REVERSE:
         return DISEASE_REVERSE[t], t
         
    return None


# -------------------------
# Minimal fallback data
# -------------------------
def get_fallback_health_df():
    data = {
        'condition': [
            'diabetes','hypertension','anemia','obesity','heart_disease',
            'kidney_disease','thyroid','pcod','gastritis','osteoporosis',
            'high_cholesterol','fatty_liver','ibs','general_health'
        ],
        'avoid_keywords': [
            'sugar,sweet,refined,maida,white rice,fried',
            'salt,pickle,papad,processed,namkeen,chips',
            'tea,coffee,calcium rich with iron',
            'fried,oily,sweet,refined,junk,fast food',
            'ghee,butter,coconut,fried,fatty,red meat',
            'protein rich,dal heavy,meat,high potassium',
            'cabbage,cauliflower,broccoli,soy,raw cruciferous',
            'sugar,refined carbs,dairy excess,processed',
            'spicy,chili,citrus,fried,oily,acidic',
            'oxalate rich,spinach excess,salt',
            'ghee,butter,coconut oil,fried,fatty',
            'alcohol,sugar,refined,fried,fatty',
            'spicy,oily,dairy,gluten,processed',
            'junk,processed,excessive sugar,trans fats'
        ],
        'recommend_keywords': [
            'brown rice,oats,ragi,bajra,vegetables,dal,salad',
            'fruits,vegetables,whole grains,low sodium,potassium rich',
            'iron rich,green leafy,dates,jaggery,vitamin c,citrus',
            'vegetables,salad,whole grains,lean protein,fiber',
            'fish,nuts,olive oil,vegetables,fruits,whole grains',
            'low protein,rice,vegetables,controlled portions',
            'iodine rich,fish,eggs,selenium,cooked vegetables',
            'whole grains,protein,fiber,antioxidants,balanced',
            'bland,curd,banana,rice,boiled,steamed',
            'calcium rich,dairy,green leafy,sesame,ragi',
            'oats,barley,vegetables,fruits,nuts,fish',
            'vegetables,fruits,whole grains,lean protein',
            'probiotics,fiber,bland,easily digestible',
            'balanced,variety,whole grains,vegetables,fruits'
        ],
        'key_nutrients': [
            'fiber,chromium,magnesium,complex carbs',
            'potassium,magnesium,calcium,low sodium',
            'iron,vitamin c,folate,vitamin b12',
            'protein,fiber,low calories,vitamins',
            'omega 3,antioxidants,fiber,potassium',
            'controlled protein,low sodium,low potassium',
            'iodine,selenium,zinc,vitamin d',
            'chromium,omega 3,vitamin d,fiber',
            'probiotics,vitamin b,zinc,bland foods',
            'calcium,vitamin d,magnesium,phosphorus',
            'fiber,omega 3,antioxidants,plant sterols',
            'antioxidants,fiber,vitamins,minerals',
            'probiotics,fiber,vitamins,minerals',
            'balanced macro and micronutrients'
        ],
        'meal_advice': [
            'Small frequent meals, avoid skipping, consistent timing',
            'Regular meal times, reduce portion size, avoid late night eating',
            'Iron rich breakfast, vitamin C with meals, avoid tea with food',
            'Portion control, frequent small meals, avoid late dinner',
            'Heart healthy timing, avoid heavy dinner, regular intervals',
            'Controlled portions, avoid excess protein, fluid management',
            'Consistent meal timing, avoid goitrogens, cooked vegetables',
            'Balanced meals, avoid sugar spikes, regular eating pattern',
            'Small frequent meals, avoid empty stomach, bland foods',
            'Calcium rich meals, vitamin D exposure, regular intake',
            'Fiber rich meals, avoid saturated fats, regular timing',
            'Avoid alcohol, reduce sugar, increase vegetables',
            'Identify triggers, small meals, avoid stress eating',
            'Balanced nutrition, variety, regular meal times'
        ]
    }
    return pd.DataFrame(data)

# -------------------------
# PLACEHOLDER CLASSES FOR UNPICKLING (REQUIRED FIX)
# -------------------------

class ICMRDietEngine:
    """Placeholder for the class used in the pickled object (diet_engine.pkl)."""
    def __init__(self, food_df, health_df, icmr_rda_df):
        self.food_df = food_df
        self.health_df = health_df
        self.icmr_rda_df = icmr_rda_df

class NutriCareNLP:
    """Placeholder for the class structure used in nlp_model.pkl."""
    def __init__(self, path=None):
        self.symptom_classifier = None
        self.intent_classifier = None
        self.sentence_model = None
        self.model_loaded = False


# -------------------------
# Diet Engine Wrapper (actual operational class)
# -------------------------
class DietEngineWrapper:
    def __init__(self, obj: Any):
        self.health_map = {}
        self._build_from(obj)

    def _build_from(self, obj: Any):
        """
        Robustly extract a health DataFrame from various pickled formats.
        """
        df = None

        if isinstance(obj, dict):
            if 'health_conditions_df' in obj and isinstance(obj.get('health_conditions_df'), pd.DataFrame):
                df = obj.get('health_conditions_df')
            elif 'health_df' in obj and isinstance(obj.get('health_df'), pd.DataFrame):
                df = obj.get('health_df')
            elif 'health_df' in obj and not isinstance(obj.get('health_df'), pd.DataFrame):
                try:
                    df_candidate = pd.DataFrame(obj.get('health_df'))
                    if not df_candidate.empty:
                        df = df_candidate
                except Exception:
                    df = None
            elif isinstance(obj, ICMRDietEngine) and isinstance(obj.health_df, pd.DataFrame):
                 df = obj.health_df

        if df is None and isinstance(obj, pd.DataFrame):
            df = obj

        if df is None and hasattr(obj, 'health_df'):
            try:
                candidate = getattr(obj, 'health_df')
                if isinstance(candidate, pd.DataFrame):
                    df = candidate
                else:
                    try:
                        tmp = pd.DataFrame(candidate)
                        if not tmp.empty:
                            df = tmp
                    except Exception:
                        df = None
            except Exception:
                df = None

        if isinstance(df, pd.DataFrame):
            for _, row in df.iterrows():
                try:
                    condition_raw = row.get('condition', '')
                except Exception:
                    continue
                key = str(condition_raw).strip().lower().replace(' ', '_')
                self.health_map[key] = {
                    'avoid': ', '.join([s.strip() for s in str(row.get('avoid_keywords', '')).split(',') if s.strip()]),
                    'recommend': ', '.join([s.strip() for s in str(row.get('recommend_keywords', '')).split(',') if s.strip()]),
                    'nutrients': ', '.join([s.strip() for s in str(row.get('key_nutrients', '')).split(',') if s.strip()]),
                    'advice': str(row.get('meal_advice', '') or '')
                }
            logger.info("DietEngineWrapper: built health_map with %d conditions", len(self.health_map))
            return

        fallback = get_fallback_health_df()
        for _, row in fallback.iterrows():
            key = str(row['condition']).strip().lower().replace(' ', '_')
            self.health_map[key] = {
                'avoid': row['avoid_keywords'],
                'recommend': row['recommend_keywords'],
                'nutrients': row['key_nutrients'],
                'advice': row['meal_advice']
            }
        logger.warning("DietEngineWrapper: no health_df found in loaded object; using fallback table.")


    def get_recommendations(self, predicted_category: str) -> Dict[str,str]:
        category_to_condition = {
            'metabolic_cardio': 'diabetes',
            'low_sodium_renal': 'hypertension',
            'deficiency_hormonal': 'anemia',
            'digestive_gi': 'gastritis',
            'general_health': 'general_health'
        }
        key = str(predicted_category or '').strip().lower().replace(' ', '_')
        if key in self.health_map:
            return self.health_map[key]
        
        mapped = category_to_condition.get(key, 'general_health')
        return self.health_map.get(mapped, self.health_map.get('general_health', {'avoid':'','recommend':'','nutrients':'','advice':''}))


# -------------------------
# Fallback and Production NLP Logic
# -------------------------
class RuleNLP:
    # ... (RuleNLP definition remains the same)
    def __init__(self, sentence_model=None):
        self.sentence_model = sentence_model
        self.stop_words = set(stopwords.words('english')) if stopwords else set()
        self.KEYWORD_MAP = {
            'diabetes': ['diabetes','blood sugar','hba1c','type 2','type2','t2d','sugar'],
            'hypertension': ['blood pressure','hypertension','bp','high bp','salt'],
            'anemia': ['anemia','iron','ferritin','low hb','hemoglobin','pale'],
            'gastritis': ['gastritis','stomach','heartburn','acidity','reflux'],
            'ibs': ['ibs','irritable bowel','bloating','constipation','diarrhea'],
            'obesity': ['overweight','obesity','bmi','weight loss','weight gain'],
            'thyroid': ['thyroid','tsh','hypothyroid','hyperthyroid'],
            'heart_disease': ['chest pain','heart','angina','palpitations'],
            'kidney_disease': ['kidney','creatinine','gfr','protein in urine','ckd'],
            'fatty_liver': ['fatty liver','nafld','liver'],
            'high_cholesterol': ['cholesterol','ldl','triglyceride','lipid'],
            'pcod': ['pcod','pcos','polycystic'],
            'osteoporosis': ['osteoporosis','bone loss','dexA']
        }

    def preprocess(self, text: str) -> str:
        if not text: return ''
        t = str(text).lower()
        t = re.sub(r'[^a-z0-9\s\-\']', ' ', t)
        t = re.sub(r'\s+', ' ', t).strip()
        return t

    def predict_condition(self, text: str) -> Tuple[str,float]:
        p = self.preprocess(text)
        for cond, kws in self.KEYWORD_MAP.items():
            for kw in kws:
                if kw in p:
                    score = min(0.95, 0.4 + len(kw)/30.0)
                    return cond, score
        return 'general_health', 0.0

    def predict_intent(self, text: str) -> str:
        t = self.preprocess(text)
        if any(x in t for x in ['plan', 'diet', 'meal', 'eat', 'what should i eat', 'suggest food']):
            return 'diet_planning'
        if any(x in t for x in ['download','pdf','export','save']):
            return 'file_generation'
        if any(x in t for x in ['how many calories','bmi','calculate']):
            return 'calculation'
        if any(x in t for x in ['hello','hi','hey','namaste']):
            return 'greeting'
        return 'general'


class NutriCareNLP:
    # ... (NutriCareNLP definition remains the same)
    def __init__(self, path: str):
        self.symptom_classifier = None
        self.intent_classifier = None
        self.sentence_model = None
        self.model_loaded = False
        self.stop_words = set(stopwords.words('english')) if stopwords else set()

        loaded = None
        try:
            with open(path, 'rb') as f:
                loaded = pickle.load(f)
            logger.info("Loaded nlp pickle type: %s", type(loaded))
        except FileNotFoundError:
            logger.warning("nlp_model.pkl not found; using rule-based fallback.")
            loaded = None
        except Exception as e:
            logger.warning("Failed to load nlp pickle: %s", e)
            loaded = None

        if hasattr(loaded, 'symptom_classifier'):
            self.symptom_classifier = getattr(loaded, 'symptom_classifier', None)
            self.intent_classifier = getattr(loaded, 'intent_classifier', None)
        elif isinstance(loaded, dict):
            self.symptom_classifier = loaded.get('symptom_classifier') or loaded.get('classifier') or loaded.get('model')
            self.intent_classifier = loaded.get('intent_classifier') or loaded.get('intent_model')
        
        need_embeddings = bool(self.symptom_classifier and not hasattr(self.symptom_classifier, 'named_steps'))
        
        if need_embeddings and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer(SENTENCE_MODEL_NAME)
                logger.info("SentenceTransformer loaded for NLP wrapper.")
            except Exception as e:
                logger.warning("Could not init SentenceTransformer: %s", e)
                self.sentence_model = None

        if self.symptom_classifier is not None:
            self.model_loaded = True
            logger.info("NutriCareNLP: classifier loaded.")
        else:
            self.fallback = RuleNLP(sentence_model=self.sentence_model)
            logger.info("NutriCareNLP: using rule-based fallback.")


    def predict_condition(self, text: str) -> Tuple[str,float]:
        if self.model_loaded and self.symptom_classifier:
            processed = self._preprocess_for_model(text)
            try:
                if hasattr(self.symptom_classifier, 'named_steps'):
                    pred = self.symptom_classifier.predict([processed])[0]
                    prob = 0.0
                    if hasattr(self.symptom_classifier, 'predict_proba'):
                        prob = max(self.symptom_classifier.predict_proba([processed])[0])
                    return str(pred), float(prob)
            except Exception as e:
                logger.exception("Pipeline predict failed: %s", e)
            
            if self.sentence_model is not None:
                try:
                    emb = self.sentence_model.encode([processed])
                    pred = self.symptom_classifier.predict(emb)[0]
                    prob = 0.0
                    if hasattr(self.symptom_classifier, 'predict_proba'):
                        prob = max(self.symptom_classifier.predict_proba(emb)[0])
                    return str(pred), float(prob)
                except Exception as e:
                    logger.exception("Embedding predict failed: %s", e)
            
            return "general_health", 0.0
        else:
            return self.fallback.predict_condition(text)

    def predict_intent(self, text: str) -> str:
        if self.model_loaded and self.intent_classifier:
            try:
                proc = self._preprocess_for_model(text)
                if hasattr(self.intent_classifier, 'named_steps'):
                    return str(self.intent_classifier.predict([proc])[0])
                if self.sentence_model is not None:
                    emb = self.sentence_model.encode([proc])
                    return str(self.intent_classifier.predict(emb)[0])
            except Exception as e:
                logger.exception("Intent predict failed: %s", e)
            return "general"
        else:
            return self.fallback.predict_intent(text)

    def _preprocess_for_model(self, text: str) -> str:
        if not text:
            return ""
        t = str(text).lower()
        t = re.sub(r'[^a-z0-9\-\s]', ' ', t)
        t = re.sub(r'\s+', ' ', t).strip()
        tokens = [tok for tok in t.split() if tok not in self.stop_words]
        return ' '.join(tokens)


# ... (Gibberish detection and Load functions remain the same)
ENGLISH_WORDS = set()
try:
    if wordnet:
        for syn in wordnet.all_synsets():
            for lemma in syn.lemmas():
                ENGLISH_WORDS.add(lemma.name().lower().replace('_',' '))
except Exception:
    pass
ENGLISH_WORDS.update({'diabetes','hypertension','anemia','gastritis','thyroid','pcod','ibs','cholesterol','liver','kidney','obesity','bmi','bp','t2d'})

def is_gibberish(text: str, debug: bool=False) -> bool:
    if not text or not text.strip():
        if debug: logger.debug("gibberish: empty")
        return True
    s = str(text).lower()
    tokens = re.findall(r"[a-z0-9\-']+", s)
    if not tokens:
        if debug: logger.debug("gibberish: no tokens")
        return True
    valid = 0
    for tok in tokens:
        if len(tok) <= 1:
            continue
        if any(c.isdigit() for c in tok):
            valid += 1
            continue
        if tok in ENGLISH_WORDS:
            valid += 1
            continue
        if sum(1 for c in tok if c in 'aeiou') >= 2 and len(tok) >= 3:
            valid += 1
            continue
    if debug: logger.debug("gibberish tokens valid=%d total=%d", valid, len(tokens))
    return valid < 1

# *** MODIFIED humanize_recommendation FUNCTION ***

def humanize_recommendation(condition: str, recs: Dict[str,str]) -> str:
    """
    Consolidated function to provide the final advice and meal plan example 
    with a more humanized, conversational tone and clear structure.
    """
    friendly_condition = condition.replace('_', ' ').title()
    
    openings = [
        f"That's a great start! Based on your concern about **{friendly_condition}**, here is a simple, actionable summary of the dietary principles you should focus on:",
        f"Dealing with **{friendly_condition}** is manageable with the right diet. We've compiled the key food guidelines and an example plan for you:",
        f"Understood. To help manage **{friendly_condition}**, a consistent diet is crucial. Here are the core nutritional pointers and advice:",
        f"Thank you for sharing. We've structured an example diet plan specifically tailored to the needs of **{friendly_condition}**."
    ]
        
    empath = random.choice(openings)
    recommend = recs.get('recommend') or "whole grains, vegetables, fruits, and lean proteins."
    avoid = recs.get('avoid') or "highly processed foods, excess sugar, and trans fats."
    advice = recs.get('advice') or "regular meal times and adequate hydration."
    
    # Structure the advice using bolding and paragraphs for clarity
    advice_content = (
        f"**Suggested Diet (What to Embrace)**: {recommend}.\n\n"
        f"**Foods to Limit/Avoid (What to Minimize)**: {avoid}.\n\n"
        f"**Meal Timing & Lifestyle Advice**: {advice}."
    )

    # Add a concluding sentence
    conclusion = f"\n\nRemember, small, consistent changes are key to managing **{friendly_condition}** and improving your overall well-being! How else can I assist you today?"
    
    msg = f"{empath}\n\n{advice_content}{conclusion}"
           
    return msg

# **********************************************


def try_load_pickle(path: str):
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        logger.info("Loaded pickle: %s (type=%s)", path, type(obj))
        return obj
    except FileNotFoundError:
        logger.warning("Pickle not found: %s", path)
        return None
    except Exception as e:
        logger.warning("Error loading pickle %s: %s", path, e)
        return None

def load_models_and_data():
    loaded_de = try_load_pickle(DIET_ENGINE_PATH)
    
    if loaded_de is not None:
        diet = DietEngineWrapper(loaded_de)
    else:
        diet = DietEngineWrapper({'health_df': get_fallback_health_df()})

    nlp_loaded = try_load_pickle(NLP_MODEL_PATH)
    nlp = NutriCareNLP(NLP_MODEL_PATH)

    return nlp, diet

nlp_model, diet_engine = load_models_and_data()


# -------------------------
# Lightweight session store
# -------------------------
SESSION_STORE = {}
def get_session_id():
    ip = request.remote_addr or 'local'
    return f"{ip}"


# -------------------------
# Endpoints
# -------------------------
GREETING_KEYWORDS = {'hi','hello','hey','namaste','good morning','good evening'}
DISCLAIMER_TEXT = "This is a generic example diet and meal timing advice based on ICMR guidelines. Always consult a healthcare professional before making significant changes to your diet."


@app.route('/', methods=['GET'])
def index():
    root = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(root, 'index.html')):
        return send_from_directory(root, 'index.html')
    return "index.html not found. Place it in the same folder.", 404


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True, silent=True) or {}
        query = data.get('query') or data.get('text') or ''
        query = str(query).strip()
        session_id = get_session_id()
        # is_final_advice_request = data.get('is_final_advice', False) # REMOVED: No longer needed

        if not query:
            return jsonify({'next_action':'RESET','condition':'NoInput','recommendation':'Please type your question.'}), 200

        # --- PRE-CHECK: Load necessary data for debug ---
        predicted_label, confidence = nlp_model.predict_condition(query)
        explicit_result = extract_explicit_disease(query)
        intent = nlp_model.predict_intent(query)
        
        debug_info = {
            'query_received': query,
            'explicit_match': explicit_result,
            'predicted_label': predicted_label, 
            'confidence': f"{confidence:.2f}", 
            'intent': intent
        }
        # ----------------------------------------------

        # 0) Greeting check (Highest priority)
        if any(w in query.lower() for w in GREETING_KEYWORDS):
            return jsonify({
                'next_action': 'RESET',
                'condition': 'Greeting',
                'recommendation': "Hi! I'm Nutri-Bot 👋 — I can suggest gentle diet ideas and meal plans. What health concern should I know about?"
            }), 200

        # 0.5) Gibberish detection
        if is_gibberish(query, debug=DEBUG_MODE):
            return jsonify({
                'next_action':'RESET',
                'condition':'Gibberish',
                'recommendation': "I couldn't understand that — could you rephrase? For example: 'I have type 2 diabetes' or 'I have gastritis'."
            }), 200

        # --------------------------------------------------------------------------
        # REMOVED: LOGIC FOR HANDLING 'YES' AFTER INITIAL ADVICE - Consolidation means this is no longer needed
        # --------------------------------------------------------------------------

        # 1) Explicit disease extraction (highest priority)
        if explicit_result:
            canon_label, raw_found = explicit_result
            recs = diet_engine.get_recommendations(canon_label)

            # --- FORCED ADVICE DELIVERY (Skipping Confirmation) ---
            msg = humanize_recommendation(canon_label, recs) # No longer passing is_initial_advice
            
            SESSION_STORE[session_id] = {'condition': canon_label, 'last_query': query}
            
            resp = {
                'next_action': 'DELIVER_ADVICE', # Directly deliver the advice
                'condition': canon_label.replace('_',' ').title(),
                'recommendation': f"{msg}\n\n*{DISCLAIMER_TEXT}*", # INCLUDED DISCLAIMER HERE
                'avoid': recs.get('avoid'),
                'nutrients': recs.get('nutrients'),
                'advice': recs.get('advice'),
            }
            if DEBUG_MODE:
                resp['_debug'] = debug_info
            return jsonify(resp), 200
            # -------------------------------------------------

        # 2) Fallback to Model-based prediction
        cond_to_use = predicted_label or 'general_health'
        recs = diet_engine.get_recommendations(cond_to_use)

        # If model is confident (skipping CLARIFY) OR if intent is explicitly asking for a diet
        if (confidence >= CONFIDENCE_THRESHOLD and predicted_label and predicted_label != 'general_health') or intent == 'diet_planning':
            
            # --- FORCED ADVICE DELIVERY (Fallthrough Path - Skipping Confirmation) ---
            msg = humanize_recommendation(cond_to_use, recs) # No longer passing is_initial_advice
            
            SESSION_STORE[session_id] = {'condition': cond_to_use, 'last_query': query}
            
            resp = {
                'next_action': 'DELIVER_ADVICE', # Directly deliver the advice
                'condition': cond_to_use.replace('_',' ').title(),
                'recommendation': f"{msg}\n\n*{DISCLAIMER_TEXT}*", # INCLUDED DISCLAIMER HERE
                'avoid': recs.get('avoid'),
                'nutrients': recs.get('nutrients'),
                'advice': recs.get('advice'),
            }
            if DEBUG_MODE: resp['_debug'] = debug_info
            return jsonify(resp), 200
            # ----------------------------------------------------------------------


        # Low confidence AND not explicit/intent is low, ask clarifying question (CLARIFY)
        follow_up = ("Could you tell me a bit more about your symptoms or exact diagnosis? "
                     "For example: 'I have type 2 diabetes' or 'I have gastritis and get heartburn'.")
        
        # Inject Debug Info into the CLARIFY response
        if DEBUG_MODE:
            debug_str = json.dumps(debug_info, indent=2)
            follow_up = (
                "DEBUG: The system failed to explicitly match your input or predict a high-confidence condition. \n"
                f"Please check the data below (explicit_match: {debug_info['explicit_match']}) and rephrase.\n\n"
                "**DEBUG INFO**:\n"
                f"Predicted Label: {debug_info['predicted_label']} (Confidence: {debug_info['confidence']})\n\n"
                "Could you tell me a bit more about your symptoms or exact diagnosis? "
                "For example: 'I have type 2 diabetes' or 'I have gastritis and get heartburn'."
            )


        resp = {'next_action':'CLARIFY','condition':'Clarify','recommendation': follow_up}
        
        return jsonify(resp), 200

    except Exception as e:
        logger.exception("Unhandled exception in /chat: %s", e)
        if DEBUG_MODE:
            import traceback
            return jsonify({'next_action':'RESET','condition':'ServerError','recommendation':'Internal error','debug':traceback.format_exc()}), 500
        return jsonify({'next_action':'RESET','condition':'ServerError','recommendation':'Internal server error'}), 500


# -------------------------
# start
# -------------------------
if __name__ == '__main__':
    logger.info("Nutri-Bot starting. SentenceTransformers available: %s", SENTENCE_TRANSFORMERS_AVAILABLE)
    app.run(host='127.0.0.1', port=5000, debug=False)