# models_def.py
class NutriCareNLP:
    def __init__(self):
        self.symptom_classifier = None
        self.intent_classifier = None

class ICMRDietEngine:
    def __init__(self, food_df, health_df, icmr_rda_df):
        self.food_df = food_df
        self.health_df = health_df
        self.icmr_rda_df = icmr_rda_df