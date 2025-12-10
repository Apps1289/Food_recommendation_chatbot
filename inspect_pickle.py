# inspect_pickles.py
import pickle, os
def inspect(path):
    print("===", path, "===")
    try:
        with open(path,'rb') as f:
            obj = pickle.load(f)
        print("Type:", type(obj))
        if isinstance(obj, dict):
            print("Keys:", list(obj.keys()))
        else:
            attrs = [a for a in dir(obj) if not a.startswith("__")]
            print("Some attributes:", attrs[:80])
        try:
            print("Size (KB):", os.path.getsize(path)/1024)
        except: pass
    except Exception as e:
        print("Error:", e)

inspect('nlp_model.pkl')
inspect('diet_engine.pkl')
