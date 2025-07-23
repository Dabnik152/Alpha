import pandas as pd


def load_jailbreak_data(model_name='gemma2'):
    
    splits = {'gemma2': 'gemma-2-2b-it_eval_data.parquet', 'qwen2.5': 'qwen2.5-1.5b-instruct_eval_data.parquet', 'llama3.1': 'llama-3.1-8b-instruct_eval_data.parquet'}
    try:
        df = pd.read_parquet("/home/morg/students/yahavt/InSPEcT/data/" + splits[model_name])
        return df
    
    except Exception as e:
        print(f"Error loading jailbreak data for {model_name}: {e}")
        return None
    
# data = load_jailbreak_data("gemma2")

