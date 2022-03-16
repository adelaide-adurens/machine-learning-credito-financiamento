import numpy as np
import pandas as pd

def show_all(df):
    
    # setando opções sem limites
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('max_colwidth', None)
    
    display(df)
    
    # resetando opções pro padrão    
    pd.reset_option('display.max_columns')
    pd.reset_option('display.max_rows')
    pd.reset_option('max_colwidth')
    
def age_range(age):
    
    if age <= 30:
        
        return "20-30"
    
    elif age <= 45:
        
        return "31-45"
    
    elif age <= 60:
        
        return "45-60"
    
    else:
        
        return "60+"