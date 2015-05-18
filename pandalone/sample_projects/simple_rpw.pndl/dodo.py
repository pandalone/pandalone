# Pandalone tasks.
import pandas as pd


def task_read_input():
    def read_input_csv(dependencies):
        df = pd.read_csv(next(iter(dependencies)))
        return df
    
    return {
        'actions': [read_input_csv],
        'file_dep': ['input.csv'],
    }
