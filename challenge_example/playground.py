import pandas as pd
import os 
from pathlib import Path

if __name__ == '__main__':
    df = pd.read_csv('C:\\Users\\Kaouther\\Documents\\GitHub\\NeurologicalRecovery\\data\\DatasetInformationParsed.csv')
    val_path = Path('C:\\Users\\Kaouther\\Documents\\GitHub\\NeurologicalRecovery\\data\\val')
    for val_case in val_path.iterdir():
        print(val_case)