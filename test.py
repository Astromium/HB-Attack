import numpy as np
import pandas as pd
import pickle

with open('./configs', 'rb') as f:
    configs = pickle.load(f)

print(f'configs {configs}')
