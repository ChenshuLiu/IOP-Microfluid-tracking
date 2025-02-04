import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
from scipy.stats import pearsonr

exp_data = pd.read_excel('./distance_pressure.xlsx')
corr, _ = pearsonr(exp_data['Displacement'], exp_data['IOP']) # correlation coefficient: 0.9894
linear_model = LinearRegression()
linear_model.fit(np.array(exp_data['Displacement']).reshape(-1, 1), 
                 np.array(exp_data['IOP']))
with open("distance_iop_linear_model.pkl", 'wb') as file:
    pickle.dump(linear_model, file)