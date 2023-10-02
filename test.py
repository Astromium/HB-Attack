import numpy as np
from utils.tensorflow_classifier import LcldTensorflowClassifier
from keras.models import load_model
from sklearn.pipeline import Pipeline
from keras.models import load_model
from ml_wrappers import wrap_model
import joblib

candidates = np.load('./candidates.npy')
scaler = joblib.load('./ressources/custom_botnet_scaler.joblib')
cf = load_model('./ressources/model_botnet.h5')
candidates = candidates.reshape(-1, 757)
wrapped = wrap_model(cf, candidates, model_task="classification")

model = Pipeline(steps=[('preprocessing', scaler), ('model', wrapped)])

print(f'candidates shape {candidates.shape}')

preds = model.predict(candidates)
print(f'preds {preds}')
args = np.where(preds == 0)[0]

print(f'len args {len(args)}')

adversarials = candidates[args]

print(f'len adversarials {adversarials.shape}')
np.save('./adversarials_botnet', adversarials)
