import numpy as np
from utils.tensorflow_classifier import TensorflowClassifier
from sklearn.pipeline import Pipeline
from keras.models import load_model
import joblib

candidates = np.load('./candidates.npy')
scaler = joblib.load('./ressources/baseline_scaler.joblib')
cf = TensorflowClassifier(load_model('./ressources/baseline_nn.model'))

model = Pipeline(steps=[('preprocessing', scaler), ('model', cf)])

candidates = candidates.reshape(-1, 63)
print(f'candidates shape {candidates.shape}')
preds = model.predict(candidates)
args = np.where(preds == 0)[0]

print(f'len args {len(args)}')

adversarials = candidates[args]

print(f'len adversarials {adversarials.shape}')
np.save('./adversarials', adversarials)
