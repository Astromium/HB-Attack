import numpy as np
from utils.tensorflow_classifier import TensorflowClassifier
from keras.models import load_model
from sklearn.pipeline import Pipeline
import joblib

scaler_path = "./ressources/baseline_scaler.joblib"
classifier_path = "./ressources/baseline_nn.model"

scaler = joblib.load(scaler_path)
clf = TensorflowClassifier(load_model(classifier_path))

model = Pipeline(steps=[('preprocessing', scaler), ('model', clf)])

candidates = np.load('./candidates.npy')
preds = model.predict(candidates.reshape(-1, 63))

args = np.where(preds == 0)[0]
print(f'len args  {len(args)}')
adversarials = candidates.reshape(-1, 63)[args]

print(f'adversarials shape {adversarials.shape}')

np.save('./adversarials.npy', adversarials)