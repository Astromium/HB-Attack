import argparse
from data_generator import DataGenerator
from hyperband import Hyperband
from moehb import MOEHB
from evaluators import BotnetEvaluator, LCLDEvaluator, URLEvaluator
from utils.tensorflow_classifier import LcldTensorflowClassifier
from sklearn.pipeline import Pipeline
from sampler import Sampler
from objective_calculator import calculate_metrics, calculate_metrics_moehb
import tensorflow as tf
import numpy as np
import joblib
import os
import warnings
import pickle
# tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings(action='ignore')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # arguments to execute the script
    parser.add_argument('-R', default=81)
    parser.add_argument('-eps', required=True)
    parser.add_argument('-eta', default=3)
    parser.add_argument('-dataset', required=True)
    parser.add_argument('-attack', required=True)
    parser.add_argument('-batch', default=10)
    parser.add_argument('-n_gen', default=300)
    parser.add_argument('-pop_size', default=150)
    parser.add_argument('-configs_path', default=None)
    parser.add_argument('-scores_path', default=None)

    args = parser.parse_args()

    dataset = args.dataset
    atk = args.attack
    R = int(args.R)
    eps = float(args.eps)
    eta = int(args.eta)
    n_gen = int(args.n_gen)
    pop_size = int(args.pop_size)
    configs_path = args.configs_path
    scores_path = args.scores_path

    data_gen = DataGenerator()
    x, y, mutables, features_min_max, int_features, feature_names, constraints = data_gen.get_dataset(
        dataset=dataset)

    scaler_path = None
    scaler = None
    evaluator = None
    classifier_path = None
    tolerance = None

    if dataset == 'url':
        scaler_path = "./ressources/baseline_scaler.joblib"
        classifier_path = "./ressources/baseline_nn.model"
        scaler = joblib.load(scaler_path)
        evaluator = URLEvaluator(constraints=constraints, scaler=scaler)
        tolerance = 0.0001
        x, y = x[:50], y[:50]
    elif dataset == "botnet":
        scaler_path = "./ressources/custom_botnet_scaler.joblib"
        classifier_path = "./ressources/model_botnet.h5"
        scaler = joblib.load(scaler_path)
        evaluator = BotnetEvaluator(
            constraints=constraints, scaler=scaler, feature_names=feature_names)
        tolerance = 0.001
    elif dataset == "lcld":
        scaler_path = "./ressources/lcld_preprocessor.joblib"
        classifier_path = "./ressources/custom_lcld_model.h5"
        scaler = joblib.load(scaler_path)
        evaluator = LCLDEvaluator(
            constraints=constraints, scaler=scaler, feature_names=feature_names)
        tolerance = 0.01

        model = LcldTensorflowClassifier(
            tf.keras.models.load_model(classifier_path))
        model_pipeline = Pipeline(
            steps=[('preprocessing', scaler), ('model', model)])
        preds = model_pipeline.predict_proba(x)
        classes = np.argmax(preds, axis=1)
        to_keep = np.where(classes == 1)[0]
        x, y = x[to_keep], y[to_keep]

    # Parameters

    dimensions = len(mutables)
    BATCH_SIZE = x.shape[0] if int(args.batch) == -1 else int(args.batch)
    seed = 202374
    sampler = Sampler()

    if atk == 'hyperband':
        hb = Hyperband(objective=evaluator, classifier_path=classifier_path, x=x[:BATCH_SIZE], y=y[:BATCH_SIZE], sampler=sampler,
                       eps=eps, dimensions=dimensions, max_configuration_size=dimensions-1, R=R, downsample=eta, distance=2, seed=seed)
        scores, configurations, candidates = hb.generate(
            scaler=scaler, dataset=dataset, mutables=mutables, features_min_max=features_min_max, int_features=int_features)

        with open('./configs', 'wb') as f:
            pickle.dump(configurations, f)
        with open('./scores', 'wb') as f:
            pickle.dump(scores, f)

        # for i in range(len(scores)):
        #     print(f'scores for example {i} : {scores[i]}')

        metrics = calculate_metrics(
            data=x[:BATCH_SIZE], scores=scores, candidates=candidates, scaler=scaler, eps=eps, tolerance=tolerance)

        print(f'metrics {metrics}')

    elif atk == 'moehb':
        hb_init = {'objective': evaluator, 'classifier_path': classifier_path, 'x': x[:BATCH_SIZE], 'y': y[:BATCH_SIZE], 'sampler': sampler,
                   'eps': eps, 'dimensions': dimensions, 'max_configuration_size': dimensions-1, 'R': R, 'downsample': eta, 'distance': 2, 'seed': seed}
        hb_gen = {'scaler': scaler, 'dataset': dataset, 'mutables': mutables,
                  'features_min_max': features_min_max, 'int_features': int_features}

        history = None

        if configs_path and scores_path:
            with open(scores_path, 'rb') as f:
                scores = pickle.load(f)
            with open(configs_path, 'rb') as f:
                configs = pickle.load(f)
            history = (configs, np.array(scores))
            with open('./configs', 'rb') as f:
                configs = pickle.load(f)
            with open('./scores', 'rb') as f:
                scores = pickle.load(f)
            history = (configs, scores)

        moehb = MOEHB(hb_init=hb_init, hb_gen=hb_gen, n_gen=n_gen, pop_size=pop_size,
                      constraints=constraints, tolerance=tolerance, feature_names=feature_names, history=history)
        solutions, scores = moehb.run()

        metrics = calculate_metrics_moehb(
            data=x[:BATCH_SIZE], scores=scores, eps=eps, tolerance=tolerance)

        print(f'metrics {metrics}')
        with open('./metrics.pkl', 'wb') as f:
            pickle.dump(metrics, f)
