import argparse
from data_generator import DataGenerator
from hyperband import Hyperband
from moehb import MOEHB
from evaluators import BotnetEvaluator, LCLDEvaluator, URLEvaluator
from sampler import Sampler
from objective_calculator import calculate_metrics, calculate_metrics_moehb
import tensorflow as tf
import joblib
import os 
import warnings
#tf.compat.v1.disable_eager_execution()
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
    parser.add_argument('-n_gen', default=100)
    parser.add_argument('-pop_size', default=100)

    args = parser.parse_args()

    dataset = args.dataset
    atk = args.attack
    R = int(args.R)
    eps = float(args.eps)
    eta = int(args.eta)
    n_gen = int(args.n_gen)
    pop_size = int(args.pop_size)

    data_gen = DataGenerator()
    x, y, mutables, features_min_max, int_features, feature_names, constraints = data_gen.get_dataset(dataset=dataset)

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
    elif dataset == "botnet":
        scaler_path = "./ressources/custom_botnet_scaler.joblib"
        classifier_path = "./ressources/model_botnet.h5"
        scaler = joblib.load(scaler_path)
        evaluator = BotnetEvaluator(constraints=constraints, scaler=scaler, feature_names=feature_names)
        tolerance = 0.001
    elif dataset == "lcld":
        scaler_path = "./ressources/lcld_preprocessor.joblib"
        classifier_path = "./ressources/custom_lcld_model.h5"
        scaler = joblib.load(scaler_path)
        evaluator = LCLDEvaluator(constraints=constraints, scaler=scaler, feature_names=feature_names)
        tolerance = 0.01

    # Parameters 

    dimensions = len(mutables)
    BATCH_SIZE = x.shape[0] if int(args.batch) == -1 else int(args.batch)
    seed = 202374
    sampler = Sampler()

    if atk == 'hyperband':
        hb = Hyperband(objective=evaluator, classifier_path=classifier_path, x=x[:BATCH_SIZE], y=y[:BATCH_SIZE], sampler=sampler,
                        eps=eps, dimensions=dimensions, max_configuration_size=dimensions-1, R=R, downsample=eta, distance=2, seed=seed)
        scores, configurations, candidates = hb.generate(scaler=scaler, dataset=dataset, mutables=mutables, features_min_max=features_min_max, int_features=int_features)

        # for i in range(len(scores)):
        #     print(f'scores for example {i} : {scores[i]}')

        metrics = calculate_metrics(data=x[:BATCH_SIZE], scores=scores, candidates=candidates, scaler=scaler, eps=eps, tolerance=tolerance)

        print(f'metrics {metrics}')
    
    elif atk == 'moehb':
        hb_init = {'objective': evaluator, 'classifier_path':classifier_path, 'x':x[:BATCH_SIZE], 'y':y[:BATCH_SIZE], 'sampler':sampler,
                        'eps':eps, 'dimensions':dimensions, 'max_configuration_size':dimensions-1, 'R':R, 'downsample':eta, 'distance':2, 'seed':seed}
        hb_gen = {'scaler':scaler, 'dataset':dataset, 'mutables':mutables, 'features_min_max':features_min_max, 'int_features':int_features}

        moehb = MOEHB(hb_init=hb_init, hb_gen=hb_gen, n_gen=n_gen, pop_size=pop_size, constraints=constraints, tolerance=tolerance, feature_names=feature_names, history=None)
        solutions , scores = moehb.run()

        metrics = calculate_metrics_moehb(data=x[:BATCH_SIZE], scores=scores, eps=eps, tolerance=tolerance)

        print(f'metrics {metrics}')


