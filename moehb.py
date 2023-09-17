from hyperband import Hyperband
from adversarial_problem import AdversarialProblem
from utils.tensorflow_classifier import TensorflowClassifier, LcldTensorflowClassifier
from constraints.constraints_executor import NumpyConstraintsExecutor
from constraints.relation_constraint import AndConstraint
from keras.models import load_model
from ml_wrappers import wrap_model
from sklearn.pipeline import Pipeline
from pymoo.util.nds import fast_non_dominated_sort
from pymoo.algorithms.moo.rnsga3 import RNSGA3
from pymoo.factory import get_crossover, get_mutation, get_reference_directions
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from tqdm import tqdm
import numpy as np
import pickle


class MOEHB():

    def __init__(self, hb_init, hb_gen, n_gen, pop_size, constraints, tolerance, feature_names, history=None) -> None:
        self.hb_init = hb_init
        self.hb_gen = hb_gen
        self.n_gen = n_gen
        self.pop_size = pop_size
        self.constraints = constraints
        self.tolerance = tolerance
        self.feature_names = feature_names
        self.history = history

    def run(self):
        # Extract parameters
        dataset = self.hb_gen['dataset']
        scaler = self.hb_gen['scaler']
        features_min_max = self.hb_gen['features_min_max']
        int_features = self.hb_gen['int_features']
        x = self.hb_init['x']
        y = self.hb_init['y']
        classifier_path = self.hb_init['classifier_path']
        eps = self.hb_init['eps']
        if dataset == 'url':
            model = TensorflowClassifier(load_model(classifier_path))
        elif dataset == 'lcld':
            model = LcldTensorflowClassifier(load_model(classifier_path))
        elif dataset == 'botnet':
            model = wrap_model(
                load_model(classifier_path), x, model_task="classification")
        classifier = Pipeline(
            steps=[('preprocessing', scaler), ('model', model)])
        constraints_executor = NumpyConstraintsExecutor(
            AndConstraint(self.constraints), feature_names=self.feature_names)
        if self.history is None:
            # No results from prior Hyperband execution
            hb = Hyperband(**self.hb_init)
            scores, configs, candidates = hb.generate(**self.hb_gen)

            # save for future tests
            with open('./configs', 'wb') as f:
                pickle.dump(configs, f)
            with open('./scores', 'wb') as f:
                pickle.dump(scores, f)

            all_solutions = []
            all_objectives = []
            for i, (score, config, candidate) in enumerate(tqdm(zip(scores, configs, candidates), total=len(scores))):
                print(f'Starting Perturbation Search For Example {i}')
                fronts = fast_non_dominated_sort.fast_non_dominated_sort(
                    np.array(score))
                best_config = config[fronts[0][0]]

                problem = AdversarialProblem(
                    x_clean=x[i],
                    n_var=len(best_config),
                    y_clean=y[i],
                    classifier=classifier,
                    constraints_executor=constraints_executor,
                    features_min_max=features_min_max,
                    scaler=scaler,
                    configuration=best_config,
                    int_features=int_features,
                    eps=eps,
                    tolerance=self.tolerance
                )

                ref_points = get_reference_directions(
                    "energy", problem.n_obj, self.pop_size, seed=1
                )
                # ref_points = get_reference_directions('uniform', self.R, problem.n_obj)
                # get_sampling('real_random')
                algorithm = RNSGA3(  # population size
                    n_offsprings=100,  # number of offsprings
                    sampling=FloatRandomSampling(),  # use the provided initial population
                    crossover=get_crossover("real_sbx", prob=0.9, eta=15),
                    mutation=get_mutation("real_pm", eta=20),
                    eliminate_duplicates=True,
                    ref_points=ref_points,
                    pop_per_ref_point=1,
                )

                res = minimize(problem, algorithm,
                               termination=('n_gen', self.n_gen))

                solutions = res.pop.get("X")
                objectives = res.pop.get("F")

                all_solutions.append((solutions, best_config, x[i]))
                all_objectives.append(objectives)

            return all_solutions, all_objectives

        else:
            all_solutions = []
            all_objectives = []
            configs, scores = self.history
            for i in tqdm(range(len(x)), total=len(x)):
                config = configs[i]
                score = scores[i]
                fronts = fast_non_dominated_sort.fast_non_dominated_sort(
                    score)
                best_config = config[fronts[0][0]]

                print(f'Starting Perturbation Search for Example {i}')

                problem = AdversarialProblem(
                    x_clean=x[i],
                    n_var=len(best_config),
                    y_clean=y[i],
                    classifier=classifier,
                    constraints_executor=constraints_executor,
                    features_min_max=features_min_max,
                    scaler=scaler,
                    configuration=best_config,
                    int_features=int_features,
                    eps=eps,
                    tolerance=self.tolerance
                )

                ref_points = get_reference_directions(
                    "energy", problem.n_obj, self.pop_size, seed=1
                )
                # ref_points = get_reference_directions('uniform', self.R, problem.n_obj)
                # get_sampling('real_random')
                algorithm = RNSGA3(  # population size
                    n_offsprings=100,  # number of offsprings
                    sampling=FloatRandomSampling(),  # use the provided initial population
                    crossover=get_crossover("real_sbx", prob=0.9, eta=15),
                    mutation=get_mutation("real_pm", eta=20),
                    eliminate_duplicates=True,
                    ref_points=ref_points,
                    pop_per_ref_point=1,
                )

                res = minimize(problem, algorithm,
                               termination=('n_gen', self.n_gen))

                solutions = res.pop.get("X")
                objectives = res.pop.get("F")

                all_solutions.append((solutions, best_config, x[i]))
                all_objectives.append(objectives)

            return all_solutions, all_objectives