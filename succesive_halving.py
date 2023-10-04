import math
import numpy as np
from tqdm import tqdm
from keras.models import load_model
from typing import Any, List, Union
from numpy.typing import NDArray
from sampler import Sampler
from evaluators import Evaluator
from utils.perturbation_generator import generate_perturbation
from utils.tensorflow_classifier import LcldTensorflowClassifier, TensorflowClassifier
from tensorflow.keras.models import load_model
from sklearn.pipeline import Pipeline
from ml_wrappers import wrap_model
from pymoo.util.nds import fast_non_dominated_sort


class SuccessiveHalving():
    def __init__(self, objective: Evaluator,
                 classifier_path: str,
                 sampler: Sampler,
                 x: NDArray,
                 y: NDArray,
                 eps: float,
                 dimensions: int,
                 max_configuration_size: int,
                 distance: str,
                 max_ressources_per_configuration: int,
                 downsample: int,
                 bracket_budget: int,
                 n_configurations: int,
                 mutables: Union[List, None],
                 features_min_max: Union[List, None],
                 int_features: Union[NDArray, None],
                 seed: int,
                 hyperband_bracket: int,
                 scaler: Any,
                 dataset: str,
                 R: int):
        self.objective = objective
        self.sampler = sampler
        self.x = x
        self.y = y
        self.eps = eps
        self.dimensions = dimensions
        self.max_configuration_size = max_configuration_size
        self.distance = distance
        self.max_ressources_per_configuration = max_ressources_per_configuration
        self.downsample = downsample
        self.bracket_budget = bracket_budget
        self.n_configurations = n_configurations
        self.mutables = mutables
        self.features_min_max = features_min_max
        self.int_features = int_features
        self.seed = seed
        self.hyperband_bracket = hyperband_bracket
        self.R = R
        self.dataset = dataset
        self.scaler = scaler
        self.classifier_path = classifier_path
        if dataset == 'url':
            model = TensorflowClassifier(load_model(self.classifier_path))
            # model = wrap_model(load_model(self.classifier_path),
            #                    self.x, model_task="classification")
        elif dataset == 'lcld':
            model = LcldTensorflowClassifier(load_model(self.classifier_path))
        elif dataset == 'botnet':
            model = wrap_model(
                load_model(self.classifier_path), self.x, model_task="classification")
        self.classifier = Pipeline(
            steps=[('preprocessing', self.scaler), ('model', model)])

    def process_one(self, candidate, idx, configuration, budget):
        new_score, new_candidate = self.objective.evaluate(
            classifier=self.classifier,
            configuration=configuration,
            budget=budget,
            x=self.x[idx],
            y=self.y[idx],
            eps=self.eps,
            distance=self.distance,
            features_min_max=self.features_min_max,
            int_features=self.int_features,
            generate_perturbation=generate_perturbation,
            candidate=candidate
        )

        return new_score, new_candidate
        # if new_score < score:
        # return tuple([new_score, new_candidate])
        # else:
        # return tuple([score, candidate])

    def run_one(self, idx):

        configurations = self.sampler.sample(
            dimensions=self.dimensions,
            num_configs=self.n_configurations,
            max_configuration_size=self.max_configuration_size,
            mutables_mask=self.mutables,
            seed=self.seed
        )

        for i in range(self.hyperband_bracket + 1):
            budget = self.bracket_budget * pow(self.downsample, i)
            results = [self.process_one(score=score, candidate=candidate, idx=idx, configuration=configuration, budget=budget,
                                        ) for score, candidate, configuration in zip(scores, candidates, configurations)]
            scores = [r[0] for r in results]
            candidates = [r[1] for r in results]
            top_indices = np.argsort(scores)[:max(
                int(len(scores) / self.downsample), 1)]
            configurations = [configurations[j] for j in top_indices]
            candidates = [candidates[j] for j in top_indices]

        return (scores, configurations, candidates)

    def run(self):
        if (self.downsample <= 1):
            raise (ValueError('Downsample must be > 1'))

        all_results = []

        for idx in range(self.x.shape[0]):

            configurations = self.sampler.sample(
                dimensions=self.dimensions,
                num_configs=self.n_configurations,
                max_configuration_size=self.max_configuration_size,
                mutables_mask=self.mutables,
                seed=self.seed
            )

            # scores = [math.inf for s in range(len(configurations))]
            candidates = [None for c in range(len(configurations))]

            for i in range(self.hyperband_bracket + 1):
                budget = self.bracket_budget * \
                    pow(self.downsample, i) if self.bracket_budget * \
                    pow(self.downsample, i) < self.R else self.R
                results = [self.process_one(candidate=None, idx=idx, configuration=configuration, budget=budget)
                           for configuration in tqdm(configurations, total=len(configurations), desc=f'SH round {i}, Evaluating {len(configurations)} with budget of {budget}')]

                scores = [r[0] for r in results]
                candidates = [r[1] for r in results]
                fronts = fast_non_dominated_sort.fast_non_dominated_sort(
                    np.array(scores))

                flattened = []
                for front in fronts:
                    for v in front:
                        flattened.append(v)
                top_indices = flattened[:max(
                    int(len(scores) / self.downsample), 1)]
                # print(f'top indices {top_indices}')
                configurations = [configurations[j] for j in top_indices]
                candidates = [candidates[j] for j in top_indices]
                scores = [scores[j] for j in top_indices]

            all_results.append(
                (scores, configurations, candidates))
        return all_results
