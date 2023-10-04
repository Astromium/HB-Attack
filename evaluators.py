from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Union, Callable, List
from numpy.typing import NDArray
from constraints.relation_constraint import BaseRelationConstraint
from constraints.constraints_executor import NumpyConstraintsExecutor
from constraints.relation_constraint import AndConstraint
from sklearn.preprocessing import MinMaxScaler
from pymoo.util.nds import fast_non_dominated_sort
from utils.fix_types import fix_feature_types
from utils.inverse_transform import inverse_transform
import numpy as np


@dataclass
class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, classifier: Any, configuration: tuple, budget: int, x: NDArray, y: int, eps: float, distance: str, features_min_max: Union[tuple, None]):
        pass


class URLEvaluator(Evaluator):
    def __init__(self, constraints: Union[List[BaseRelationConstraint], None], scaler: Union[MinMaxScaler, None]):
        super().__init__()
        self.constraints = constraints
        self.constraint_executor = NumpyConstraintsExecutor(
            AndConstraint(constraints)) if constraints is not None else None
        self.scaler = scaler

    def evaluate(self, classifier: Any, configuration: tuple, budget: int, x: NDArray, y: int, eps: float, distance: str, features_min_max: Union[tuple, None], int_features: Union[NDArray, None], generate_perturbation: Callable, candidate: NDArray):
        scores = [0] * budget
        adversarials = [None] * budget
        if candidate is None:
            adv = np.copy(x)
        else:
            adv = np.copy(candidate)
        for i in range(budget):
            # perturbation = generate_perturbation(shape=np.array(configuration).shape, eps=eps, distance=distance)
            perturbation = generate_perturbation(
                configuration=configuration, features_min=features_min_max[0], features_max=features_min_max[1], x=x)
            adv[list(configuration)] += perturbation

            adv_scaled = self.scaler.transform(adv[np.newaxis, :])[0]
            x_scaled = self.scaler.transform(x[np.newaxis, :])[0]
            dist = np.linalg.norm(adv_scaled - x_scaled, ord=distance)
            if dist > eps:
                adv_scaled = x_scaled + (adv_scaled - x_scaled) * eps / dist
                # transform back to pb space
                adv = self.scaler.inverse_transform(
                    adv_scaled[np.newaxis, :])[0]

            # clipping
            adv = np.clip(adv, features_min_max[0], features_min_max[1])
            # casting
            adv = fix_feature_types(
                perturbation=perturbation, adv=adv, int_features=int_features, configuration=configuration)

            # pred = classifier.predict_proba(adv[np.newaxis, :])[0][y]
            # violations = self.constraint_executor.execute(adv[np.newaxis, :])[0]
            # scores[i] = [pred, violations]
            adversarials[i] = np.copy(adv)

        preds = classifier.predict_proba(np.array(adversarials))[:, y]
        violations = self.constraint_executor.execute(np.array(adversarials))

        scores = [[p, v] for p, v in zip(preds, violations)]

        fronts = fast_non_dominated_sort.fast_non_dominated_sort(
            np.array(scores))

        return scores[fronts[0][0]], adversarials[fronts[0][0]]


class BotnetEvaluator(Evaluator):
    def __init__(self, constraints: Union[List[BaseRelationConstraint], None], scaler: Union[MinMaxScaler, None], feature_names: List[str]):
        super().__init__()
        self.constraints = constraints
        self.constraint_executor = NumpyConstraintsExecutor(
            AndConstraint(constraints), feature_names=feature_names) if constraints is not None else None
        self.scaler = scaler

    def process_one(self, p, x, configuration, distance, eps, features_min_max, int_features):
        adv = np.copy(x)
        adv[list(configuration)] += p
        adv = fix_feature_types(
            perturbation=p, adv=adv, int_features=int_features, configuration=configuration)
        adv_scaled = self.scaler.transform(adv[np.newaxis, :])[0]
        x_scaled = self.scaler.transform(x[np.newaxis, :])[0]
        dist = np.linalg.norm(adv_scaled - x_scaled, ord=distance)
        # print(f'dist before projection {dist}')
        if dist > eps:
            adv_scaled = x_scaled + (adv_scaled - x_scaled) * eps / dist
            # transform back to pb space
            adv = self.scaler.inverse_transform(
                adv_scaled[np.newaxis, :])[0]

        adv = np.clip(adv, features_min_max[0], features_min_max[1])
        adv = fix_feature_types(
            perturbation=p, adv=adv, int_features=int_features, configuration=configuration)

        return adv

    def evaluate(self, classifier: Any, configuration: tuple, budget: int, x: NDArray, y: int, eps: float, distance: str, features_min_max: Union[tuple, None], int_features: Union[NDArray, None], generate_perturbation: Callable, candidate: NDArray):

        perturbations = [generate_perturbation(
            configuration=configuration, features_min=features_min_max[0], features_max=features_min_max[1], x=x) for _ in range(budget)]
        scores = [0] * budget
        adversarials = np.array([self.process_one(
            p, x, configuration, distance, eps, features_min_max, int_features) for p in perturbations])
        preds = classifier.predict_proba(adversarials)[:, y]
        violations = self.constraint_executor.execute(adversarials)
        scores = [[p, v] for p, v in zip(preds, violations)]

        fronts = fast_non_dominated_sort.fast_non_dominated_sort(
            np.array(scores))
        return scores[fronts[0][0]], adversarials[fronts[0][0]]


class LCLDEvaluator(Evaluator):
    def __init__(self, constraints: Union[List[BaseRelationConstraint], None], scaler: Union[MinMaxScaler, None], feature_names: List[str]):
        super().__init__()
        self.constraints = constraints
        self.constraint_executor = NumpyConstraintsExecutor(
            AndConstraint(constraints), feature_names=feature_names) if constraints is not None else None
        self.scaler = scaler

    def process_one(self, p, x, configuration, distance, eps, features_min_max, int_features):
        adv = np.copy(x)
        adv[list(configuration)] += p

        adv = fix_feature_types(
            perturbation=p, adv=adv, int_features=int_features, configuration=configuration)

        adv_scaled = self.scaler.transform(adv[np.newaxis, :])[0]
        x_scaled = self.scaler.transform(x[np.newaxis, :])[0]
        dist = np.linalg.norm(adv_scaled - x_scaled, ord=distance)
        start = self.scaler.transformers_[1][2][0]
        # print(f'dist before projection {dist}')
        if dist > eps:
            adv_scaled = x_scaled + (adv_scaled - x_scaled) * eps / dist
            adv_scaled[start:] = list(map(int, adv_scaled[start:]))
            # transform back to pb space
            adv = inverse_transform(preprocessor=self.scaler, x=adv_scaled)

        adv = np.clip(adv, features_min_max[0], features_min_max[1])

        adv = fix_feature_types(
            perturbation=p, adv=adv, int_features=int_features, configuration=configuration)

        return adv

    def evaluate(self, classifier: Any, configuration: tuple, budget: int, x: NDArray, y: int, eps: float, distance: str, features_min_max: Union[tuple, None], int_features: Union[NDArray, None], generate_perturbation: Callable, candidate: NDArray):
        perturbations = [generate_perturbation(
            configuration=configuration, features_min=features_min_max[0], features_max=features_min_max[1], x=x) for _ in range(budget)]
        scores = [0] * budget
        adversarials = np.array([self.process_one(
            p, x, configuration, distance, eps, features_min_max, int_features) for p in perturbations])
        preds = classifier.predict_proba(adversarials)[:, y]

        if self.constraints:
            violations = self.constraint_executor.execute(adversarials)
            scores = [[p, v] for p, v in zip(preds, violations)]
        else:
            scores = preds

        fronts = fast_non_dominated_sort.fast_non_dominated_sort(
            np.array(scores))
        return scores[fronts[0][0]], adversarials[fronts[0][0]]
