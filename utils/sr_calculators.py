import numpy as np
from scipy.special import softmax
from typing import List, Any
from numpy.typing import NDArray
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pymoo.util.nds import fast_non_dominated_sort


@dataclass
class SuccessRateCalculator(ABC):
    classifier: Any
    data: Any
    labels: Any
    scores: Any
    candidates: List
    scaler: Any
    eps: float

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError()


class TfCalculator(SuccessRateCalculator):
    def __init__(self, classifier, data, labels, scores, candidates):
        super().__init__(classifier=classifier, data=data,
                         labels=labels, scores=scores, candidates=candidates)

    def evaluate(self):
        correct = 0
        success_rate = 0
        adversarials = []
        best_candidates = []
        for i, (x, y) in enumerate(zip(self.data, self.labels)):
            pred = np.argmax(
                softmax(self.classifier.predict(x[np.newaxis, :])))
            if pred != y:
                # print('inside the if')
                continue

            correct += 1
            best_score_idx = np.argmin(self.scores[i])
            best_candidate = self.candidates[i][best_score_idx]
            pred = np.argmax(
                softmax(self.classifier.predict(best_candidate[np.newaxis, :])))
            best_candidates.append(best_candidate)

            if pred != y:
                # print(f'adversarial {i}')
                adversarials.append(best_candidate)
                success_rate += 1
        eps = 0.0001 if correct == 0 else 0

        return round(success_rate / correct + eps, 3), best_candidates, adversarials


class TorchCalculator(SuccessRateCalculator):
    def __init__(self, classifier, data, labels, scores, candidates, scaler, eps):
        super().__init__(classifier=classifier, data=data, labels=labels,
                         scores=scores, candidates=candidates, scaler=scaler, eps=eps)

    def evaluate(self):
        correct = 0
        success_rate = 0
        adversarials = []
        best_candidates = []
        for i, (x, y) in enumerate(zip(self.data, self.labels)):
            pred = self.classifier.predict(x[np.newaxis, :])[0]
            if pred != y:
                continue

            correct += 1
            fronts = fast_non_dominated_sort.fast_non_dominated_sort(
                self.scores[i])
            best_score_idx = fronts[0][0]
            # best_score_idx = np.argmin(self.scores[i])
            best_candidate = self.candidates[i][best_score_idx]
            bc_scaled = self.scaler.transform(best_candidate[np.newaxis, :])[0]
            x_scaled = self.scaler.transform(x[np.newaxis, :])[0]
            dist = np.linalg.norm(bc_scaled - x_scaled)
            pred = self.classifier.predict(best_candidate[np.newaxis, :])[0]
            best_candidates.append(best_candidate)
            print(
                f'pred {self.classifier.predict_proba(best_candidate[np.newaxis, :])[0]}; y {y}')
            if pred != y and dist <= self.eps:
                # print(f'adversarial {i}')
                adversarials.append(best_candidate)
                success_rate += 1
        eps = 0.0001 if correct == 0 else 0
        print(f'Correct {correct}')
        return round(success_rate / correct + eps, 3), best_candidates, adversarials


class SickitCalculator(SuccessRateCalculator):
    def __init__(self, classifier, data, labels, scores, candidates):
        super().__init__(classifier=classifier, data=data,
                         labels=labels, scores=scores, candidates=candidates)

    def evaluate(self):
        correct = 0
        success_rate = 0
        adversarials = []
        best_candidates = []
        for i, (x, y) in enumerate(zip(self.data, self.labels)):
            pred = self.classifier.predict(x[np.newaxis, :])[0]
            if pred != y:
                # print('inside the if')
                continue

            correct += 1
            best_score_idx = np.argmin(self.scores[i])
            best_candidate = self.candidates[i][best_score_idx]
            pred = self.classifier.predict(best_candidate[np.newaxis, :])[0]
            best_candidates.append(best_candidate)

            if pred != y:
                # print(f'adversarial {i}')
                adversarials.append(best_candidate)
                success_rate += 1
        eps = 0.0001 if correct == 0 else 0

        return round(success_rate / correct + eps, 3), best_candidates, adversarials