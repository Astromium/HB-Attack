from pymoo.util.nds import fast_non_dominated_sort
import numpy as np


def check_common(a, b):
    for el in a:
        if el in b:
            return True
    return False


def calculate_metrics(data, scores, candidates, scaler, eps, tolerance):
    C = 0
    M = 0
    CM = 0
    for i, x in enumerate(data):
        score = scores[i]
        preds = np.array([obj[0] for obj in score])
        violations = np.array([obj[1] for obj in score])
        candidates_scaled = scaler.transform(np.array(candidates[i]))
        x_scaled = scaler.transform(x.reshape(1, -1))
        distances = np.array([np.linalg.norm(adv - x_scaled)
                             for adv in candidates_scaled])

        distance_respected = distances <= eps
        constraints_respected = violations <= tolerance
        misclassification = preds < 0.5

        M += int(np.any(distance_respected * misclassification))
        C += int(np.any(constraints_respected * distance_respected))
        CM += int(np.any(distance_respected *
                  misclassification * constraints_respected))

    total = len(data)
    return {'C': round((C / total) * 100, 2), 'M': round((M / total) * 100, 2), 'C&M': round((CM / total) * 100, 2)}


def calculate_metrics_moehb(data, scores, eps, tolerance):
    C = 0
    M = 0
    CM = 0
    for i, x in enumerate(data):
        score = scores[i]
        # print(f'scores {i} : {score}')
        preds = np.array([obj[0] for obj in score])
        distances = np.array([obj[1] for obj in score])
        violations = np.array([obj[2] for obj in score])

        distance_respected = distances <= eps
        constraints_respected = violations <= tolerance
        misclassification = preds < 0.5

        M += int(np.any(distance_respected * misclassification))

        C += int(np.any(constraints_respected * distance_respected))
        CM += int(np.any(distance_respected *
                  misclassification * constraints_respected))

    total = len(data)
    return {'C': round((C / total) * 100, 2), 'M': round((M / total) * 100, 2), 'C&M': round((CM / total) * 100, 2)}
