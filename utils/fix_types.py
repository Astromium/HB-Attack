import math

def fix_feature_types(perturbation, adv, int_features, configuration):
        for i, c in enumerate(configuration):
            if c in int_features:
                adv[c] = math.ceil(
                    adv[c]) if perturbation[i] < 0 else math.floor(adv[c])
        return adv