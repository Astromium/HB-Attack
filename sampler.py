import random
from dataclasses import dataclass
from typing import List, Union


@dataclass
class Sampler():
    consistancy : bool
    constraints_list: List

    def sample(self, dimensions: int, num_configs: int, max_configuration_size: int, mutables_mask: Union[List[int], None], seed: int) -> List[int]:
        configurations = [None] * num_configs
        if mutables_mask:
            sample_list = mutables_mask
        else:
            sample_list = list(range(0, dimensions))

        for i in range(num_configs):
            n = random.randint(1, max_configuration_size + 1)
            config = random.sample(sample_list, n)

            if self.consistancy:
                # we have to make sure that a configuration is consistant
                # i.e (I'll explain this later)
                features_to_add = []
                for feature in config:
                    constraints = [constraint for constraint in self.constraints_list if feature in constraint]
                    for c in constraints:
                        features_to_add.extend(c)
                    features_to_add = list(set(features_to_add))

                config.extend(features_to_add)


            configurations[i] = list(set(config))
        
        return configurations
