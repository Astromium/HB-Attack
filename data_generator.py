
from constraints.url_constraints import get_url_relation_constraints
from constraints.botnet_constraints import get_relation_constraints
from constraints.lcld_constraints import get_relation_constraints as get_lcld_relation_constraints
from mlc.datasets.dataset_factory import get_dataset
import numpy as np
import pandas as pd

class DataGenerator():
    def __init__(self) -> None:
        pass

    def get_dataset(self, dataset):
        if dataset == 'url':
            x_clean = np.load('./ressources/baseline_X_test_candidates.npy')
            y_clean = np.load('./ressources/baseline_y_test_candidates.npy')
            # x_clean = scaler.transform(x_clean)

            metadata = pd.read_csv('./ressources/url_metadata.csv')
            min_constraints = metadata['min'].to_list()[:63]
            max_constraints = metadata['max'].to_list()[:63]
            feature_types = metadata['type'].to_list()[:63]
            feature_names = metadata['feature'].to_list()[:63]
            int_features = np.where(np.array(feature_types) == 'int')[0]
            features_min_max = (min_constraints, max_constraints)
            mutables = metadata.index[metadata['mutable'] == True].tolist()
            constraints = get_url_relation_constraints()

            return x_clean, y_clean, mutables, features_min_max, int_features, feature_names, constraints 

        elif dataset == 'botnet':
            ds = get_dataset('ctu_13_neris')
            X, y = ds.get_x_y()
            feature_names = X.columns.to_list()
            # print(f'columns {X.columns.to_list()}')
            constraints = get_relation_constraints(X)
            metadata = ds.get_metadata()
            
            X = X.to_numpy()
            X_test, y_test = X[143046:], y[143046:]

            # filter non botnet examples
            botnet = np.where(y_test == 1)[0]
            X_test_botnet, y_test_botnet = X_test[botnet], y_test[botnet]
            mutables = metadata.index[metadata['mutable'] == True].tolist()
            min_constraints = metadata['min'].to_list()[:-1]
            max_constraints = metadata['max'].to_list()[:-1]
            features_min_max = (min_constraints, max_constraints)
            int_features = metadata.index[metadata['type'] == 'int'].to_list()

            return X_test_botnet, y_test_botnet, mutables, features_min_max, int_features, feature_names, constraints
        
        elif dataset == 'lcld':
            ds = get_dataset('lcld_v2_iid')
            splits = ds.get_splits()
            x, y = ds.get_x_y()

            categorical = ['home_ownership', 'verification_status', 'purpose', 'initial_list_status', 'application_type']

        #x[categorical] = x[categorical].astype(str)

            numerical = [col for col in x.columns if col not in categorical]
            num_indices = [x.columns.get_loc(col) for col in numerical]
            col_order = list(numerical) + list(categorical)
            x = x[col_order]
            feature_names = x.columns.to_list()
            cat_indices = [x.columns.get_loc(col) for col in categorical]

            x = x.to_numpy()
            x_test, y_test = x[splits['test']], y[splits['test']]
            charged_off = np.where(y_test == 1)[0]
            x_charged_off, y_charged_off = x_test[charged_off], y_test[charged_off] 

            metadata = pd.read_csv('./ressources/lcld_v2_metadata_transformed.csv')
            min_constraints = metadata['min'].to_list()
            min_constraints = list(map(float, min_constraints))
            max_constraints = metadata['max'].to_list()
            max_constraints = list(map(float, max_constraints))
            feature_types = metadata['type'].to_list()
            mutables = metadata.index[metadata['mutable'] == True].tolist()
            int_features = np.where(np.array(feature_types) == 'int')[0]
            cat_features = np.where(np.array(feature_types) == 'cat')[0]
            int_features = list(int_features) + list(cat_features)
            features_min_max = (min_constraints, max_constraints)
            constraints = get_lcld_relation_constraints()

            return x_charged_off, y_charged_off, mutables, features_min_max, int_features, feature_names, constraints 
        else :
            raise NotImplementedError