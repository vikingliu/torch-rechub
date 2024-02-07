import pandas as pd
from torch_rechub.basic.features import DenseFeature, SparseFeature
from examples.ranking import Runner, ModelData


def get_census_data_dict(model_name, data_path='./data/census-income'):
    df_train = pd.read_csv(data_path + '/census_income_train_sample.csv')
    df_val = pd.read_csv(data_path + '/census_income_val_sample.csv')
    df_test = pd.read_csv(data_path + '/census_income_test_sample.csv')
    print("train : val : test = %d %d %d" % (len(df_train), len(df_val), len(df_test)))
    train_idx, val_idx = df_train.shape[0], df_train.shape[0] + df_val.shape[0]
    data = pd.concat([df_train, df_val, df_test], axis=0)
    data = data.fillna(0)  #Nan -> 0
    # task 1 (as cvr): main task, income prediction
    # task 2(as ctr): auxiliary task, marital status prediction
    data.rename(columns={'income': 'cvr_label', 'marital status': 'ctr_label'}, inplace=True)
    data["ctcvr_label"] = data['cvr_label'] * data['ctr_label']

    col_names = data.columns.values.tolist()
    dense_cols = ['age', 'wage per hour', 'capital gains', 'capital losses', 'divdends from stocks',
                  'num persons worked for employer', 'weeks worked in year']
    sparse_cols = [col for col in col_names if
                   col not in dense_cols and col not in ['cvr_label', 'ctr_label', 'ctcvr_label']]
    print("sparse cols:%d dense cols:%d" % (len(sparse_cols), len(dense_cols)))
    modeldata = ModelData()
    # define dense and sparse features
    if model_name == "ESMM":
        label_cols = ['cvr_label', 'ctr_label', "ctcvr_label"]  # the order of 3 labels must fixed as this
        # ESMM only for sparse features in origin paper
        user_cols = ['industry code', 'occupation code', 'race', 'education',
                     'sex']  # assumption features split for user and item
        item_cols = [col for col in sparse_cols if col not in user_cols]
        modeldata.user_features = [SparseFeature(col, data[col].max() + 1, embed_dim=16) for col in user_cols]
        modeldata.item_features = [SparseFeature(col, data[col].max() + 1, embed_dim=16) for col in item_cols]
        modeldata.x_train, modeldata.y_train = {name: data[name].values[:train_idx] for name in sparse_cols}, \
                                               data[label_cols].values[:train_idx]
        modeldata.x_val, modeldata.y_val = {name: data[name].values[train_idx:val_idx] for name in sparse_cols}, \
                                           data[label_cols].values[train_idx:val_idx]
        modeldata.x_test, modeldata.y_test = {name: data[name].values[val_idx:] for name in sparse_cols}, \
                                             data[label_cols].values[val_idx:]
        # return user_features, item_features, x_train, y_train, x_val, y_val, x_test, y_test
    else:
        label_cols = ['cvr_label', 'ctr_label']  # the order of labels can be any
        used_cols = sparse_cols + dense_cols
        modeldata.features = [SparseFeature(col, data[col].max() + 1, embed_dim=4) for col in sparse_cols] \
                             + [DenseFeature(col) for col in dense_cols]
        modeldata.x_train, modeldata.y_train = {name: data[name].values[:train_idx] for name in used_cols}, \
                                               data[label_cols].values[:train_idx]
        modeldata.x_val, modeldata.y_val = {name: data[name].values[train_idx:val_idx] for name in used_cols}, \
                                           data[label_cols].values[train_idx:val_idx]
        modeldata.x_test, modeldata.y_test = {name: data[name].values[val_idx:] for name in used_cols}, \
                                             data[label_cols].values[val_idx:]
        # return features, x_train, y_train, x_val, y_val, x_test, y_test
    return modeldata


if __name__ == '__main__':
    runner = Runner(default_dataset_path="./data/census-income",
                    default_model_name='MMOE',
                    default_weight_decay=1e-4)
    args = runner.get_args()
    modelparams = {
        'SharedBottom': {
            'bottom_params': {"dims": [117]},
            'tower_params_list': [{"dims": [8]}, {"dims": [8]}]
        },
        'ESMM': {
            'cvr_params': {"dims": [16, 8]},
            'ctr_params': {"dims": [16, 8]}
        },
        'MMOE': {
            'n_expert': 8,
            'expert_params': {"dims": [16]},
            'tower_params_list': [{"dims": [8]}, {"dims": [8]}]
        },

        'PLE': {
            'n_level': 1,
            'n_expert_specific': 2,
            'n_expert_shared': 1,
            'expert_params': {"dims": [16]},
            'tower_params_list': [{"dims": [8]}, {"dims": [8]}]
        },
        'AITM': {
            'n_task': 2,
            'bottom_params': {"dims": [32, 16]},
            'tower_params_list': [{"dims": [8]}, {"dims": [8]}]
        }

    }
    modeldata = get_census_data_dict(args.model_name, args.dataset_path)
    runner.run_multi(modeldata, modelparams)

"""
python run_census.py --model_name SharedBottom
python run_census.py --model_name ESMM
python run_census.py --model_name MMOE
python run_census.py --model_name PLE
python run_census.py --model_name AITM
"""
