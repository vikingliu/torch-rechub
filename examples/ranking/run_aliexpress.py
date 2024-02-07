import pandas as pd
from torch_rechub.basic.features import DenseFeature, SparseFeature
from examples.ranking import Runner, ModelData


def get_aliexpress_data_dict(data_path='./data/aliexpress'):
    df_train = pd.read_csv(data_path + '/aliexpress_train_sample.csv')
    df_test = pd.read_csv(data_path + '/aliexpress_test_sample.csv')
    print("train : test = %d %d" % (len(df_train), len(df_test)))
    train_idx = df_train.shape[0]
    data = pd.concat([df_train, df_test], axis=0)
    col_names = data.columns.values.tolist()
    sparse_cols = [name for name in col_names if name.startswith("categorical")]  # categorical
    dense_cols = [name for name in col_names if name.startswith("numerical")]  # numerical
    print("sparse cols:%d dense cols:%d" % (len(sparse_cols), len(dense_cols)))
    label_cols = ["conversion", "click"]

    used_cols = sparse_cols + dense_cols
    modeldata = ModelData()
    modeldata.features = [SparseFeature(col, data[col].max() + 1, embed_dim=5) for col in sparse_cols] \
                         + [DenseFeature(col) for col in dense_cols]
    modeldata.x_train, modeldata.y_train = {name: data[name].values[:train_idx] for name in used_cols}, \
                                           data[label_cols].values[:train_idx]
    modeldata.x_test, modeldata.y_test = {name: data[name].values[train_idx:] for name in used_cols}, \
                                         data[label_cols].values[train_idx:]
    return modeldata


if __name__ == '__main__':
    runner = Runner(default_dataset_path="./data/aliexpress",
                    default_model_name='MMOE',
                    default_weight_decay=1e-5)
    args = runner.get_args()
    modelparams = {
        'SharedBottom': {
            'bottom_params': {"dims": [192, 96, 48]},
            'tower_params_list': [{"dims": [8]}, {"dims": [8]}]
        },
        'MMOE': {
            'n_expert': 3,
            'expert_params': {"dims": [64, 32, 16]},
            'tower_params_list': [{"dims": [8]}, {"dims": [8]}]
        },

        'PLE': {
            'n_level': 1,
            'n_expert_specific': 1,
            'n_expert_shared': 1,
            'expert_params': {"dims": [64, 32, 16]},
            'tower_params_list': [{"dims": [8]}, {"dims": [8]}]
        },
        'AITM': {
            'n_task': 2,
            'bottom_params': {"dims": [128, 64, 32]},
            'tower_params_list': [{"dims": [8]}, {"dims": [8]}]
        }

    }
    modeldata = get_aliexpress_data_dict(args.dataset_path)
    runner.run_multi(modeldata, modelparams)
"""
python run_aliexpress.py --model_name SharedBottom
python run_aliexpress.py --model_name ESMM
python run_aliexpress.py --model_name MMOE
python run_aliexpress.py --model_name PLE
python run_aliexpress.py --model_name AITM
"""
