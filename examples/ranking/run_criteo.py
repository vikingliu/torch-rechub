import numpy as np
import pandas as pd
from torch_rechub.basic.features import DenseFeature, SparseFeature
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from examples.ranking import Runner, ModelData


def convert_numeric_feature(val):
    v = int(val)
    if v > 2:
        return int(np.log(v) ** 2)
    else:
        return v - 2


def get_criteo_data_dict(data_path):
    if data_path.endswith(".gz"):  # if the raw_data is gz file:
        data = pd.read_csv(data_path, compression="gzip")
    else:
        data = pd.read_csv(data_path)
    print("data load finished")
    dense_features = [f for f in data.columns.tolist() if f[0] == "I"]
    sparse_features = [f for f in data.columns.tolist() if f[0] == "C"]

    data[sparse_features] = data[sparse_features].fillna('0')
    data[dense_features] = data[dense_features].fillna(0)

    for feat in tqdm(dense_features):  # discretize dense feature and as new sparse feature
        sparse_features.append(feat + "_cat")
        data[feat + "_cat"] = data[feat].apply(lambda x: convert_numeric_feature(x))

    sca = MinMaxScaler()  # scaler dense feature
    data[dense_features] = sca.fit_transform(data[dense_features])
    modeldata = ModelData()
    for feat in tqdm(sparse_features):  # encode sparse feature
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    modeldata.dense_feas = [DenseFeature(feature_name) for feature_name in dense_features]
    sparse_feas = [SparseFeature(feature_name, vocab_size=data[feature_name].nunique(), embed_dim=16) for feature_name
                   in sparse_features]
    modeldata.ffm_linear_feas = [SparseFeature(feature.name, vocab_size=feature.vocab_size, embed_dim=1) for feature in
                                 sparse_feas]
    modeldata.ffm_cross_feas = [
        SparseFeature(feature.name, vocab_size=feature.vocab_size * len(sparse_feas), embed_dim=10) for
        feature in sparse_feas]
    modeldata.sparse_feas = sparse_feas
    modeldata.y_train = data["label"]
    del data["label"]
    modeldata.x_train = data
    modeldata.has_test_val = False
    return modeldata


if __name__ == '__main__':
    runner = Runner(default_dataset_path="./data/criteo/criteo_sample.csv",
                    default_model_name='fibinet',
                    default_weight_decay=1e-5)
    args = runner.get_args()
    modeldata = get_criteo_data_dict(args.dataset_path)
    runner.run_ctr(modeldata)

"""
python run_criteo.py --model_name widedeep
python run_criteo.py --model_name deepfm
python run_criteo.py --model_name dcn
python run_criteo.py --model_name dcn_v2
python run_criteo.py --model_name edcn
python run_criteo.py --model_name deepffm
python run_criteo.py --model_name fat_deepffm
"""
