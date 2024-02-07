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


def get_avazu_data_dict(data_path):
    df_train = pd.read_csv(data_path + "/train_sample.csv")
    df_val = pd.read_csv(data_path + "/valid_sample.csv")
    df_test = pd.read_csv(data_path + "/test_sample.csv")
    print("train : val : test = %d %d %d" % (len(df_train), len(df_val), len(df_test)))
    train_idx, val_idx = df_train.shape[0], df_train.shape[0] + df_val.shape[0]
    data = pd.concat([df_train, df_val, df_test], axis=0)
    print("data load finished")
    features = [f for f in data.columns.tolist() if f[0] == "f"]
    dense_features = features[:3]
    sparse_features = features[3:]
    data[sparse_features] = data[sparse_features].fillna("-996", )
    data[dense_features] = data[dense_features].fillna(0, )
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
    sparse_feas = [
        SparseFeature(
            feature_name, vocab_size=data[feature_name].nunique(), embed_dim=16
        )
        for feature_name in features
    ]
    modeldata.ffm_linear_feas = [
        SparseFeature(
            feature.name, vocab_size=feature.vocab_size, embed_dim=1
        )
        for feature in sparse_feas
    ]
    modeldata.ffm_cross_feas = [
        SparseFeature(
            feature.name, vocab_size=feature.vocab_size * len(sparse_feas),
            embed_dim=10
        )
        for feature in sparse_feas
    ]
    modeldata.sparse_feas = sparse_feas
    y = data["label"]
    del data["label"]
    x = data
    modeldata.x_train, modeldata.y_train = x[:train_idx], y[:train_idx]
    modeldata.x_val, modeldata.y_val = x[train_idx:val_idx], y[train_idx:val_idx]
    modeldata.x_test, modeldata.y_test = x[val_idx:], y[val_idx:]

    return modeldata


if __name__ == "__main__":
    runner = Runner(default_dataset_path="./data/avazu",
                    default_model_name='fat_deepffm',
                    default_weight_decay=1e-5)
    args = runner.get_args()
    modeldata = get_avazu_data_dict(args.dataset_path)
    runner.run_ctr(modeldata)

"""
python run_avazu.py --model_name widedeep
python run_avazu.py --model_name deepfm
python run_avazu.py --model_name dcn
python run_avazu.py --model_name deepffm
python run_avazu.py --model_name fat_deepffm
"""
