import pandas as pd

from examples.ranking import Runner, ModelData
from torch_rechub.basic.features import DenseFeature, SparseFeature


def get_ali_ccp_data_dict(data_path):
    # 训练集
    df_train = pd.read_csv(data_path + '/ali_ccp_train_sample.csv')
    # 验证集
    df_val = pd.read_csv(data_path + '/ali_ccp_val_sample.csv')
    # 测试集
    df_test = pd.read_csv(data_path + '/ali_ccp_test_sample.csv')
    print("train : val : test = %d %d %d" % (len(df_train), len(df_val), len(df_test)))
    train_idx, val_idx = df_train.shape[0], df_train.shape[0] + df_val.shape[0]
    data = pd.concat([df_train, df_val, df_test], axis=0)

    col_names = data.columns.values.tolist()
    dense_cols = ['D109_14', 'D110_14', 'D127_14', 'D150_14', 'D508', 'D509', 'D702', 'D853']
    sparse_cols = [col for col in col_names if col not in dense_cols and col not in ['click', 'purchase']]
    print("sparse cols:%d dense cols:%d" % (len(sparse_cols), len(dense_cols)))
    modeldata = ModelData()

    modeldata.dense_feas = [DenseFeature(col) for col in dense_cols]
    modeldata.sparse_feas = [SparseFeature(col, vocab_size=data[col].max() + 1, embed_dim=16) for col in sparse_cols]

    y = data["click"]
    del data["click"]
    x = data
    modeldata.x_train, modeldata.y_train = x[:train_idx], y[:train_idx]
    modeldata.x_val, modeldata.y_val = x[train_idx:val_idx], y[train_idx:val_idx]
    modeldata.x_test, modeldata.y_test = x[val_idx:], y[val_idx:]

    return modeldata


if __name__ == '__main__':
    runner = Runner(default_dataset_path="./data/ali-ccp",
                    default_model_name='widedeep',
                    default_weight_decay=1e-3)
    args = runner.get_args()
    modeldata = get_ali_ccp_data_dict(args.dataset_path)
    runner.run_ctr(modeldata)
"""
python run_ali_ccp_ctr_ranking.py --model_name widedeep
python run_ali_ccp_ctr_ranking.py --model_name deepfm
python run_ali_ccp_ctr_ranking.py --model_name dcn
"""
