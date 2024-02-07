import pandas as pd
from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.utils.data import generate_seq_feature, df_to_dict

from examples.ranking import Runner, ModelData


def get_amazon_data_dict(dataset_path):
    data = pd.read_csv(dataset_path)
    print('========== Start Amazon ==========')
    train, val, test = generate_seq_feature(data=data, user_col="user_id", item_col="item_id", time_col='time',
                                            item_attribute_cols=["cate_id"])
    print('INFO: Now, the dataframe named: ', train.columns)
    n_users, n_items, n_cates = data["user_id"].max(), data["item_id"].max(), data["cate_id"].max()
    print(train)

    modeldata = ModelData()

    features = [SparseFeature("target_item_id", vocab_size=n_items + 1, embed_dim=8),
                SparseFeature("target_cate_id", vocab_size=n_cates + 1, embed_dim=8),
                SparseFeature("user_id", vocab_size=n_users + 1, embed_dim=8)]
    modeldata.features = features
    modeldata.target_feas = features
    modeldata.history_feas = [
        SequenceFeature("hist_item_id", vocab_size=n_items + 1, embed_dim=8, pooling="concat",
                        shared_with="target_item_id"),
        SequenceFeature("hist_cate_id", vocab_size=n_cates + 1, embed_dim=8, pooling="concat",
                        shared_with="target_cate_id")
    ]

    print('========== Generate input dict ==========')
    train = df_to_dict(train)
    val = df_to_dict(val)
    test = df_to_dict(test)
    modeldata.y_train, modeldata.y_val, modeldata.y_test = train["label"], val["label"], test["label"]

    del train["label"]
    del val["label"]
    del test["label"]
    modeldata.x_train, modeldata.x_val, modeldata.x_test = train, val, test
    return modeldata


if __name__ == '__main__':
    runner = Runner(default_dataset_path="./data/amazon-electronics/amazon_electronics_sample.csv",
                    default_model_name='din',
                    default_weight_decay=1e-3,
                    default_earlystop_patience=4)
    args = runner.get_args()
    modeldata = get_amazon_data_dict(args.dataset_path)
    runner.run_ctr(modeldata)
"""
python run_amazon_electronics.py
"""
