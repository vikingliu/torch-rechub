"""
    util function for movielens data.
"""

import collections
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from torch_rechub.utils.match import Annoy
from torch_rechub.basic.metric import topk_metrics
from collections import Counter
from torch_rechub.basic.features import DenseFeature, SparseFeature, SequenceFeature
from torch_rechub.utils.match import generate_seq_feature_match, gen_model_input
from torch_rechub.utils.data import df_to_dict, array_replace_with_dict


class ModelData(object):

    def __init__(self):
        self.user_features = []
        self.history_features = []
        self.item_features = []
        self.neg_item_features = []
        self.x_train = []
        self.y_train = []
        self.all_item = []
        self.test_user = []
        self.num_users = 0
        self.num_items = 0


def get_movielens_data(data_path, load_cache=False, embed_dim=16,
                       sample_method=1,  # 0 随机， 1.热度加权采样， 2. log(count+1)+1e-6, 3.tencent RALM
                       mode=2,  # 0 point-wise, 1 pair-wise, 2 list-wise
                       neg_ratio=3,  # 采样数量
                       min_item=0,  # 最小 history item
                       seq_max_len=50,  # 保留最大 history item 数量
                       history_features_pooling="concat",  # history features pooling 方法：mean, concat
                       sample_weight=False,  # 增加一列 sample weight
                       item_feature_clos=['movie_id'],  # item feature 列名
                       neg_cat=False):  # 负采样的时候 是否增加 neg_cat 列
    data = pd.read_csv(data_path)
    # genres 类型 x|y|z; 取第一个分类，这里处理不是很好
    data["cate_id"] = data["genres"].apply(lambda x: x.split("|")[0])
    # 稀疏特征
    sparse_features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip', "cate_id"]
    user_col, item_col, label_col = "user_id", "movie_id", "label"
    modeldata = ModelData()
    feature_max_idx = {}
    for feature in sparse_features:
        lbe = LabelEncoder()
        # label -> id + 1
        data[feature] = lbe.fit_transform(data[feature]) + 1
        # max id + 1
        feature_max_idx[feature] = data[feature].max() + 1
        if feature == user_col:
            user_map = {encode_id + 1: raw_id for encode_id, raw_id in
                        enumerate(lbe.classes_)}  # encode user id: raw user id
        if feature == item_col:
            item_map = {encode_id + 1: raw_id for encode_id, raw_id in
                        enumerate(lbe.classes_)}  # encode item id: raw item id
    # user_map 和 item 存起来
    np.save("./data/ml-1m/saved/raw_id_maps.npy", np.array((user_map, item_map), dtype=object))

    user_cols = ['user_id', 'gender', 'age', 'occupation', 'zip']
    item_cols = ["movie_id", "cate_id"]
    if sample_weight:
        # 获取每个 movie_id的权重
        sample_weight_map = get_item_sample_weight(data[item_col].tolist())
        # 增加一列，sample_weight，值就是movie_id的权重
        data["sample_weight"] = data[item_col].map(sample_weight_map)
        item_cols += ["sample_weight"]
        modeldata.sample_weight_feature = [DenseFeature("sample_weight")]  # it is one of item feature
    user_profile = data[user_cols].drop_duplicates('user_id')
    item_profile = data[item_cols].drop_duplicates('movie_id')

    if load_cache:  # if you have run this script before and saved the preprocessed data
        x_train, y_train, x_test = np.load("./data/ml-1m/saved/data_cache.npy", allow_pickle=True)
    else:
        # Note: list-wise or pair-wise negative sample generate, saved in last col "neg_items"
        # mode = 1(pair-wise) or 2(list-wise),
        # --df_tain/df_test = [user_col, item_col, hist_movie_id, histlen_movie_id, neg_items]
        # mode = 0(point-wise)
        # --df_tain/df_test = [user_col, item_col, hist_movie_id, histlen_movie_id, label]
        df_train, df_test = generate_seq_feature_match(data,
                                                       user_col,
                                                       item_col,
                                                       time_col="timestamp",
                                                       item_attribute_cols=[],
                                                       sample_method=sample_method,
                                                       mode=mode,
                                                       neg_ratio=neg_ratio,
                                                       min_item=min_item)
        # x_train =  [user_col, user_profile,  item_col, item_profile, hist_movie_id, histlen_movie_id, neg_items]
        x_train = gen_model_input(df_train, user_profile, user_col, item_profile, item_col, seq_max_len=seq_max_len,
                                  padding='post', truncating='post')
        if neg_cat:
            cate_map = dict(zip(item_profile["movie_id"], item_profile["cate_id"]))
            # generate negative sample feature
            x_train["neg_cate"] = array_replace_with_dict(x_train["neg_items"], cate_map)

        # if mode=1 or 2, the label col is useless.
        x_test = gen_model_input(df_test, user_profile, user_col, item_profile, item_col, seq_max_len=seq_max_len,
                                 padding='post', truncating='post')
        #
        # mode=0 取 label， mode=1 不需要
        # mode=2, label=0 means the first pred value is positive sample, label=0 表示排序第一位
        y_train = x_train["label"] if mode == 0 else np.array([0] * df_train.shape[0])
        np.save("./data/ml-1m/saved/data_cache.npy", np.array((x_train, y_train, x_test), dtype=object))

    modeldata.x_train = x_train
    modeldata.y_train = y_train
    modeldata.user_features = [
        SparseFeature(name,
                      vocab_size=feature_max_idx[name],
                      embed_dim=embed_dim)
        for name in user_cols
    ]

    modeldata.history_features = [
        SequenceFeature("hist_movie_id",
                        vocab_size=feature_max_idx["movie_id"],
                        embed_dim=embed_dim,
                        pooling=history_features_pooling,
                        shared_with="movie_id")
    ]

    modeldata.item_features = [
        SparseFeature(name,
                      vocab_size=feature_max_idx[name],
                      embed_dim=embed_dim)
        for name in item_feature_clos
    ]

    if mode == 1:
        modeldata.neg_item_features = [
            SparseFeature("neg_items", vocab_size=feature_max_idx["movie_id"], embed_dim=16)
        ]
    elif mode == 2:
        modeldata.neg_item_features = [
            SequenceFeature('neg_items',
                            vocab_size=feature_max_idx[item_col],
                            embed_dim=embed_dim,
                            pooling="concat",
                            shared_with=item_col)
        ]

    if neg_cat:
        modeldata.neg_item_features.append(
            SparseFeature("neg_cate", vocab_size=feature_max_idx["cate_id"], embed_dim=16)
        )

    modeldata.all_item = df_to_dict(item_profile)
    modeldata.test_user = x_test
    modeldata.num_users, modeldata.num_items = feature_max_idx['user_id'], feature_max_idx['movie_id']
    return modeldata


def get_item_sample_weight(items):
    # It is a effective weight used in word2vec
    items_cnt = Counter(items)
    # item weight =  item_count ** 0.75 / sum(all_count)
    p_sample = {item: count ** 0.75 for item, count in items_cnt.items()}
    p_sum = sum([v for k, v in p_sample.items()])
    item_sample_weight = {k: v / p_sum for k, v in p_sample.items()}
    return item_sample_weight


def match_evaluation(user_embedding, item_embedding, test_user, all_item, user_col='user_id', item_col='movie_id',
                     raw_id_maps="./data/ml-1m/saved/raw_id_maps.npy", topk=10):
    print("evaluate embedding matching on test data")
    annoy = Annoy(n_trees=10)
    annoy.fit(item_embedding)

    # for each user of test dataset, get ann search topk result
    print("matching for topk")
    user_map, item_map = np.load(raw_id_maps, allow_pickle=True)
    match_res = collections.defaultdict(dict)  # user id -> predicted item ids
    for user_id, user_emb in zip(test_user[user_col], user_embedding):
        if len(user_emb.shape) == 2:
            # 多兴趣召回, user_emb = [user_emb1, user_emb2,...]
            items_idx = []
            items_scores = []
            for i in range(user_emb.shape[0]):
                temp_items_idx, temp_items_scores = annoy.query(v=user_emb[i], n=topk)  # the index of topk match items
                items_idx += temp_items_idx
                items_scores += temp_items_scores
            temp_df = pd.DataFrame()
            temp_df['item'] = items_idx
            temp_df['score'] = items_scores
            temp_df = temp_df.sort_values(by='score', ascending=True)
            temp_df = temp_df.drop_duplicates(subset=['item'], keep='first', inplace=False)
            recall_item_list = temp_df['item'][:topk].values
            match_res[user_map[user_id]] = np.vectorize(item_map.get)(all_item[item_col][recall_item_list])
        else:
            # 普通召回
            items_idx, items_scores = annoy.query(v=user_emb, n=topk)  # the index of topk match items
            match_res[user_map[user_id]] = np.vectorize(item_map.get)(all_item[item_col][items_idx])

    # get ground truth
    print("generate ground truth")

    data = pd.DataFrame({user_col: test_user[user_col], item_col: test_user[item_col]})
    data[user_col] = data[user_col].map(user_map)
    data[item_col] = data[item_col].map(item_map)
    user_pos_item = data.groupby(user_col).agg(list).reset_index()
    ground_truth = dict(zip(user_pos_item[user_col], user_pos_item[item_col]))  # user id -> ground truth

    print("compute topk metrics")
    out = topk_metrics(y_true=ground_truth, y_pred=match_res, topKs=[topk])
    print(out)
