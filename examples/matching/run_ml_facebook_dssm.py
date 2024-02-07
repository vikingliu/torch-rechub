from torch_rechub.models.matching import FaceBookDSSM
from examples.matching.movielens_utils import get_movielens_data
from examples.matching import Runner

if __name__ == '__main__':
    runner = Runner(default_model_name="FaceBookDSSM")
    args = runner.get_args()
    # mode=1 pair-wise negative sample
    # 增加了两个 neg_items, neg_cat
    modeldata = get_movielens_data(
        args.dataset_path,
        mode=1,
        neg_cat=True,
        item_feature_clos=['movie_id', "cate_id"],
        history_features_pooling="mean",
        seq_max_len=args.seq_max_len
    )
    modeldata.user_features += modeldata.history_features
    model = FaceBookDSSM(modeldata.user_features,
                         modeldata.item_features,
                         modeldata.neg_item_features,
                         temperature=0.02,
                         user_params={
                             "dims": [256, 128, 64, 32],
                         },
                         item_params={
                             "dims": [256, 128, 64, 32],
                         })
    # pair-wise loss_func= BPRLoss
    runner.run(modeldata, model, mode=1)
