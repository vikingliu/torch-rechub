from torch_rechub.models.matching import DSSM
from examples.matching.movielens_utils import get_movielens_data
from examples.matching import Runner

if __name__ == '__main__':
    runner = Runner(default_model_name="DSSM")
    args = runner.get_args()
    # mode=0, point-wise negative-sample
    modeldata = get_movielens_data(args.dataset_path,
                                   mode=0,
                                   item_feature_clos=['movie_id', "cate_id"],
                                   history_features_pooling='mean',
                                   seq_max_len=args.seq_max_len)
    modeldata.user_features += modeldata.history_features
    model = DSSM(modeldata.user_features,
                 modeldata.item_features,
                 temperature=0.02,
                 user_params={
                     "dims": [256, 128, 64],
                     "activation": 'prelu',  # important!!
                 },
                 item_params={
                     "dims": [256, 128, 64],
                     "activation": 'prelu',  # important!!
                 })
    # point wise  loss_func = BCELoss
    runner.run(modeldata, model, mode=0)
