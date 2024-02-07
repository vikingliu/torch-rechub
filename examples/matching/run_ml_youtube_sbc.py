from torch_rechub.models.matching import YoutubeSBC
from examples.matching.movielens_utils import get_movielens_data
from examples.matching import Runner

if __name__ == '__main__':
    runner = Runner(default_model_name="YoutubeSBC")
    args = runner.get_args()

    # mode=0, point-wise negative-sample
    modeldata = get_movielens_data(args.dataset_path, load_cache=False,
                                   mode=0,
                                   neg_ratio=0,  # batch里面做负采样
                                   sample_weight=True, history_features_pooling="mean")

    modeldata.user_features += modeldata.history_features

    model = YoutubeSBC(modeldata.user_features,
                       modeldata.item_features,
                       modeldata.sample_weight_feature,
                       user_params={"dims": [128, 64, 16]},
                       item_params={"dims": [128, 64, 16]},
                       batch_size=args.batch_size,  # !! should be same as batch size of dataloader
                       n_neg=3,
                       temperature=0.02)
    # list-wise loss_func=CrossEntropyLoss
    runner.run(modeldata, model, mode=2)
