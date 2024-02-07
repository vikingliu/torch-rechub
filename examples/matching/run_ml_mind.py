from torch_rechub.models.matching import MIND
from examples.matching.movielens_utils import get_movielens_data
from examples.matching import Runner

if __name__ == '__main__':
    runner = Runner(default_model_name="MIND")
    args = runner.get_args()

    # mode=2 list-wise negative-sample
    modeldata = get_movielens_data(args.dataset_path, mode=2, seq_max_len=args.seq_max_len)

    model = MIND(modeldata.user_features, modeldata.history_features, modeldata.item_features,
                 modeldata.neg_item_features,
                 max_length=args.seq_max_len,
                 temperature=0.02)
    # list-wise loss_func = CrossEntropyLoss
    runner.run(modeldata, model, mode=2)
