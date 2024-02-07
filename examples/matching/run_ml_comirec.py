from torch_rechub.models.matching import ComirecSA, ComirecDR
from examples.matching.movielens_utils import get_movielens_data
from examples.matching import Runner

if __name__ == '__main__':
    runner = Runner(default_model_name='comirec_sa')
    args = runner.get_args()
    # mode =2 list-wise negative-sample
    modeldata = get_movielens_data(args.dataset_path, mode=2, seq_max_len=args.seq_max_len)

    if args.model_name.lower() == 'comirec_dr':
        model = ComirecDR(modeldata.user_features, modeldata.history_features, modeldata.item_features,
                          modeldata.neg_item_features, max_length=args.seq_max_len,
                          temperature=0.02)
    else:
        model = ComirecSA(modeldata.user_features, modeldata.history_features, modeldata.item_features,
                          modeldata.neg_item_features, temperature=0.02, )
    # list-wise loss_func = CrossEntropyLoss
    runner.run(modeldata, model, mode=2)
