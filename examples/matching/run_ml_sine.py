from torch_rechub.models.matching import SINE
from examples.matching.movielens_utils import get_movielens_data
from examples.matching import Runner

if __name__ == '__main__':
    runner = Runner(default_model_name="SINE")
    runner.add_argument('--embedding_dim', type=int, default=128)
    runner.add_argument('--hidden_dim', type=int, default=512)
    runner.add_argument('--num_concept', type=int, default=10)
    runner.add_argument('--num_intention', type=int, default=2)
    runner.add_argument('--temperature', type=int, default=0.1)
    args = runner.get_args()

    # mode=2 list-wise negative-sample
    modeldata = get_movielens_data(args.dataset_path, mode=2, seq_max_len=args.seq_max_len)

    item_features, history_features, neg_item_features = ["movie_id"], ["hist_movie_id"], [
        "neg_items"]
    model = SINE(history_features, item_features, neg_item_features, modeldata.num_items, args.embedding_dim,
                 args.hidden_dim, args.num_concept, args.num_intention, args.seq_max_len, temperature=args.temperature)

    #list-wise loss_func = CrossEntropyLoss
    runner.run(modeldata, model, mode=2, gpus=[0])
