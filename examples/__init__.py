import argparse


class RunnerBase(object):

    def __init__(self, default_model_name='',
                 default_dataset_path="",
                 default_epoch=10,
                 default_learning_rate=1e-3,
                 default_batch_size=256,
                 default_weight_decay=1e-6,
                 default_seq_max_len=50,
                 default_device='cpu',
                 default_save_dir='',
                 default_seed=2022,
                 default_earlystop_patience=10,
                 ):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--dataset_path', default=default_dataset_path)
        self.parser.add_argument('--model_name', default=default_model_name)
        self.parser.add_argument('--epoch', type=int, default=default_epoch)  # 10
        self.parser.add_argument('--learning_rate', type=float, default=default_learning_rate)
        self.parser.add_argument('--batch_size', type=int, default=default_batch_size)  # 4096
        self.parser.add_argument('--weight_decay', type=float, default=default_weight_decay)
        self.parser.add_argument('--seq_max_len', type=float, default=default_seq_max_len)
        self.parser.add_argument('--device', default=default_device)  # cuda:0
        self.parser.add_argument('--save_dir', default=default_save_dir)
        self.parser.add_argument('--seed', type=int, default=default_seed)
        self.parser.add_argument('--earlystop_patience', type=int, default=default_earlystop_patience)

    def add_argument(self, key, type, default):
        self.parser.add_argument(key, type=type, default=default)

    def set_default(self, key, value):
        self.parser.set_defaults(**{key: value})

    def get_default(self, key):
        return self.parser.get_default(key)

    def get_args(self):
        self.args = self.parser.parse_args()
        return self.args
