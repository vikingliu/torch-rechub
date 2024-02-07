import os
import torch

from torch_rechub.trainers import MatchTrainer
from torch_rechub.utils.data import MatchDataGenerator
from examples.matching.movielens_utils import match_evaluation
from examples import RunnerBase


class Runner(RunnerBase):

    def __init__(self, default_model_name='',
                 default_dataset_path="./data/ml-1m/ml-1m_sample.csv",
                 default_epoch=10,
                 default_learning_rate=1e-3,
                 default_batch_size=256,
                 default_weight_decay=1e-6,
                 default_seq_max_len=50,
                 default_device='cpu',
                 default_save_dir='./data/ml-1m/saved/',
                 default_seed=2022,
                 default_earlystop_patience=10,
                 ):
        super(Runner, self).__init__(default_model_name, default_dataset_path,
                                     default_epoch, default_learning_rate, default_batch_size,
                                     default_weight_decay, default_seq_max_len, default_device, default_save_dir,
                                     default_seed, default_earlystop_patience)

    def run(self, modeldata, model, mode=1, gpus=None):
        """

        Args:
            modeldata: model train and test data
            model: match model
            mode: 0: point-wise, 1: pair-wise, 2: list-wise
            gpus: gpu list

        Returns:

        """
        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir)
        # 设置随机种子是为了确保每次生成固定的随机数，这就使得每次实验结果显示一致了，有利于实验的比较和改进。
        torch.manual_seed(self.args.seed)

        trainer = MatchTrainer(model,
                               mode=mode,
                               optimizer_params={
                                   "lr": self.args.learning_rate,
                                   "weight_decay": self.args.weight_decay
                               },
                               n_epoch=self.args.epoch,
                               earlystop_patience=self.args.earlystop_patience,
                               device=self.args.device,
                               model_path=self.args.save_dir,
                               gpus=gpus)

        dg = MatchDataGenerator(x=modeldata.x_train, y=modeldata.y_train)

        # 训练数据，测试用户数据，所有item数据
        train_dl, test_user_dl, all_item_dl = dg.generate_dataloader(modeldata.test_user, modeldata.all_item,
                                                                     batch_size=self.args.batch_size,
                                                                     num_workers=0)
        # 模型训练, 没有验证集
        trainer.fit(train_dl)

        print("inference embedding")
        # test user 计算 embedding
        user_embedding = trainer.inference_embedding(model=model, mode="user", data_loader=test_user_dl,
                                                     model_path=self.args.save_dir)

        # 所有的 item 计算 embedding
        item_embedding = trainer.inference_embedding(model=model, mode="item", data_loader=all_item_dl,
                                                     model_path=self.args.save_dir)
        print(user_embedding.shape, item_embedding.shape)
        # torch.save(user_embedding.data.cpu(), save_dir + "user_embedding.pth")
        # torch.save(item_embedding.data.cpu(), save_dir + "item_embedding.pth")
        # 模型效果评估
        match_evaluation(user_embedding, item_embedding, modeldata.test_user, modeldata.all_item, topk=10)
