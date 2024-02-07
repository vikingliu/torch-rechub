import torch
from examples import RunnerBase
from torch_rechub.models.ranking import WideDeep, DeepFM, DCN, DIN, DeepFFM, FatDeepFFM, DCNv2, FiBiNet, EDCN
from torch_rechub.models.multi_task import SharedBottom, ESMM, MMOE, PLE, AITM
from torch_rechub.trainers import MTLTrainer, CTRTrainer
from torch_rechub.utils.data import DataGenerator


class Runner(RunnerBase):
    def __init__(self, default_model_name,
                 default_dataset_path,
                 default_epoch=2,
                 default_learning_rate=1e-3,
                 default_batch_size=2048,
                 default_weight_decay=1e-6,
                 default_seq_max_len=50,
                 default_device='cpu',
                 default_save_dir='./data/saved',
                 default_seed=2022,
                 default_earlystop_patience=10,
                 ):
        super(Runner, self).__init__(default_model_name, default_dataset_path,
                                     default_epoch, default_learning_rate, default_batch_size,
                                     default_weight_decay, default_seq_max_len, default_device, default_save_dir,
                                     default_seed, default_earlystop_patience)

    def add_argument(self, key, type, default):
        self.parser.add_argument(key, type=type, default=default)

    def get_args(self):
        self.args = self.parser.parse_args()
        return self.args

    def run_ctr(self, modeldata, model_param_dims=[256, 128], dropout=0.2, n_cross_layers=3):
        torch.manual_seed(self.args.seed)

        dg = DataGenerator(modeldata.x_train, modeldata.y_train)
        if modeldata.has_test_val:
            train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(x_val=modeldata.x_val,
                                                                                       y_val=modeldata.y_val,
                                                                                       x_test=modeldata.x_test,
                                                                                       y_test=modeldata.y_test,
                                                                                       batch_size=self.args.batch_size)
        else:
            train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(split_ratio=[0.7, 0.1],
                                                                                       batch_size=self.args.batch_size)
        if self.args.model_name == "WideDeep":
            model = WideDeep(wide_features=modeldata.dense_feas, deep_features=modeldata.sparse_feas,
                             mlp_params={"dims": model_param_dims, "dropout": dropout, "activation": "relu"})
        elif self.args.model_name == "DeepFM":
            model = DeepFM(deep_features=modeldata.dense_feas, fm_features=modeldata.sparse_feas,
                           mlp_params={"dims": model_param_dims, "dropout": dropout, "activation": "relu"})
        elif self.args.model_name == "DCN":
            model = DCN(features=modeldata.dense_feas + modeldata.sparse_feas, n_cross_layers=n_cross_layers,
                        mlp_params={"dims": model_param_dims})
        elif self.args.model_name == "DCNv2":
            model = DCNv2(features=modeldata.dense_feas + modeldata.sparse_feas, n_cross_layers=n_cross_layers,
                          mlp_params={"dims": model_param_dims, "dropout": dropout, "activation": "relu"})
        elif self.args.model_name == "FiBiNet":
            model = FiBiNet(features=modeldata.dense_feas + modeldata.sparse_feas, reduction_ratio=n_cross_layers,
                            mlp_params={"dims": model_param_dims, "dropout": dropout, "activation": "relu"})
        elif self.args.model_name == "EDCN":
            model = EDCN(features=modeldata.dense_feas + modeldata.sparse_feas, n_cross_layers=n_cross_layers,
                         mlp_params={"dims": model_param_dims, "dropout": dropout, "activation": "relu"})
        elif self.args.model_name == 'DIN':
            model = DIN(features=modeldata.features, history_features=modeldata.history_feas,
                        target_features=modeldata.target_feas,
                        mlp_params={"dims": model_param_dims}, attention_mlp_params={"dims": [256, 128]})
        elif self.args.model_name == "DeepFFM":
            model = DeepFFM(
                linear_features=modeldata.ffm_linear_feas,
                cross_features=modeldata.ffm_cross_feas,
                embed_dim=10,
                mlp_params={"dims": [1600, 1600], "dropout": dropout, "activation": "relu"},
            )
        elif self.args.model_name == "FatDeepFFM":
            model = FatDeepFFM(
                linear_features=modeldata.ffm_linear_feas,
                cross_features=modeldata.ffm_cross_feas,
                embed_dim=10,
                reduction_ratio=1,
                mlp_params={"dims": [1600, 1600], "dropout": dropout, "activation": "relu"},
            )

        ctr_trainer = CTRTrainer(model,
                                 optimizer_params={"lr": self.args.learning_rate,
                                                   "weight_decay": self.args.weight_decay},
                                 n_epoch=self.args.epoch, earlystop_patience=self.args.earlystop_patience,
                                 device=self.args.device, model_path=self.args.save_dir)
        # scheduler_fn=torch.optim.lr_scheduler.StepLR,scheduler_params={"step_size": 2,"gamma": 0.8},
        ctr_trainer.fit(train_dataloader, val_dataloader)
        auc = ctr_trainer.evaluate(ctr_trainer.model, test_dataloader)
        print(f'test auc: {auc}')

    def run_multi(self, modeldata, modelparams):
        model_name = self.args.model_name
        params = modelparams[model_name]
        if model_name == "SharedBottom":
            task_types = ["classification", "classification"]
            model = SharedBottom(modeldata.features, task_types,
                                 bottom_params=params["bottom_params"],
                                 tower_params_list=params["tower_params_list"])
        elif model_name == "ESMM":
            task_types = ["classification", "classification", "classification"]  # cvr,ctr,ctcvr
            model = ESMM(modeldata.user_features, modeldata.item_features,
                         cvr_params=params["cvr_params"],
                         ctr_params=params["ctr_params"])
        elif model_name == "MMOE":
            task_types = ["classification", "classification"]
            model = MMOE(modeldata.features, task_types,
                         n_expert=params["n_expert"],
                         expert_params=params["expert_params"],
                         tower_params_list=params["tower_params_list"])
        elif model_name == "PLE":
            task_types = ["classification", "classification"]
            model = PLE(modeldata.features, task_types,
                        n_level=params["n_level"],
                        n_expert_specific=params["n_expert_specific"],
                        n_expert_shared=params["n_expert_shared"],
                        expert_params=params["expert_params"],
                        tower_params_list=params["tower_params_list"])
        elif model_name == "AITM":
            task_types = ["classification", "classification"]
            model = AITM(modeldata.features,
                         n_task=params["n_task"],
                         bottom_params=params["bottom_params"],
                         tower_params_list=params["tower_params_list"])

        dg = DataGenerator(modeldata.x_train, modeldata.y_train)
        if modeldata.has_test_val:
            train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(x_val=modeldata.x_val,
                                                                                       y_val=modeldata.y_val,
                                                                                       x_test=modeldata.x_test,
                                                                                       y_test=modeldata.y_test,
                                                                                       batch_size=self.args.batch_size)
        else:
            train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(split_ratio=[0.7, 0.1],
                                                                                       batch_size=self.args.batch_size)

        # adaptive weight loss:
        # mtl_trainer = MTLTrainer(model, task_types=task_types, optimizer_params={"lr": learning_rate, "weight_decay": weight_decay}, adaptive_params={"method": "uwl"}, n_epoch=epoch, earlystop_patience=10, device=device, model_path=save_dir)

        mtl_trainer = MTLTrainer(model, task_types=task_types,
                                 optimizer_params={"lr": self.args.learning_rate,
                                                   "weight_decay": self.args.weight_decay}, n_epoch=self.args.epoch,
                                 earlystop_patience=self.args.earlystop_patience, device=self.args.device,
                                 model_path=self.args.save_dir)
        mtl_trainer.fit(train_dataloader, val_dataloader)
        auc = mtl_trainer.evaluate(mtl_trainer.model, test_dataloader)
        print(f'test auc: {auc}')


class ModelData(object):
    def __init__(self):
        self.dense_feas = []
        self.sparse_feas = []
        self.history_feas = []
        self.target_feas = []
        self.x_train = []
        self.y_train = []
        self.x_val = []
        self.y_val = []
        self.x_test = []
        self.y_test = []
        self.features = []
        self.has_test_val = True
