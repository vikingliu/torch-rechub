import torch
from .trainer import Trainer


class CTRTrainer(Trainer):
    """A general trainer for single task learning.

    Args:
        model (nn.Module): any multi task learning model.
        optimizer_fn (torch.optim): optimizer function of pytorch (default = `torch.optim.Adam`).
        optimizer_params (dict): parameters of optimizer_fn.
        scheduler_fn (torch.optim.lr_scheduler) : torch scheduling class, eg. `torch.optim.lr_scheduler.StepLR`.
        scheduler_params (dict): parameters of optimizer scheduler_fn.
        n_epoch (int): epoch number of training.
        earlystop_patience (int): how long to wait after last time validation auc improved (default=10).
        device (str): `"cpu"` or `"cuda:0"`
        gpus (list): id of multi gpu (default=[]). If the length >=1, then the model will wrapped by nn.DataParallel.
        model_path (str): the path you want to save the model (default="./"). Note only save the best weight in the validation data.
    """

    def __init__(
            self,
            model,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=None,
            scheduler_fn=None,
            scheduler_params=None,
            n_epoch=10,
            earlystop_patience=10,
            device="cpu",
            gpus=None,
            model_path="./",
    ):
        super(CTRTrainer, self).__init__(model, 0, optimizer_fn, optimizer_params, scheduler_fn, scheduler_params,
                                         n_epoch, earlystop_patience, device, gpus, model_path)
