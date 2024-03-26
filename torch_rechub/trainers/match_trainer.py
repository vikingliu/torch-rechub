import os
import torch
import tqdm

from .trainer import Trainer


class MatchTrainer(Trainer):

    def __init__(
            self,
            model,
            mode=0,
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
        super(MatchTrainer, self).__init__(model, mode, optimizer_fn, optimizer_params, scheduler_fn, scheduler_params,
                                           n_epoch, earlystop_patience, device, gpus, model_path)

    def inference_embedding(self, model, mode, data_loader, model_path):
        # inference
        assert mode in ["user", "item"], "Invalid mode={}.".format(mode)
        model.mode = mode
        model.load_state_dict(torch.load(os.path.join(model_path, "model.pth")))
        model = model.to(self.device)
        model.eval()
        predicts = []
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="%s inference" % (mode), smoothing=0, mininterval=1.0)
            for i, x_dict in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y_pred = model(x_dict)
                predicts.append(y_pred.data)
        return torch.cat(predicts, dim=0)
