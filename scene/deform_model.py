import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.pos_utils import MappingNetwork
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func


class DeformModel:
    def __init__(self):
        self.deform = MappingNetwork(
            D=8,
            W=256,
            input_dims={'pts': 3, 'view': 3, 'tx': 3},
            multires={'pts': 10, 'view': 10, 'tx': 6},
            is_embeded={'pts': True, 'view': True, 'tx': True},
            token_dim=4,
            skips=[4],
            use_view=False,

        ).cuda()
        self.optimizer = None
        self.spatial_lr_scale = 5

    def step(self, xyz, time_emb):
        """
        Returns:
            d_rotation: [N, 4] rotation deformation
            d_scaling:  [N, 3] scaling deformation
            d_signal:   [N, 3] signal for 3DGS rendering
        """
        d_rotation, d_scaling, d_signal = self.deform(xyz, time_emb)
        return d_rotation, d_scaling, d_signal

    def train_setting(self, training_args):
        l = [
            {'params': list(self.deform.parameters()),
             'lr': training_args.position_lr_init * self.spatial_lr_scale,
             "name": "deform"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.deform_lr_max_steps
        )

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "deform/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "deform/iteration_{}/deform.pth".format(loaded_iter))
        self.deform.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
