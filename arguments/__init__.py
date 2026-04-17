#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 0
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    DATASET_CONFIGS = {
        'rfid': {
            'iterations': 200_000,
            'test_iterations': [i for i in range(20000, 200000, 40000)],
            'save_iterations': [100_000, 120_000, 150_000, 170_000, 200_000],
            'checkpoint_iterations': [100_000, 120_000, 150_000, 170_000, 200_000],
            'position_lr_init': 0.00016,
            'position_lr_final': 0.0000016,
            'position_lr_delay_mult': 0.01,
            'position_lr_max_steps': 30_000,
            'deform_lr_max_steps': 200_000,
            'feature_lr': 0.0025,
            'opacity_lr': 0.025,
            'scaling_lr': 0.005,
            'rotation_lr': 0.001,
            'exposure_lr_init': 0.01,
            'exposure_lr_final': 0.001,
            'exposure_lr_delay_steps': 0,
            'exposure_lr_delay_mult': 0.0,
            'percent_dense': 0.01,
            'lambda_dssim': 0.2,
            'lambda_l1': 0.0,
            'lambda_delta': 0.01,
            'densification_interval': 100,
            'opacity_reset_interval': 3_000,
            'densify_from_iter': 500,
            'densify_until_iter': 15_000,
            'densify_grad_threshold': 0.0002,
            'min_opacity': 0.005,
            'max_gaussians': 0,
            'init_num_points': 20000,
            'depth_l1_weight_init': 1.0,
            'depth_l1_weight_final': 0.01,
            'random_background': False,
            'optimizer_type': "default",
        },
    }

    def __init__(self, parser):
        for key, value in self.DATASET_CONFIGS['rfid'].items():
            if not isinstance(value, list):
                setattr(self, key, value)
        super().__init__(parser, "Optimization Parameters", fill_none=True)

    @classmethod
    def apply_dataset_config(cls, opt, dataset_type):
        if dataset_type not in cls.DATASET_CONFIGS:
            return opt
        config = cls.DATASET_CONFIGS[dataset_type]
        for key, value in config.items():
            if isinstance(value, list):
                continue
            if not hasattr(opt, key) or getattr(opt, key) is None:
                setattr(opt, key, value)
        return opt


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v is not None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
