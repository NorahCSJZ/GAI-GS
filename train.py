#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
import os
import sys

# Avoid OpenBLAS / MKL over-spawning threads during NumPy import on shared systems.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import torch
from tqdm import tqdm
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation
from skimage.metrics import structural_similarity as skimage_ssim

from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from scene.scene_mapping_network import SceneMappingNetwork
from utils.general_utils import safe_state
from utils.generate_camera import generate_new_cam
from utils.loss_utils import l1_loss
from arguments import ModelParams, PipelineParams, OptimizationParams


try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except ImportError:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except ImportError:
    SPARSE_ADAM_AVAILABLE = False

torch.manual_seed(3407)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def training(dataset, opt, pipe, testing_iterations, checkpoint,
             debug_from, dataset_type):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit("sparse_adam is not installed. "
                 "Run: pip install [3dgs_accel].")

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    gaussians.gaussian_init(num_points=opt.init_num_points)

    smn = SceneMappingNetwork()
    smn.train_setting(opt)

    scene = Scene(dataset, gaussians, dataset_type=dataset_type)
    scene.dataset_init()
    gaussians.training_setup(opt)

    first_iter = 0
    if checkpoint:
        model_params, first_iter = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    ema_loss_for_log = 0.0
    gaussians_count = 0
    low_opacity_count = 0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training",
                        mininterval=10)
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                (custom_cam, do_training, pipe.convert_SHs_python,
                 pipe.compute_cov3D_python, keep_alive,
                 scaling_modifer) = network_gui.receive()
                if custom_cam is not None:
                    net_image = render(
                        custom_cam, gaussians, pipe, background,
                        scaling_modifier=scaling_modifer,
                        use_trained_exp=dataset.train_test_exp,
                        separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, 0, 1) * 255)
                        .byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < opt.iterations) or not keep_alive):
                    break
            except Exception:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
            gaussians_count = gaussians.get_xyz.shape[0]
            low_opacity_count = (
                gaussians.get_opacity.squeeze() < opt.min_opacity
            ).sum().item()

        try:
            data_input, data_label = next(scene.train_iter_dataset)
        except StopIteration:
            scene.dataset_init()
            data_input, data_label = next(scene.train_iter_dataset)

        R = torch.from_numpy(
            Rotation.from_quat(scene.gateway_orientation).as_matrix()
        ).float()
        r_o = scene.r_o
        resolution = (360, 90)

        if (iteration - 1) == debug_from:
            pipe.debug = True
        bg = torch.rand(3, device="cuda") if opt.random_background else background

        gt_spectrum = data_label.squeeze(0).cuda()
        viewpoint_cam = generate_new_cam(R, r_o, resolution=resolution)

        N = gaussians.get_xyz.shape[0]
        time_input = data_input[:, :3].cuda().expand(N, -1)
        d_rotation, d_scaling, d_signal, d_attenuation = smn.step(
            gaussians.get_xyz.detach(), time_input,
            opacities=gaussians.get_opacity.detach().squeeze())

        render_pkg = render(
            viewpoint_cam, gaussians, pipe, bg,
            d_rotation, d_scaling, d_signal, d_attenuation,
            use_trained_exp=dataset.train_test_exp,
            separate_sh=SPARSE_ADAM_AVAILABLE,
        )
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        pred_spectrum = torch.abs(image[0] + 1j * image[1])
        ll1 = l1_loss(pred_spectrum, gt_spectrum)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(
                pred_spectrum.unsqueeze(0).unsqueeze(0),
                gt_spectrum.unsqueeze(0).unsqueeze(0),
            )
        else:
            ssim_value = skimage_ssim(
                pred_spectrum.detach().cpu().numpy(),
                gt_spectrum.detach().cpu().numpy(),
                data_range=1.0,
                channel_axis=None,
            )
        loss = (1.0 - opt.lambda_dssim) * ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        if d_attenuation is not None:
            loss = loss + 0.02 * torch.norm(d_attenuation, p=2)

        lambda_delta = getattr(opt, 'lambda_delta', 0.0)
        if lambda_delta > 0.0 and d_attenuation is not None:
            loss = loss + lambda_delta * torch.mean(d_attenuation ** 2)

        grad_accum_steps = getattr(opt, 'grad_accum_steps', 1)
        (loss / grad_accum_steps).backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.5 * loss.item() + 0.5 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.7f}",
                    "Gaussians": gaussians_count,
                    f"low-op(<{opt.min_opacity:.3f})": low_opacity_count,
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor,
                                                  visibility_filter)
                if (iteration > opt.densify_from_iter
                        and iteration % opt.densification_interval == 0):
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, opt.min_opacity,
                        scene.cameras_extent, size_threshold, radii)
                if (iteration % opt.opacity_reset_interval == 0
                        or (dataset.white_background
                            and iteration == opt.densify_from_iter)):
                    gaussians.reset_opacity()

            if iteration < opt.iterations and iteration % grad_accum_steps == 0:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                smn.optimizer.step()
                smn.optimizer.zero_grad(set_to_none=True)
                smn.update_learning_rate(iteration)

            if iteration in testing_iterations:
                _evaluate(scene, gaussians, smn, iteration, pipe, bg)
                torch.cuda.empty_cache()

    _evaluate(scene, gaussians, smn, opt.iterations, pipe, bg, final=True)
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate(scene, gaussians, smn, iteration, pipe, bg, final=False):
    R = torch.from_numpy(
        Rotation.from_quat(scene.gateway_orientation).as_matrix()
    ).float()
    viewpoint_cam = generate_new_cam(R, scene.r_o, resolution=(360, 90))
    N = gaussians.get_xyz.shape[0]
    all_ssim = []
    low_std_count = 0

    for test_input, test_label in scene.test_iter:
        tx_pos = test_input.cuda()
        time_input = tx_pos.expand(N, -1)
        d_rotation, d_scaling, d_signal, d_attenuation = smn.step(
            gaussians.get_xyz.detach(), time_input,
            opacities=gaussians.get_opacity.detach().squeeze())

        image = render(viewpoint_cam, gaussians, pipe, bg,
                       d_rotation, d_scaling, d_signal, d_attenuation)["render"]

        pred_spectrum = torch.abs(image[0] + 1j * image[1]).detach().cpu().numpy()
        gt_spectrum = test_label.squeeze(0).detach().cpu().numpy()

        if np.std(pred_spectrum) < 0.1:
            low_std_count += 1

        ssim_val = skimage_ssim(pred_spectrum, gt_spectrum,
                                data_range=1.0, channel_axis=None)
        all_ssim.append(ssim_val)

    total_test = len(all_ssim)
    low_std_ratio = low_std_count / total_test if total_test > 0 else 0.0
    median_ssim = float(np.median(all_ssim)) if all_ssim else float("nan")
    print(
        f"[Eval {iteration}] Median SSIM: {median_ssim:.6f} | "
        f"Low-std images: {low_std_count}/{total_test} ({low_std_ratio:.1%})"
    )

    if final:
        ssim_arr = np.array(all_ssim)
        cdf_fractions = np.arange(0.0, 1.01, 0.1)
        ssim_at_cdf = np.percentile(ssim_arr, cdf_fractions * 100)
        lines = [f"Final evaluation at iteration {iteration}:"]
        lines.append("CDF-SSIM (fraction of test set -> SSIM threshold):")
        for p, s in zip(cdf_fractions, ssim_at_cdf):
            lines.append(f"  CDF={p:.1f} -> SSIM={s:.4f}")
        print("\n".join(lines))

        fractions = np.arange(0.0, 1.01, 0.1)
        de_lines = ["Data efficiency curve (fraction of test set -> mean SSIM):"]
        for p in fractions:
            n = int(len(ssim_arr) * p)
            mean_ssim = float(np.mean(ssim_arr[:n])) if n > 0 else float('nan')
            if n > 0:
                de_lines.append(f"  {p:.1f} -> mean SSIM={mean_ssim:.4f}")
            else:
                de_lines.append(f"  {p:.1f} -> mean SSIM=NaN")
        print("\n".join(de_lines))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser(description="GAI-GS RFID Training Script")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.106")
    parser.add_argument('--port', type=int, default=5466)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--test_iterations', nargs="+", type=int, default=[])
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument('--start_checkpoint', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset_type', type=str, default='rfid',
                        choices=['rfid'])
    args = parser.parse_args(sys.argv[1:])

    from arguments import OptimizationParams as _OP
    _cfg = _OP.DATASET_CONFIGS.get(args.dataset_type, {})
    if not args.test_iterations:
        args.test_iterations = _cfg.get('test_iterations', [])

    torch.cuda.set_device(args.gpu)
    safe_state(args.quiet)

    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    dataset_params = lp.extract(args)
    opt_params = op.extract(args)
    opt_params = OptimizationParams.apply_dataset_config(opt_params, args.dataset_type)
    pipe_params = pp.extract(args)

    print(f"Dataset: {args.dataset_type} | "
          f"Iterations: {opt_params.iterations} | "
          f"grad_accum: {getattr(opt_params, 'grad_accum_steps', 1)} | "
          f"Test at: {args.test_iterations}")

    training(dataset_params, opt_params, pipe_params,
             args.test_iterations, args.start_checkpoint,
             args.debug_from, args.dataset_type)

    print("\nTraining complete.")
