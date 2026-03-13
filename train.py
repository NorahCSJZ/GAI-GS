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
import datetime
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from scipy.spatial.transform import Rotation
from skimage.metrics import structural_similarity as skimage_ssim

from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from scene.deform_model import DeformModel
from scene.dataloader import amplitude2rssi
from utils.general_utils import safe_state
from utils.generate_camera import generate_new_cam
from utils.loss_utils import l1_loss, l2_loss
from utils.logger import logger_config
from utils.data_painter import paint_spectrum_compare
from arguments import ModelParams, PipelineParams, OptimizationParams
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

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
# Distributed training helpers
# ---------------------------------------------------------------------------

def setup_ddp(rank, world_size):
    """Initialize the DDP process group."""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Destroy the DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def training(dataset, opt, pipe, testing_iterations, saving_iterations,
             checkpoint_iterations, checkpoint, debug_from, dataset_type,
             use_ddp=False):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit("sparse_adam is not installed. "
                 "Run: pip install [3dgs_accel].")

    rank = get_rank()
    world_size = get_world_size()

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = os.path.join('logs', current_time)

    if is_main_process():
        os.makedirs(logdir, exist_ok=True)
        logger = logger_config(log_savepath=os.path.join(logdir, "logger.log"),
                               logging_name='wrf_gs')
        logger.info("logdir: %s | dataset: %s | world_size: %d",
                    logdir, dataset_type, world_size)
    else:
        logger = None

    tb_writer = prepare_output_and_logger(dataset, current_time) if is_main_process() else None

    # ---- Model initialization ----
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    init_num_points = 10000 if dataset_type == "ble" else 200000
    gaussians.gaussian_init(num_points=init_num_points)

    deform = DeformModel()
    if use_ddp:
        deform.deform = DDP(deform.deform, device_ids=[rank],
                            output_device=rank, find_unused_parameters=True)
    deform.train_setting(opt)

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
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training",
                        disable=not is_main_process(), mininterval=10)
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        if use_ddp:
            dist.barrier()

        # GUI viewer hook (optional)
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

        # Increase SH degree every 1000 iterations
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
            n_pts = gaussians.get_xyz.shape[0]
            low_op = (gaussians.get_opacity.squeeze() < opt.min_opacity).sum().item()
            print(f"[iter {iteration}] Gaussians: {n_pts} | "
                  f"low-opacity (<{opt.min_opacity:.3f}): {low_op}")

        # ---- Fetch one training sample ----
        try:
            data_input, data_label = next(scene.train_iter_dataset)
        except StopIteration:
            scene.dataset_init()
            data_input, data_label = next(scene.train_iter_dataset)

        if dataset_type == 'rfid':
            R = torch.from_numpy(Rotation.from_quat(scene.gateway_orientation).as_matrix()).float()
            r_o = scene.r_o
            resolution = (360, 90)
        elif dataset_type == 'ble':
            R = torch.eye(3).float()
            r_o = data_input[0, 3:6]    # gateway position for this sample
            resolution = (36, 9)

        if (iteration - 1) == debug_from:
            pipe.debug = True
        bg = torch.rand(3, device="cuda") if opt.random_background else background

        gt_spectrum = data_label.squeeze(0).cuda()
        viewpoint_cam = generate_new_cam(R, r_o, dataset_type=dataset_type, resolution=resolution)

        N = gaussians.get_xyz.shape[0]
        # RFID: data_input is [1, 3] (tx only).
        # BLE:  data_input is [1, 6] (tx_pos + gw_pos); only tx_pos goes to model.
        time_input = data_input[:, :3].cuda().expand(N, -1)  # [N, 3]
        d_rotation, d_scaling, d_signal = deform.step(
            gaussians.get_xyz.detach(), time_input)

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg,
                            d_rotation, d_scaling, d_signal,
                            use_trained_exp=dataset.train_test_exp,
                            separate_sh=SPARSE_ADAM_AVAILABLE)
        image = render_pkg["render"]                    # [3, H, W]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        if is_main_process() and tb_writer is not None:
            tb_writer.add_image('render/spectrum', image[0].unsqueeze(0), iteration)


        if dataset_type == 'rfid':

            # Treat channel 0 as real part and channel 1 as imaginary part; take magnitude as predicted spectrum.
            pred_spectrum = torch.abs(image[0] + 1j * image[1])  # [H, W]

            Ll1 = l2_loss(pred_spectrum, gt_spectrum)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(pred_spectrum.unsqueeze(0).unsqueeze(0),
                                        gt_spectrum.unsqueeze(0).unsqueeze(0))
            else:
                ssim_value = skimage_ssim(
                    pred_spectrum.detach().cpu().numpy(),
                    gt_spectrum.detach().cpu().numpy(),
                    data_range=1.0, channel_axis=None)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # ---- BLE: predict scalar RSSI (amplitude) ----
        elif dataset_type == 'ble':
            pred_rssi = torch.abs(torch.mean(image[0] + 1j * image[1], dim=(-1,-2)))
            loss = l2_loss(pred_rssi, data_label.squeeze(0).cuda())

        grad_accum_steps = getattr(opt, 'grad_accum_steps', 1)
        (loss / grad_accum_steps).backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.5 * loss.item() + 0.5 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}, PredRSSI: {pred_rssi.item():.7f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if is_main_process() and tb_writer is not None:
                tb_writer.add_scalar('train/loss', loss.item(), iteration)
                tb_writer.add_scalar('train/num_gaussians',
                                     gaussians.get_xyz.shape[0], iteration)

            if (iteration in saving_iterations) and is_main_process():
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            # Gaussian densification and pruning
            # Densification stats accumulate every step regardless of grad_accum_steps.
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
            # Optimizer step every grad_accum_steps iterations.
            if iteration < opt.iterations and iteration % grad_accum_steps == 0:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.step()
                deform.optimizer.zero_grad(set_to_none=True)
                deform.update_learning_rate(iteration)

            if (iteration in checkpoint_iterations) and is_main_process():
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration),
                           os.path.join(scene.model_path, f"chkpnt{iteration}.pth"))

            if iteration in testing_iterations and is_main_process():
                _evaluate(scene, gaussians, deform, logdir, iteration,
                          dataset_type, pipe, bg, logger)
                torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate(scene, gaussians, deform, logdir, iteration,
              dataset_type, pipe, bg, logger):
    """Dispatch evaluation to the dataset-specific function."""
    if logger is not None:
        logger.info("Evaluation at iteration %d", iteration)

    iteration_path = os.path.join(logdir, 'pred_spectrum', str(iteration))
    result_path = os.path.join(logdir, str(iteration))
    os.makedirs(iteration_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)

    if dataset_type == 'rfid':
        _evaluate_rfid(scene, gaussians, deform, pipe, bg,
                       iteration_path, result_path, logger)
    elif dataset_type == 'ble':
        _evaluate_ble(scene, gaussians, deform, pipe, bg,
                      result_path, logger)


def _evaluate_rfid(scene, gaussians, deform, pipe, bg,
                   iteration_path, result_path, logger):
    """RFID evaluation: per-sample pixel error and SSIM."""
    R = torch.from_numpy(
        Rotation.from_quat(scene.gateway_orientation).as_matrix()).float()
    viewpoint_cam = generate_new_cam(R, scene.r_o, resolution=(360, 90), dataset_type='rfid')
    N = gaussians.get_xyz.shape[0]
    all_ssim = []

    for idx, (test_input, test_label) in enumerate(scene.test_iter):
        tx_pos = test_input.cuda()                              # [1, 3]
        time_input = tx_pos.expand(N, -1)
        d_rotation, d_scaling, d_signal = deform.step(
            gaussians.get_xyz.detach(), time_input)

        image = render(viewpoint_cam, gaussians, pipe, bg,
                       d_rotation, d_scaling, d_signal)["render"]

        pred_spectrum = torch.abs(image[0] + 1j * image[1]).detach().cpu().numpy()
        gt_spectrum = test_label.squeeze(0).detach().cpu().numpy()

        pixel_error = np.mean(np.abs(pred_spectrum - gt_spectrum))
        ssim_val = skimage_ssim(pred_spectrum, gt_spectrum,
                                data_range=1.0, channel_axis=None)
        all_ssim.append(ssim_val)

        if logger is not None:
            logger.info("Sample %d | pixel_error=%.6f | SSIM=%.6f",
                        idx, pixel_error, ssim_val)
        paint_spectrum_compare(pred_spectrum, gt_spectrum,
                               save_path=os.path.join(iteration_path, f'{idx}.png'))

    if logger is not None:
        logger.info("Median SSIM: %.6f", np.median(all_ssim))
    np.savetxt(os.path.join(result_path, 'all_ssim.txt'), all_ssim, fmt='%.4f')


def _evaluate_ble(scene, gaussians, deform, pipe, bg, result_path, logger):
    """BLE evaluation: Median AE and RMSE in dBm."""
    R = torch.eye(3).float()
    N = gaussians.get_xyz.shape[0]

    all_errors_db, all_pred_db, all_gt_db = [], [], []

    for test_input, test_label in scene.test_iter:
        gw_pos = test_input[0, 3:6]                             # [3]
        viewpoint_cam = generate_new_cam(R, gw_pos, resolution=(36, 9), dataset_type='ble')

        # Only tx_pos (3-dim) goes to model; gw_pos is used for camera only.
        time_input = test_input[:, :3].cuda().expand(N, -1)     # [N, 3]
        d_rotation, d_scaling, d_signal = deform.step(
            gaussians.get_xyz.detach(), time_input)

        image = render(viewpoint_cam, gaussians, pipe, bg,
                       d_rotation, d_scaling, d_signal)["render"]
        pred_rssi = torch.abs(torch.mean(image[0] + 1j * image[1], dim=(-1,-2)))

        pred_db = amplitude2rssi(pred_rssi.item())
        gt_db = amplitude2rssi(test_label.item())
        all_errors_db.append(abs(pred_db - gt_db))
        all_pred_db.append(pred_db)
        all_gt_db.append(gt_db)

    med = np.median(all_errors_db)
    rmse = np.sqrt(np.mean(np.array(all_errors_db) ** 2))

    if logger is not None:
        logger.info("BLE | Median AE: %.2f dB | RMSE: %.2f dB", med, rmse)

    np.savetxt(os.path.join(result_path, 'ble_errors_db.txt'), all_errors_db, fmt='%.2f')
    np.savetxt(os.path.join(result_path, 'ble_pred_db.txt'), all_pred_db, fmt='%.2f')
    np.savetxt(os.path.join(result_path, 'ble_gt_db.txt'), all_gt_db, fmt='%.2f')
    with open(os.path.join(result_path, 'ble_summary.txt'), 'w') as f:
        f.write(f"Median AE: {med:.2f} dB\n")
        f.write(f"RMSE:      {rmse:.2f} dB\n")
        f.write(f"Samples:   {len(all_errors_db)}\n")


# ---------------------------------------------------------------------------
# Output and logger setup
# ---------------------------------------------------------------------------

def prepare_output_and_logger(args, time):
    if not args.model_path:
        args.model_path = os.path.join("./output/", time)
    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as f:
        f.write(str(Namespace(**vars(args))))
    if TENSORBOARD_FOUND:
        return SummaryWriter(args.model_path)
    print("Tensorboard not available: progress will not be logged.")
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser(description="WRF-GS+ Training Script")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.74")
    parser.add_argument('--port', type=int, default=8686)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--test_iterations', nargs="+", type=int, default=[])
    parser.add_argument('--save_iterations', nargs="+", type=int, default=[])
    parser.add_argument('--checkpoint_iterations', nargs="+", type=int, default=[])
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument('--start_checkpoint', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--dataset_type', type=str, default='rfid',
                        choices=['rfid', 'ble'])
    parser.add_argument('--use_ddp', action='store_true', default=False)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)

    args = parser.parse_args(sys.argv[1:])

    # Dataset-specific iteration schedules (used when not provided via CLI)
    if args.dataset_type == 'rfid':
        if not args.test_iterations:
            args.test_iterations = [10000, 20000, 50000, 100000, 150000, 200000]
        if not args.save_iterations:
            args.save_iterations = [50000, 100000, 150000, 200000]
        if not args.checkpoint_iterations:
            args.checkpoint_iterations = [100000, 200000]
    elif args.dataset_type == 'ble':
        if not args.test_iterations:
            args.test_iterations = [5000, 10000, 50000, 100000, 200000, 500000, 1000000, 1500000, 2000000]
        if not args.save_iterations:
            args.save_iterations = [100000, 500000, 1000000, 1500000, 2000000]
        if not args.checkpoint_iterations:
            args.checkpoint_iterations = [100000, 500000, 1000000, 1500000, 2000000]

    # DDP setup
    if args.use_ddp:
        local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
        world_size = int(os.environ.get('WORLD_SIZE', args.world_size))
        setup_ddp(local_rank, world_size)
    else:
        torch.cuda.set_device(args.gpu)

    safe_state(args.quiet)

    if not args.disable_viewer and not args.use_ddp:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    dataset_params = lp.extract(args)
    opt_params = op.extract(args)
    opt_params = OptimizationParams.apply_dataset_config(opt_params, args.dataset_type)
    pipe_params = pp.extract(args)

    # Always save at the final iteration
    if opt_params.iterations not in args.save_iterations:
        args.save_iterations.append(opt_params.iterations)

    if is_main_process():
        print(f"Dataset: {args.dataset_type} | "
              f"Iterations: {opt_params.iterations} | "
              f"Test at: {args.test_iterations}")

    try:
        training(dataset_params, opt_params, pipe_params,
                 args.test_iterations, args.save_iterations,
                 args.checkpoint_iterations, args.start_checkpoint,
                 args.debug_from, args.dataset_type, args.use_ddp)
    finally:
        if args.use_ddp:
            cleanup_ddp()

    if is_main_process():
        print("\nTraining complete.")
