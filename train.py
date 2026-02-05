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
import datetime
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, sig2mse
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
from utils.generate_camera import generate_new_cam

import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    from tensorboardX import SummaryWriter  

os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['NCCL_BLOCKING_WAIT'] = '1'

TENSORBOARD_FOUND = True

# except ImportError:
#     TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False
from utils.logger import logger_config 
from scipy.spatial.transform import Rotation
from utils.data_painter import paint_spectrum_compare 
from skimage.metrics import structural_similarity as ssim
from scene.mapping_network import MappingNetwork, raw_to_signal, create_mapping_network


def amplitude2rssi(amplitude):
    """convert amplitude to rssi
    """
    return -100 * (1 - amplitude)


def safe_barrier(timeout=None):
    """
    Safe barrier function, compatible with PyTorch versions without timeout parameter
    """
    try:
        if timeout is not None:
            # Try using version with timeout (PyTorch 1.14+)
            dist.barrier(timeout=timeout)
        else:
            dist.barrier()
    except TypeError:
        # If timeout parameter not supported, use version without it
        dist.barrier()


def training(dataset, opt, pipe, testing_iterations, checkpoint_iterations, checkpoint, debug_from, model_weights=None, rank=0, world_size=1, ddp=False, data_type='rfid'):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    # Initialize DDP
    if ddp:
        torch.cuda.set_device(rank)
        # Set NCCL timeout and debug options
        os.environ.setdefault('NCCL_TIMEOUT', '1800')  # 30 minutes
        os.environ.setdefault('NCCL_BLOCKING_WAIT', '1')  # Blocking wait for debugging
        # Set NCCL debug level (optional, helps diagnose issues)
        if 'NCCL_DEBUG' not in os.environ:
            os.environ['NCCL_DEBUG'] = 'INFO'  # Set to INFO level for more output
        
        # Initialize process group
        try:
            print(f"[Rank {rank}] Initializing process group...")
            dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=1800))
            print(f"[Rank {rank}] Process group initialized successfully")
        except Exception as e:
            print(f"[Rank {rank}] Process group initialization failed: {e}")
            raise
        
        device = torch.device(f'cuda:{rank}')
        
        # Verify process group is correctly initialized
        if dist.is_initialized():
            print(f"[Rank {rank}] Process group verified, world_size={dist.get_world_size()}, rank={dist.get_rank()}")
    else:
        device = torch.device('cuda')
    
    datadir = 'data'
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = os.path.join('logs', current_time)
    log_filename = "logger.log"
    log_savepath = os.path.join(logdir, log_filename)
    
    # Only create logs and tensorboard on rank 0
    if ddp:
        print(f"[Rank {rank}] Preparing to create logs and tensorboard...")
    if not ddp or rank == 0:
        os.makedirs(logdir,exist_ok=True)
        logger = logger_config(log_savepath=log_savepath, logging_name='gsss')
        logger.info("datadir:%s, logdir:%s", datadir, logdir)
        print(f"[Rank {rank}] Preparing output and logger...")
        tb_writer = prepare_output_and_logger(dataset,current_time)
        print(f"[Rank {rank}] Output and logger preparation complete")
    else:
        logger = None
        tb_writer = None
    
    # Sync point: ensure all processes complete logger initialization
    if ddp:
        try:
            safe_barrier(timeout=datetime.timedelta(seconds=60))
            if rank == 0:
                print("All processes completed logger initialization")
        except Exception as e:
            print(f"[Rank {rank}] Warning: Sync failed after logger init: {e}")
    
    first_iter = 0
    if ddp:
        print(f"[Rank {rank}] Creating GaussianModel...")
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type, data_type=data_type)
    gaussians.gaussian_init()
    if ddp:
        print(f"[Rank {rank}] GaussianModel creation complete")
    
    # ============ Create MappingNetwork (independent of GaussianModel) ============
    if ddp:
        print(f"[Rank {rank}] Creating MappingNetwork...")
    mapping_network = create_mapping_network(data_type=data_type, device=device)
    if ddp:
        print(f"[Rank {rank}] MappingNetwork creation complete")
    
    # Move network model to correct device
    if ddp:
        print(f"[Rank {rank}] Moving model to device {device}...")
    # Note: network_fn in GaussianModel still exists (for compatibility), but training uses mapping_network
    if hasattr(gaussians, 'network_fn'):
        gaussians.network_fn = gaussians.network_fn.to(device)
    if hasattr(gaussians, 'render'):
        gaussians.render = gaussians.render.to(device)
    if ddp:
        print(f"[Rank {rank}] Model moved to device")
    
    # Sync point: ensure all processes complete model creation and movement
    if ddp:
        try:
            safe_barrier(timeout=datetime.timedelta(seconds=60))
            if rank == 0:
                print("All processes completed model creation and movement")
        except Exception as e:
            print(f"[Rank {rank}] Warning: Sync failed after model creation: {e}")
    
    # Wrap network model with DDP
    if ddp:
        print(f"[Rank {rank}] Wrapping DDP model...")
        # Wrap mapping_network
        if not isinstance(mapping_network, DDP):
            print(f"[Rank {rank}] Wrapping mapping_network as DDP...")
            mapping_network = DDP(mapping_network, device_ids=[rank], output_device=rank, 
                                 find_unused_parameters=True, broadcast_buffers=True)
            print(f"[Rank {rank}] mapping_network wrapped as DDP")
        print(f"[Rank {rank}] DDP model wrapping complete")
        
        # Sync point: ensure all processes complete DDP wrapping
        try:
            safe_barrier(timeout=datetime.timedelta(seconds=120))
            if rank == 0:
                print("All processes completed DDP model wrapping")
        except Exception as e:
            print(f"[Rank {rank}] Warning: Sync failed after DDP wrapping: {e}")
    
    print("Creating Scene...")
    scene = Scene(dataset, gaussians, ddp=ddp, rank=rank, world_size=world_size, data_type=data_type)
    print("Scene created, initializing dataset...")
    scene.dataset_init()
    print("Dataset initialization complete")
    
    # Sync point: ensure all processes complete data loading
    if ddp:
        try:
            safe_barrier(timeout=datetime.timedelta(seconds=120))
            if rank == 0:
                print("All processes completed data loading")
        except Exception as e:
            print(f"[Rank {rank}] Warning: Sync failed after data loading: {e}")
    
    print("Setting up training parameters...")
    gaussians.training_setup(opt)
    print("Training parameters setup complete")
    
    # ============ MappingNetwork Optimizer & Scheduler ============
    import math
    mapping_lr = 7e-4  # Learning rate
    mapping_optimizer = torch.optim.AdamW(mapping_network.parameters(), lr=mapping_lr, weight_decay=1e-5)
    
    # Cosine scheduler with warmup
    total_iters = opt.iterations
    warmup_iters = int(total_iters * 0.05)
    def mapping_lr_lambda(step):
        if step < warmup_iters:
            return 0.01 + 0.99 * step / warmup_iters
        else:
            progress = (step - warmup_iters) / max(1, total_iters - warmup_iters)
            return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))
    mapping_scheduler = torch.optim.lr_scheduler.LambdaLR(mapping_optimizer, lr_lambda=mapping_lr_lambda)
    print(f"MappingNetwork optimizer created, lr={mapping_lr}")
    
    # Load model weights (mapping_network)
    if model_weights is not None:
        if ddp:
            print(f"[Rank {rank}] Loading model weights...")
        map_location = {'cuda:%d' % 0: f'cuda:{rank}'} if ddp else None
        weights = torch.load(model_weights, map_location=map_location)
        # Handle DDP wrapper: if DDP, load to .module
        mapping_network_target = mapping_network.module if isinstance(mapping_network, DDP) else mapping_network
        if 'mapping_network' in weights:
            mapping_network_target.load_state_dict(weights['mapping_network'])
        elif 'network_fn' in weights:  # Backward compatible with old weights
            mapping_network_target.load_state_dict(weights['network_fn'])
        if not ddp or rank == 0:
            print(f"MappingNetwork weights loaded from {model_weights}")
        if ddp:
            try:
                safe_barrier(timeout=datetime.timedelta(seconds=120))
            except Exception as e:
                print(f"[Rank {rank}] Warning: Sync failed after model weights loading: {e}")
    
    if checkpoint:
        if ddp:
            print(f"[Rank {rank}] Loading checkpoint...")
        map_location = {'cuda:%d' % 0: f'cuda:{rank}'} if ddp else None
        (model_params, first_iter) = torch.load(checkpoint, map_location=map_location)
        gaussians.restore(model_params, opt)
        if ddp:
            print(f"[Rank {rank}] Checkpoint loaded")
            
        # Sync point: ensure all processes complete checkpoint loading
        if ddp:
            try:
                safe_barrier(timeout=datetime.timedelta(seconds=120))
                if rank == 0:
                    print("All processes completed checkpoint loading")
            except Exception as e:
                print(f"[Rank {rank}] Warning: Sync failed after checkpoint loading: {e}")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = None
    
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    
    # Sync all processes before training starts
    if ddp:
        try:
            safe_barrier(timeout=datetime.timedelta(seconds=120))
            if rank == 0:
                print("All processes synced, starting training...")
        except Exception as e:
            print(f"[Rank {rank}] Warning: Pre-training sync failed: {e}")
            # Continue training as this is not fatal
    
    # Only show progress bar on rank 0
    if not ddp or rank == 0:
        progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    else:
        progress_bar = None
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        # network_gui only runs on rank 0
        if (not ddp or rank == 0) and network_gui.conn == None:
            network_gui.try_connect()
        if (not ddp or rank == 0):
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                    if custom_cam != None:
                        net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, dataset.source_path)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None

        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 300 == 0:
            gaussians.oneupSHdegree()
        if iteration % 300 == 0 and (not ddp or rank == 0):
            print("nums of gaussians:", gaussians.get_xyz.shape[0])

        # Pick a random Camera
        try:
            tx_pos, spectrum = next(scene.train_iter_dataset)

        except (StopIteration, RuntimeError):
            # Set sampler epoch at the start of each epoch
            # Fix: use more accurate epoch calculation
            if hasattr(scene, 'train_set') and len(scene.train_set) > 0:
                # Calculate current epoch (based on completed iterations)
                samples_per_epoch = len(scene.train_set)
                if ddp and hasattr(scene, 'train_sampler') and scene.train_sampler is not None:
                    # In DDP mode, each process only handles part of the data
                    samples_per_epoch = len(scene.train_sampler)
                epoch = (iteration - 1) // max(1, samples_per_epoch)
            else:
                epoch = 0
            
            # In DDP mode, ensure all processes reinitialize dataset simultaneously
            if ddp:
                try:
                    safe_barrier(timeout=datetime.timedelta(seconds=30))
                except Exception as e:
                    if rank == 0:
                        print(f"Warning: Barrier timeout during dataset reinit at iteration {iteration}: {e}")
            
            scene.dataset_init(epoch=epoch)
            
        
        B = spectrum.shape[0] 
        viewpoint_cam = None
        
        # RFID camera setup
        r_o = scene.r_o
        gateway_orientation = scene.gateway_orientation 
        R = torch.from_numpy(Rotation.from_quat(gateway_orientation).as_matrix()).float()
        tx_pos = tx_pos.to(device)
        viewpoint_cam = generate_new_cam(R, r_o)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device=device) if opt.random_background else background
        
        # Ensure spectrum and tx_pos are in batch form (RFID)
        if spectrum.dim() == 2:
            spectrum = spectrum.unsqueeze(0)
        if tx_pos.dim() == 1:
            tx_pos = tx_pos.unsqueeze(0)
        
        # Single sample processing
        gt_image = spectrum[0].to(device)
        tx_pos_sample = tx_pos[0] if tx_pos.dim() > 1 else tx_pos
        tx_pos_sample = tx_pos_sample.to(device)

        # ============ Step 1: Zero gradients ============
        gaussians.optimizer.zero_grad(set_to_none=True)
        gaussians.exposure_optimizer.zero_grad(set_to_none=True)
        mapping_optimizer.zero_grad(set_to_none=True)
        # ============ Step 2: Forward pass (MappingNetwork -> Render) ============
        pts = gaussians.get_xyz  # [N, 3]
        N = pts.size(0)
        
        # RFID: tx_pos_sample is position [3]
        view = pts - tx_pos_sample  # [N, 3]
        if tx_pos_sample.dim() == 1:
            tx_pos_sample = tx_pos_sample.unsqueeze(0)
        tx_condition = tx_pos_sample.expand(N, -1)  # [N, 3]
        
        # MappingNetwork forward
        raw = mapping_network(pts, tx_condition, view)  # [N, 4]
        signal = raw_to_signal(raw, pts, r_o)  # [N, 3]
        
        # Retain gradients for debugging (only when needed)
        if iteration % 100 == 0:
            raw.retain_grad()
            signal.retain_grad()
        
        # Render with computed signal as override_features
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, 
                           use_trained_exp=dataset.train_test_exp, 
                           separate_sh=SPARSE_ADAM_AVAILABLE,
                           override_features=signal)
        
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        
        # RFID: extract prediction from rendered image
        channel, height, width = image.shape
        pred = image[0, :height, :]
        
        # ============ Step 3: Compute loss ============
        Ll1 = l1_loss(pred, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(pred.unsqueeze(0).unsqueeze(0), gt_image.unsqueeze(0).unsqueeze(0))
        else:
            ssim_value = ssim(pred, gt_image)
        loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)) 
        
        # ============ Step 4: Backward pass ============
        loss.backward()
        
        # Debug: check gradient flow (must be before optimizer.step()!)
        if iteration % 100 == 0:
            mapping_net_model = mapping_network.module if isinstance(mapping_network, DDP) else mapping_network
            mapping_grad_norm = sum(p.grad.norm().item() for p in mapping_net_model.parameters() if p.grad is not None)
            num_params_with_grad = sum(1 for p in mapping_net_model.parameters() if p.grad is not None)
            total_params = sum(1 for p in mapping_net_model.parameters())
            
            # Check signal gradient (requires retain_grad)
            signal_grad_norm = signal.grad.norm().item() if hasattr(signal, 'grad') and signal.grad is not None else 0.0
            
            # Check raw gradient
            raw_grad_norm = raw.grad.norm().item() if hasattr(raw, 'grad') and raw.grad is not None else 0.0
            
            # Check image gradient
            image_grad_norm = image.grad.norm().item() if hasattr(image, 'grad') and image.grad is not None else 0.0
            
            print(f"[ITER {iteration}] Gradient Chain Check (BEFORE optimizer.step):")
            print(f"  loss: {loss.item():.6f}")
            print(f"  image.grad: {image_grad_norm:.6f}")
            print(f"  signal.grad: {signal_grad_norm:.6f}")
            print(f"  raw.grad: {raw_grad_norm:.6f}")
            print(f"  mapping_network.grad: {mapping_grad_norm:.6f} ({num_params_with_grad}/{total_params} params)")
            
            if mapping_grad_norm < 1e-8:
                print(f"  ⚠️  WARNING: mapping_network gradient is near zero!")
                if signal_grad_norm < 1e-8:
                    print(f"  ❌ signal.grad is also zero - rasterizer may not propagate gradients to colors_precomp!")
                elif raw_grad_norm < 1e-8:
                    print(f"  ❌ raw.grad is zero - raw_to_signal may have gradient flow issue!")
                else:
                    print(f"  ✓ signal and raw have gradients, but network doesn't - check network forward pass")
        
        # Visualization (RFID)
        render_image_show = pred.reshape(1, 90, 360).to(device)
        
        if tb_writer is not None and render_image_show is not None:
            tb_writer.add_image('render-img', render_image_show, iteration)

        # Depth regularization
        Ll1depth_pure = 0.0
        
        Ll1depth = 0

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0 and progress_bar is not None:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations and progress_bar is not None:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss', loss.item(), iteration)            
                tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            
            # Save model weights every 2000 iterations
            if iteration > 0 and iteration % 2000 == 0:
                if not ddp or rank == 0:
                    # Create save directory model_weight/{data_type}/
                    weight_dir = os.path.join("model_weight", data_type)
                    os.makedirs(weight_dir, exist_ok=True)
                    
                    # Save model weights (handle DDP wrapper)
                    weight_path = os.path.join(weight_dir, f"model_iter_{iteration}.pth")
                    # Get actual model (if DDP wrapped, use .module)
                    mapping_net_model = mapping_network.module if isinstance(mapping_network, DDP) else mapping_network
                    save_dict = {
                        'mapping_network': mapping_net_model.state_dict()
                    }
                    torch.save(save_dict, weight_path)
                    print(f"\n[ITER {iteration}] Model weights saved to {weight_path}")
                    
                    # Keep at most 10 files, delete oldest
                    weight_files = sorted(
                        [f for f in os.listdir(weight_dir) if f.startswith("model_iter_") and f.endswith(".pth")],
                        key=lambda x: int(x.split("_")[-1].replace(".pth", ""))
                    )
                    while len(weight_files) > 10:
                        oldest_file = weight_files.pop(0)
                        oldest_path = os.path.join(weight_dir, oldest_file)
                        os.remove(oldest_path)
                        print(f"  Deleted old weight file: {oldest_file}")
                
                # DDP sync
                if ddp:
                    try:
                        safe_barrier(timeout=datetime.timedelta(seconds=120))
                    except Exception as e:
                        print(f"[Rank {rank}] Warning: Save barrier timeout at iteration {iteration}: {e}")

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)


                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # # Log stats to help tune densify_grad_threshold and min_opacity
                    # with torch.no_grad():
                    #     grads = (gaussians.xyz_gradient_accum / gaussians.denom).nan_to_num()
                    #     if rank == 0:
                    #         print(f"\n[ITER {iteration}] Grad Stats: max={grads.max().item():.6f}, mean={grads.mean().item():.6f} (threshold: {opt.densify_grad_threshold})")
                    #         print(f"           Opacity Stats: max={gaussians.get_opacity.max().item():.4f}, mean={gaussians.get_opacity.mean().item():.4f}")
                    #     if tb_writer:
                    #         tb_writer.add_scalar('stats/max_grad', grads.max().item(), iteration)
                    #         tb_writer.add_scalar('stats/mean_grad', grads.mean().item(), iteration)

                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    min_opacity = 0.005
                    gaussians.densify_and_prune(opt.densify_grad_threshold, min_opacity, scene.cameras_extent, size_threshold, radii)
                
                # Reset opacity
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # ============ Step 5: Optimizer step ============
            if iteration < opt.iterations:
                # DDP barrier
                if ddp:
                    try:
                        safe_barrier(timeout=datetime.timedelta(seconds=60))
                    except Exception as e:
                        print(f"[Rank {rank}] Warning: Pre-optimizer barrier timeout at iteration {iteration}: {e}")
                
        
                # Gaussian optimizer step + scheduler
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                else:
                    gaussians.optimizer.step()
                gaussians.gaussian_scheduler.step()
                
                # Exposure optimizer step + scheduler
                gaussians.exposure_optimizer.step()
                gaussians.exposure_scheduler.step()
                
                # MappingNetwork optimizer step + scheduler
                mapping_optimizer.step()
                mapping_scheduler.step()
            


            if (iteration in checkpoint_iterations):
                # DDP mode requires syncing all processes
                if ddp:
                    try:
                        safe_barrier(timeout=datetime.timedelta(seconds=120))
                    except Exception as e:
                        print(f"[Rank {rank}] Warning: Checkpoint barrier timeout at iteration {iteration}: {e}")
                if not ddp or rank == 0:
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    # If using DDP, extract model from DDP wrapper
                    if ddp:
                        checkpoint_data = gaussians.capture()
                    else:
                        checkpoint_data = gaussians.capture()
                    torch.save((checkpoint_data, iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                # Sync after checkpoint save
                if ddp:
                    try:
                        safe_barrier(timeout=datetime.timedelta(seconds=120))
                    except Exception as e:
                        print(f"[Rank {rank}] Warning: Post-checkpoint barrier timeout at iteration {iteration}: {e}")
           
            if iteration in testing_iterations:
                # DDP mode requires syncing all processes to reach evaluation phase
                if ddp:
                    try:
                        # Sync first, ensure all processes reach here
                        safe_barrier(timeout=datetime.timedelta(seconds=120))
                    except Exception as e:
                        print(f"[Rank {rank}] Warning: Pre-evaluation barrier failed at iteration {iteration}: {e}")
                        # If barrier fails, skip evaluation and continue training
                        continue
                
                # Evaluation only runs on rank 0, other processes wait for sync
                eval_success = True
                if not ddp or rank == 0:
                    # Save DDP model reference for recovery
                    original_network_fn = None
                    
                    try:
                        torch.cuda.empty_cache()
                        
                        if logger is not None:
                            logger.info("Start evaluation")
                        iteration_path = os.path.join(logdir, 'pred_spectrum', str(iteration))
                        os.makedirs(iteration_path, exist_ok=True) 
                        full_path = os.path.join(logdir, str(iteration))
                        os.makedirs(full_path, exist_ok=True)
                        save_img_idx = 0
                        all_ssim = []
                        all_pixel_error = []
                        
                        # Disable gradient computation during evaluation to avoid DDP sync issues
                        # Get underlying model, bypass DDP wrapper
                        if ddp:
                            # Temporarily use underlying model to avoid DDP sync
                            original_mapping_network = mapping_network
                            mapping_network_eval = original_mapping_network.module
                        else:
                            mapping_network_eval = mapping_network
                            original_mapping_network = None
                        
                        with torch.no_grad():
                            for test_input, test_label in scene.test_iter: 
                                # Process batch test data (collate_fn ensures batch form)
                                if test_input.dim() == 2:
                                    test_input = test_input.unsqueeze(0)
                                if test_label.dim() == 1:
                                    test_label = test_label.unsqueeze(0)
                                
                                B_test = test_input.shape[0]
                                
                                r_o = scene.r_o
                                
                                # RFID evaluation
                                gateway_orientation = scene.gateway_orientation 
                                R = torch.from_numpy(Rotation.from_quat(gateway_orientation).as_matrix()).float()
                                tx_pos = test_label.to(device)
                                viewpoint_cam = generate_new_cam(R, r_o)
                                
                                # Use MappingNetwork to compute signal (eval mode)
                                pts = gaussians.get_xyz
                                N = pts.size(0)
                                view = pts - tx_pos
                                tx_condition = tx_pos.unsqueeze(0).expand(N, -1) if tx_pos.dim() == 1 else tx_pos.expand(N, -1)
                                raw = mapping_network_eval(pts, tx_condition, view)
                                signal = raw_to_signal(raw, pts, r_o)
                                
                                render_pkg = render(viewpoint_cam, gaussians, pipe, bg, override_features=signal)
                                image = render_pkg["render"]
                                channel, height, width = image.shape
                                pred_spectrum = image[0, :height, :]

                                pred_spectrum = pred_spectrum.detach().cpu().numpy()
                                gt_spectrum = test_input.squeeze(0).detach().cpu().numpy()
                                
                                ssim_i = ssim(pred_spectrum, gt_spectrum, data_range=1, multichannel=False)
                                
                                if logger is not None:
                                    pixel_error = np.mean(abs(pred_spectrum - gt_spectrum))
                                    logger.info(
                                        "Spectrum {:d}, Mean pixel error = {:.6f}; SSIM = {:.6f}".format(save_img_idx, pixel_error, ssim_i))
                                paint_spectrum_compare(pred_spectrum, gt_spectrum,
                                                    save_path=os.path.join(iteration_path, f'{save_img_idx}.png'))
                                
                                pred_np = np.asarray(pred_spectrum)
                                gt_np = np.asarray(gt_spectrum)
                                pixel_error = float(np.mean(np.abs(pred_np - gt_np)))
                                all_ssim.append(ssim_i)
                                all_pixel_error.append(pixel_error)
                                save_img_idx += 1
                        
                        # Output summary results after loop
                        if logger is not None:
                                logger.info("Median SSIM is {:.6f}; Median pixel error is {:.6f}".format(np.median(all_ssim), np.median(all_pixel_error)))
                        
                        # Save RFID results
                        np.savetxt(os.path.join(full_path, 'all_ssim.txt'), all_ssim, fmt='%.4f')
                        
                        if logger is not None:
                            logger.info(f"Iteration {iteration} evaluation completed")
                        
                        # Restore DDP model
                        if ddp:
                            gaussians.network_fn = original_network_fn
                        
                        torch.cuda.empty_cache()
                    except Exception as e:
                        # If evaluation fails, log error but don't interrupt training
                        eval_success = False
                        # Ensure DDP model is restored (if it was replaced)
                        if ddp and original_network_fn is not None:
                            gaussians.network_fn = original_network_fn
                        if logger is not None:
                            logger.error(f"Evaluation failed at iteration {iteration}: {e}")
                        import traceback
                        traceback.print_exc()
                        torch.cuda.empty_cache()
                
                # Sync all processes after evaluation (regardless of success)
                # This ensures all processes can reach here, avoiding deadlock
                if ddp:
                    try:
                        # Set longer timeout since evaluation may take longer
                        # But if evaluation fails, other processes shouldn't wait indefinitely
                        safe_barrier(timeout=datetime.timedelta(seconds=600))
                    except Exception as e:
                        print(f"[Rank {rank}] Warning: Post-evaluation barrier failed at iteration {iteration}: {e}")
                        # If barrier fails, log but continue training (avoid deadlock)
                        # This usually means a process errored or timed out during evaluation
                        pass
    
    # Cleanup DDP
    if ddp:
        dist.destroy_process_group()


def prepare_output_and_logger(args,time):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", time)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def main_worker(rank, world_size, full_args):
    """Main function for DDP worker process"""
    # Get rank and world_size from environment variables or parameters
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        ddp = True
    else:
        ddp = full_args.use_ddp if hasattr(full_args, 'use_ddp') else False
        if not ddp:
            rank = 0
            world_size = 1
    
    if ddp:
        # Set environment variables (if running under torchrun, these are already set)
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = full_args.master_addr if hasattr(full_args, 'master_addr') else 'localhost'
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = full_args.master_port if hasattr(full_args, 'master_port') else '12355'
        
        # In DDP mode, rank is automatically mapped to GPU in CUDA_VISIBLE_DEVICES
        # e.g., if CUDA_VISIBLE_DEVICES=2,3,5,7, rank 0 uses device 0 (physical GPU 2), rank 1 uses device 1 (physical GPU 3), etc.
        torch.cuda.set_device(rank)
        device_id = torch.cuda.current_device()
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        
        # Parse actual physical GPU ID
        if cuda_visible:
            physical_gpu_ids = [int(x.strip()) for x in cuda_visible.split(',')]
            if device_id < len(physical_gpu_ids):
                physical_gpu_id = physical_gpu_ids[device_id]
                print(f"[Process {rank}/{world_size-1}] Using device index {device_id} (physical GPU {physical_gpu_id}, CUDA_VISIBLE_DEVICES={cuda_visible})")
            else:
                print(f"[Process {rank}/{world_size-1}] Using device index {device_id} (CUDA_VISIBLE_DEVICES={cuda_visible})")
        else:
            print(f"[Process {rank}/{world_size-1}] Using device index {device_id}")
    else:
        # Single GPU mode: if CUDA_VISIBLE_DEVICES is set, use device 0; otherwise use specified gpu
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        if cuda_visible:
            # CUDA_VISIBLE_DEVICES is set, use logical device 0 (first in physical device list)
            torch.cuda.set_device(0)
            device_id = 0
            physical_gpu_ids = [int(x.strip()) for x in cuda_visible.split(',')]
            physical_gpu_id = physical_gpu_ids[0] if physical_gpu_ids else 0
            print(f"[Single GPU mode] Using device index {device_id} (physical GPU {physical_gpu_id}, CUDA_VISIBLE_DEVICES={cuda_visible})")
        else:
            # CUDA_VISIBLE_DEVICES not set, use specified GPU
            torch.cuda.set_device(full_args.gpu)
            device_id = torch.cuda.current_device()
            print(f"[Single GPU mode] Using GPU {device_id}")
    
    # Parse parameters
    lp = ModelParams(ArgumentParser())
    op = OptimizationParams(ArgumentParser())
    pp = PipelineParams(ArgumentParser())
    
    if rank == 0:
        print("Optimizing " + full_args.model_path)
    
    # Initialize system state (RNG)
    safe_state(full_args.quiet)
    
    # Start GUI server (only on rank 0)
    if not full_args.disable_viewer and (not ddp or rank == 0):
        network_gui.init(full_args.ip, full_args.port)
    
    torch.autograd.set_detect_anomaly(full_args.detect_anomaly)
    # Get data_type parameter (RFID only)
    data_type = getattr(full_args, 'data_type', 'rfid')
    training(lp.extract(full_args), op.extract(full_args), pp.extract(full_args), 
             full_args.test_iterations, full_args.checkpoint_iterations, 
             full_args.start_checkpoint, full_args.debug_from, 
             model_weights=full_args.model_weights,
             rank=rank, world_size=world_size, ddp=ddp, data_type=data_type)
    
    if rank == 0:
        print("\nTraining complete.")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=8686)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[20000, 40000, 60000, 80000, 100000])
    parser.add_argument("--start_checkpoint", type=str, default=None, help="Full checkpoint path (contains Gaussian parameters)")
    parser.add_argument("--model_weights", type=str, default=None, help="Model weights path (network_fn)")
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID for single GPU training')
    # DDP related parameters
    parser.add_argument('--use_ddp', action='store_true', default=False, help='Use distributed data parallel training')
    parser.add_argument('--world_size', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--master_addr', type=str, default='localhost', help='Master node address')
    parser.add_argument('--master_port', type=str, default='12306', help='Master node port')
    parser.add_argument('--gpus', type=str, default=None, help='GPU list to use, comma separated, e.g.: "0,1,2,3" or "0,2,4,6"')
    parser.add_argument('--data_type', type=str, default='rfid', help='Data type')
    
    args = parser.parse_args(sys.argv[1:])
    
    # Handle GPU selection
    if args.gpus is not None:
        # Parse GPU list
        gpu_list = [int(gpu_id.strip()) for gpu_id in args.gpus.split(',')]
        gpu_list_str = ','.join(map(str, gpu_list))
        
        # If user explicitly specified --gpus, use command line arg and override env variable
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            old_value = os.environ['CUDA_VISIBLE_DEVICES']
            print(f"Detected existing CUDA_VISIBLE_DEVICES={old_value}, will be overridden by --gpus {gpu_list_str}")
        
        # Set CUDA_VISIBLE_DEVICES to command line specified GPU list
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list_str
        print(f"Set CUDA_VISIBLE_DEVICES={gpu_list_str} (physical GPUs: {gpu_list})")
        
        # If using DDP, update world_size to GPU count
        if args.use_ddp and args.world_size == 1:
            args.world_size = len(gpu_list)
            print(f"Auto-set world_size={args.world_size} (based on specified GPU count)")
    
    # Check if using torchrun or environment variable launch
    use_ddp = args.use_ddp or ('RANK' in os.environ and 'WORLD_SIZE' in os.environ)
    
    if use_ddp:
        # Using torchrun launch, directly call main_worker
        if 'RANK' in os.environ:
            # Launched via torchrun
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            # If GPU list specified, verify count matches
            if args.gpus is not None:
                gpu_list = [int(gpu_id.strip()) for gpu_id in args.gpus.split(',')]
                if world_size != len(gpu_list):
                    print(f"Warning: torchrun process count ({world_size}) doesn't match GPU list count ({len(gpu_list)})")
            main_worker(rank, world_size, args)
        else:
            # Manually start multi-process
            # If GPU list specified, use GPU count as process count
            if args.gpus is not None:
                gpu_list = [int(gpu_id.strip()) for gpu_id in args.gpus.split(',')]
                if args.world_size == 1:
                    args.world_size = len(gpu_list)
                    print(f"Using specified GPU count as process count: {args.world_size}")
            mp.spawn(main_worker, args=(args.world_size, args), nprocs=args.world_size, join=True)
    else:
        # Single GPU training
        # If GPU list specified but only one GPU, use the first one
        if args.gpus is not None:
            gpu_list = [int(gpu_id.strip()) for gpu_id in args.gpus.split(',')]
            if len(gpu_list) == 1:
                args.gpu = gpu_list[0]
                print(f"Single GPU training, using GPU {args.gpu}")
            else:
                print(f"Warning: Single GPU training mode, but multiple GPUs specified. Will use --gpu parameter GPU {args.gpu}")
        main_worker(0, 1, args)
