
import os, time, datetime, argparse, os.path as osp, numpy as np
import torch
import torch.distributed as dist
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# os.environ['PYTORCHCUDAALLOCCONF'] = 'max_split_size_mb:128'

from utils.iou_eval import IOUEvalBatch
from utils.loss_record import LossRecord
from utils.load_save_util import revise_ckpt, revise_ckpt_2

from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.optim import build_optim_wrapper
from mmengine.logging import MMLogger
from mmengine.utils import symlink
from timm.scheduler import CosineLRScheduler

import warnings
warnings.filterwarnings("ignore")

try:
    import gpu_affinity
except ImportError as e:
    raise ImportError(
        "An error occurred while trying to import : gpu_affinity, "
        + "install gpu_affinity by 'pip install git+https://github.com/NVIDIA/gpu_affinity' please"
    )


def pass_print(*args, **kwargs):
    pass

def is_main_process():
    if not dist.is_available():
        return True
    elif not dist.is_initialized():
        return True
    else:
        return dist.get_rank() == 0

def main(args):
    # global settings
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    set_random_seed(cfg.seed)
    cfg.work_dir = args.work_dir
    max_num_epochs = cfg.max_epochs
    eval_freq = cfg.eval_freq
    print_freq = cfg.print_freq

    # init DDP
    print('distributed init start')
    distributed = True
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    node_num = os.environ.get("NODE_NUM", None)
    gpu_count = os.environ.get("GPU_COUNT", None)
    world_size = int(os.environ.get("WORLD_SIZE", None))
    if node_num is not None and gpu_count is not None:
        assert int(node_num) * int(gpu_count) == world_size
        print(f"node_num={node_num}, gpu_count={gpu_count}", flush=True)

    device_count = torch.cuda.device_count()
    if dist.is_available():
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            if local_rank == 0:
                print(
                    "lpai torch distributed is already initialized, "
                    "skipping initialization ...",
                    flush=True,
                )
        else:
            # Manually set the device ids.
            if device_count > 0:
                device = rank % device_count
                assert device == local_rank
                # if args.local_rank is not None:
                #    assert args.local_rank == device, \
                #        'expected local-rank to be the same as rank % device-count.'
                # else:
                #    args.local_rank = device
                torch.cuda.set_device(device)
            dist.init_process_group(
                backend=args.backend,
                world_size=world_size,
                rank=rank,
                timeout=datetime.timedelta(hours=1),
            )
    affinity = gpu_affinity.set_affinity(local_rank, device_count)
    print(f"rank={rank}, local_rank={local_rank}, world_size={world_size}", flush=True)

    if not is_main_process():
        import builtins
        builtins.print = pass_print

    # configure logger
    if is_main_process():
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = MMLogger(name='bevworld', log_file=log_file, log_level='INFO')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    from model import build_model
    my_model = build_model(cfg.model)
    if cfg.get('freeze_perception', False):
        for n, p in my_model.named_parameters():
            if 'backbone' in n or 'neck' in n:
            # if not 'future_decoder' in n:
                p.requires_grad_(False)
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    logger.info(f'Model:\n{my_model}')
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', True)
        if cfg.get('track_running_stats', True):
            my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
            logger.info('converted sync bn.')
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            find_unused_parameters=find_unused_parameters)
        my_model._set_static_graph() 
    else:
        my_model = my_model.cuda()
    print('done ddp model')

    # build dataloader
    from dataset import build_dataloader
    train_dataset_loader, val_dataset_loader = \
        build_dataloader(
            cfg.train_dataset_config,
            cfg.val_dataset_config,
            cfg.train_wrapper_config,
            cfg.val_wrapper_config,
            cfg.train_loader_config,
            cfg.val_loader_config,
            dist=distributed,
        )

    # get optimizer, loss, scheduler
    amp = cfg.get('amp', True)
    optimizer = build_optim_wrapper(my_model, cfg.optimizer_wrapper)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    from loss import GPD_LOSS
    loss_func = GPD_LOSS.build(cfg.loss).cuda()
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=len(train_dataset_loader) * max_num_epochs,
            lr_min=cfg.optimizer_wrapper["optimizer"]["lr"] * 0.1, #1e-6,
            warmup_t=cfg.get('warmup_iters', 500),
            warmup_lr_init=1e-6,
            t_in_epochs=False
    )

    batch_iou = len(cfg.model.encoder.return_layer_idx)
    CalMeanIou_sem = IOUEvalBatch(n_classes=18, bs=batch_iou, device=torch.device('cpu'), ignore=[0], is_distributed=distributed)
    CalMeanIou_geo = IOUEvalBatch(n_classes=2, bs=batch_iou, device=torch.device('cpu'), ignore=[], is_distributed=distributed)
    
    # resume and load
    epoch = 0
    global_iter = 0

    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from
    if args.load_from:
        cfg.load_from = args.load_from
    
    print('resume from: ', cfg.resume_from)
    print('load from: ', cfg.load_from)
    print('work dir: ', args.work_dir)

    if cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        print(my_model.load_state_dict(revise_ckpt(ckpt['state_dict']), strict=False))
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        epoch = ckpt['epoch']
        global_iter = ckpt['global_iter']
        print(f'successfully resumed from epoch {epoch}')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        state_dict = revise_ckpt(state_dict)
        try:
            print(my_model.load_state_dict(state_dict, strict=False))
        except:
            state_dict = revise_ckpt_2(state_dict)
            print(my_model.load_state_dict(state_dict, strict=False))
        

    # training
    while epoch < max_num_epochs:
        my_model.train()
        if hasattr(train_dataset_loader.sampler, 'set_epoch'):
            train_dataset_loader.sampler.set_epoch(epoch)
        loss_record = LossRecord(loss_func=loss_func)
        time.sleep(1)
        data_time_s = time.time()
        time_s = time.time()
        for i_iter, data in enumerate(train_dataset_loader):
            for i in range(len(data)):
                if isinstance(data[i], torch.Tensor):
                    data[i] = data[i].cuda()
            (imgs, metas, label) = data
            # forward + backward + optimize
            data_time_e = time.time()
            with torch.cuda.amp.autocast(enabled=amp):
                result_dict = my_model(imgs=imgs, metas=metas, label=label, test_mode=False)
            loss, loss_dict = loss_func(result_dict)
            loss_record.update(loss=loss.item(), loss_dict=loss_dict)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)
            # gradient anomaly detection
            valid_grad = True
            for n, p in my_model.module.named_parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        loss_info = loss_record.loss_info()
                        logger.warning(f'Parameter {n} has Nan or Inf grad. Skipping this step \n ' + 
                                       '[Skipping] Epoch %d Iter %5d/%d   SceneName %s   ' % (epoch+1, i_iter, len(train_dataset_loader), 
                                        metas[0]['scene_name']) + loss_info + 'GradNorm: %.3f' % (grad_norm))
                        valid_grad = False
                        break
            if not valid_grad:
                optimizer.zero_grad()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step_update(global_iter)
            time_e = time.time()

            global_iter += 1
            if i_iter % print_freq == 0 and is_main_process():
                lr = optimizer.param_groups[-1]['lr']
                loss_info = loss_record.loss_info()
                logger.info('[TRAIN] Epoch %d Iter %5d/%d   ' % (epoch+1, i_iter, len(train_dataset_loader)) + loss_info +
                            'GradNorm: %.3f,   lr: %.7f,   time: %.3f (%.3f)' % (grad_norm, lr, time_e - time_s, data_time_e - data_time_s))
                loss_record.reset()
            torch.cuda.empty_cache()
            data_time_s = time.time()
            time_s = time.time()
        
        # save checkpoint
        if is_main_process():
            dict_to_save = {
                'state_dict': my_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'global_iter': global_iter,
            }
            save_file_name = os.path.join(os.path.abspath(args.work_dir), f'epoch_{epoch+1}.pth')
            torch.save(dict_to_save, save_file_name)
            dst_file = osp.join(args.work_dir, 'latest.pth')
            symlink(save_file_name, dst_file)

        epoch += 1
        
        # eval
        if epoch % eval_freq == 0:
            my_model.eval()
            CalMeanIou_sem.reset()
            CalMeanIou_geo.reset()
            loss_record = LossRecord(loss_func=loss_func)
            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            with torch.no_grad():
                for i_iter_val, data in enumerate(val_dataset_loader):
                    for i in range(len(data)):
                        if isinstance(data[i], torch.Tensor):
                            data[i] = data[i].cuda()
                    (imgs, metas, label) = data
                    
                    with torch.cuda.amp.autocast(enabled=amp):
                        result_dict = my_model(imgs=imgs, metas=metas, label=label, test_mode=True)
                    loss, loss_dict = loss_func(result_dict)
                    
                    loss_record.update(loss=loss.item(), loss_dict=loss_dict)
                    voxel_predict = result_dict['ce_input'].argmax(dim=1).long()
                    voxel_label = result_dict['ce_label'].long()
                    iou_predict = ((voxel_predict > 0) & (voxel_predict < 17)).long()
                    iou_label = ((voxel_label > 0) & (voxel_label < 17)).long()
                    CalMeanIou_sem.addBatch(voxel_predict, voxel_label)
                    CalMeanIou_geo.addBatch(iou_predict, iou_label)
                                        
                    if i_iter_val % print_freq == 0 and is_main_process():
                        loss_info = loss_record.loss_info()
                        logger.info('[EVAL] Iter %5d/%d   '%(i_iter_val, len(val_dataset_loader)) + loss_info)
                        # loss_record.reset()
                    torch.cuda.empty_cache()

            val_iou_sem = CalMeanIou_sem.getIoU()
            val_iou_geo = CalMeanIou_geo.getIoU()
            info_sem = [float('{:.4f}'.format(iou)) for iou in val_iou_sem[:, 1:17].mean(-1).tolist()]
            info_geo = [float('{:.4f}'.format(iou)) for iou in val_iou_geo[:, 1].tolist()]
            
            logger.info(val_iou_sem.cpu().tolist())
            logger.info(f'Current val iou of sem is {info_sem}')
            logger.info(f'Current val iou of geo is {info_geo}')
        

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv_occ.py')
    parser.add_argument('--work-dir', type=str, default='./work_dir/tpv_occ')
    parser.add_argument('--load-from', type=str, default='')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument(
        "--backend",
        type=str,
        help="Distributed backend",
        choices=[dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI],
        default=dist.Backend.NCCL,
    )

    args, _ = parser.parse_known_args()
    main(args)