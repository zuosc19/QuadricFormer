
import os, time, argparse, os.path as osp, numpy as np
import torch
import torch.distributed as dist
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from utils.iou_eval import IOUEvalBatch
from utils.loss_record import LossRecord
from utils.load_save_util import revise_ckpt, revise_ckpt_2

from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging import MMLogger

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
    cfg.val_dataset_config.scene_name = args.scene_name
    print_freq = cfg.print_freq

    # init DDP
    distributed = True
    world_size = int(os.environ["WORLD_SIZE"])  # number of nodes
    rank = int(os.environ["RANK"])  # node id
    gpu = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(
        backend="nccl", init_method=f"env://", 
        world_size=world_size, rank=rank
    )
    # dist.barrier()
    torch.cuda.set_device(gpu)

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
            device_ids=[gpu],
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

    amp = cfg.get('amp', True)
    from loss import GPD_LOSS
    loss_func = GPD_LOSS.build(cfg.loss).cuda()
    batch_iou = 1
    CalMeanIou_sem = IOUEvalBatch(n_classes=18, bs=batch_iou, device=torch.device('cpu'), ignore=[0], is_distributed=distributed)
    CalMeanIou_geo = IOUEvalBatch(n_classes=2, bs=batch_iou, device=torch.device('cpu'), ignore=[], is_distributed=distributed)
    
    # resume and load
    if args.load_from:
        cfg.load_from = args.load_from
    print('work dir: ', args.work_dir)
    if cfg.load_from:
        print('load from: ', cfg.load_from)
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
    
    # eval
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
                result_dict = my_model(imgs=imgs, metas=metas, label=label)
            loss, loss_dict = loss_func(result_dict)

            loss_record.update(loss=loss.item(), loss_dict=loss_dict)
            voxel_predict = result_dict['ce_input'][-1:].argmax(dim=1).long()
            voxel_label = result_dict['ce_label'][-1:].long()
            iou_predict = ((voxel_predict > 0) & (voxel_predict < 17)).long()
            iou_label = ((voxel_label > 0) & (voxel_label < 17)).long()
            CalMeanIou_sem.addBatch(voxel_predict, voxel_label)
            CalMeanIou_geo.addBatch(iou_predict, iou_label)
        
            if i_iter_val % print_freq == 0 and is_main_process():
                loss_info = loss_record.loss_info()
                logger.info('[EVAL] Iter %5d/%d    Memory  %4d M  '%(i_iter_val, len(val_dataset_loader), int(torch.cuda.max_memory_allocated()/1e6)) + loss_info)
                # loss_record.reset()
            # torch.cuda.empty_cache()

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
    parser.add_argument('--load-from', type=str, default=None)
    parser.add_argument('--scene-name', type=str, default=None)

    args, _ = parser.parse_known_args()
    main(args)