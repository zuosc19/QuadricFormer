import os
offscreen = False
WH = (1280, 720)
if os.environ.get('DISP', 'f') == 'f':
    from pyvirtualdisplay import Display
    display = Display(visible=False, size=WH)
    display.start()
    offscreen = True
# from xvfbwrapper import Xvfb
# vdisplay = Xvfb(width=1920, height=1080)
# vdisplay.start()

from mayavi import mlab
import mayavi
mlab.options.offscreen = offscreen
print("Set mlab.options.offscreen={}".format(mlab.options.offscreen))

import os, time, argparse, os.path as osp, numpy as np
import torch
import torch.distributed as dist
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from utils.iou_eval import IOUEvalBatch
from utils.loss_record import LossRecord
from utils.load_save_util import revise_ckpt, revise_ckpt_2

from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging.logger import MMLogger
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from pyquaternion import Quaternion

import warnings
warnings.filterwarnings("ignore")


def plot_opa_hist(opas, save_name):
    plt.cla(); plt.clf()
    plt.hist(opas, range=(0, 1), bins=20)
    plt.savefig(save_name)
    plt.cla(); plt.clf()

def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0]) # [0, 1, ..., 256]
    # g_xx = g_xx[::-1]
    g_yy = np.arange(0, dims[1]) # [0, 1, ..., 256]
    # g_yy = g_yy[::-1]
    g_zz = np.arange(0, dims[2]) # [0, 1, ..., 32]

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)
    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])

    coords_grid = (coords_grid * resolution) + resolution / 2

    return coords_grid

def draw(
    voxels=None,          # semantic occupancy predictions
    vox_origin=None,
    voxel_size=0.2,  # voxel size in the real world
    sem=False,
    save_path=None,
    v_min=0.0,
    v_max=100.0,
    use_grad=True
):
    w, h, z = voxels.shape
    # grid = grid.astype(np.int)
    # voxels[98:102, 95:105, 8:10] = 0

    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    ) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])

    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T

    # Get the voxels inside FOV
    fov_grid_coords = grid_coords

    # Remove empty and unknown voxels
    fov_voxels = fov_grid_coords[
        (fov_grid_coords[:, 3] >= 0) & (fov_grid_coords[:, 3] < 17)
    ]
    print('occ num:', len(fov_voxels))

    figure = mlab.figure(size=WH, bgcolor=(1, 1, 1))
    # Draw occupied inside FOV voxels
    voxel_size = sum(voxel_size) / 3
    if not sem:
        plt_plot_fov = mlab.points3d(
            fov_voxels[:, 0],
            fov_voxels[:, 1],
            fov_voxels[:, 2],
            fov_voxels[:, 3],
            colormap="jet",
            scale_factor=1.0 * voxel_size,
            mode="cube",
            opacity=1.0,
            # transparent=True,
            # vmin=v_min,
            # vmax=v_max, # 16
        )
    else:
        plt_plot_fov = mlab.points3d(
            fov_voxels[:, 0],
            -fov_voxels[:, 1],
            fov_voxels[:, 2],
            fov_voxels[:, 3],
            scale_factor=1.0 * voxel_size,
            mode="cube",
            opacity=1.0,
            # transparent=True,
            vmin=0,
            vmax=16, # 16
        )
        plt_plot_fov.glyph.scale_mode = "scale_by_vector"
        colors = np.array(
            [
                [  0,   0,   0, 255],       # others
                [255, 120,  50, 255],       # barrier              orange
                [255, 192, 203, 255],       # bicycle              pink
                [255, 255,   0, 255],       # bus                  yellow
                [  0, 150, 245, 255],       # car                  blue
                [  0, 255, 255, 255],       # construction_vehicle cyan
                [255, 127,   0, 255],       # motorcycle           dark orange
                [255,   0,   0, 255],       # pedestrian           red
                [255, 240, 150, 255],       # traffic_cone         light yellow
                [135,  60,   0, 255],       # trailer              brown
                [160,  32, 240, 255],       # truck                purple                
                [255,   0, 255, 255],       # driveable_surface    dark pink
                # [175,   0,  75, 255],       # other_flat           dark red
                [139, 137, 137, 255],
                [ 75,   0,  75, 255],       # sidewalk             dard purple
                [150, 240,  80, 255],       # terrain              light green          
                [230, 230, 250, 255],       # manmade              white
                [  0, 175,   0, 255],       # vegetation           green
                # [  0, 255, 127, 255],       # ego car              dark cyan
                # [255,  99,  71, 255],       # ego car
                # [  0, 191, 255, 255]        # ego car
            ]
        ).astype(np.uint8)
        plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors
    
    scene = figure.scene
    if False:
        scene.camera.position = [0, 0, 220]   # 远离地面
        scene.camera.focal_point = [0, 0, 0]
        scene.camera.view_up = [1, 0, 0]       # 或[0,1,0], 取决于你Y/X朝向
        scene.camera.compute_view_plane_normal()
        scene.render()

        scene.camera.elevation(-5)
        mlab.pitch(-8)
        mlab.move(up=35)
        scene.camera.orthogonalize_view_up()
        scene.render()
    else:
        scene.camera.position = [118.7195754824976, 118.70290907014409, 120.11124225247899]
        scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, 1.399999976158142]
        scene.camera.view_angle = 30.0
        scene.camera.view_up = [0.0, 0.0, 1.0]
        scene.camera.clipping_range = [114.42016931210819, 320.9039783052695]
        scene.camera.compute_view_plane_normal()
        scene.render()
        scene.camera.azimuth(-5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(-5)
        scene.render()
        scene.camera.position = [-138.7379881436844, -0.008333206176756428, 99.5084646673331]
        scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, 1.399999976158142]
        scene.camera.view_angle = 30.0
        scene.camera.view_up = [0.0, 0.0, 1.0]
        scene.camera.clipping_range = [104.37185230017721, 252.84608651497263]
        scene.camera.compute_view_plane_normal()
        scene.render()
        scene.camera.position = [-114.65804807470022, -0.008333206176756668, 82.48137575398867]
        scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, 1.399999976158142]
        scene.camera.view_angle = 30.0
        scene.camera.view_up = [0.0, 0.0, 1.0]
        scene.camera.clipping_range = [75.17498702830105, 222.91192666552377]
        scene.camera.compute_view_plane_normal()
        scene.render()
        scene.camera.position = [-94.75727115818437, -0.008333206176756867, 68.40940144543957]
        scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, 1.399999976158142]
        scene.camera.view_angle = 30.0
        scene.camera.view_up = [0.0, 0.0, 1.0]
        scene.camera.clipping_range = [51.04534630774225, 198.1729515833347]
        scene.camera.compute_view_plane_normal()
        scene.render()
        scene.camera.elevation(5)
        scene.camera.orthogonalize_view_up()
        scene.render()
        scene.camera.position = [-107.15500034628069, -0.008333206176756742, 92.16667026873841]
        scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, 1.399999976158142]
        scene.camera.view_angle = 30.0
        scene.camera.view_up = [0.6463156430702276, -6.454925414290924e-18, 0.7630701733934554]
        scene.camera.clipping_range = [78.84362692774403, 218.2948716014858]
        scene.camera.compute_view_plane_normal()
        scene.render()
        scene.camera.position = [-107.15500034628069, -0.008333206176756742, 92.16667026873841]
        scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, 1.399999976158142]
        scene.camera.view_angle = 30.0
        scene.camera.view_up = [0.6463156430702277, -6.4549254142909245e-18, 0.7630701733934555]
        scene.camera.clipping_range = [78.84362692774403, 218.2948716014858]
        scene.camera.compute_view_plane_normal()
        scene.render()
        scene.camera.elevation(5)
        scene.camera.orthogonalize_view_up()
        scene.render()
        scene.camera.elevation(5)
        scene.camera.orthogonalize_view_up()
        scene.render()
        scene.camera.elevation(-5)
        mlab.pitch(-8)
        mlab.move(up=15)
        scene.camera.orthogonalize_view_up()
        scene.render()
    # scene.camera.position = [-80, -80, 40]
    # scene.camera.focal_point = [0, 0, 1.4]
    # scene.camera.view_up = [0, 0, 1]
    # scene.camera.clipping_range = [180, 200]
    # scene.camera.compute_view_plane_normal()
    # scene.camera.elevation(5)
    # mlab.pitch(-10)
    # mlab.move(up=8)
    # scene.camera.orthogonalize_view_up()
    # scene.render()

    mlab.savefig(save_path, size=WH)
    mlab.close()

def get_nuscenes_colormap():
    colors = np.array(
        [
            [  0,   0,   0, 255],       # others
            [255, 120,  50, 255],       # barrier              orange
            [255, 192, 203, 255],       # bicycle              pink
            [255, 255,   0, 255],       # bus                  yellow
            [  0, 150, 245, 255],       # car                  blue
            [  0, 255, 255, 255],       # construction_vehicle cyan
            [255, 127,   0, 255],       # motorcycle           dark orange
            [255,   0,   0, 255],       # pedestrian           red
            [255, 240, 150, 255],       # traffic_cone         light yellow
            [135,  60,   0, 255],       # trailer              brown
            [160,  32, 240, 255],       # truck                purple                
            [255,   0, 255, 255],       # driveable_surface    dark pink
            [139, 137, 137, 255],       # other_flat           dark red
            [ 75,   0,  75, 255],       # sidewalk             dard purple
            [150, 240,  80, 255],       # terrain              light green          
            [230, 230, 250, 255],       # manmade              white
            [  0, 175,   0, 255],       # vegetation           green
        ]
    ).astype(np.float32) / 255.
    return colors

def spow(a, p):
    return np.sign(a) * (np.abs(a) ** p)

def save_gaussian(save_dir, gaussian, name, scalar=1.5, ignore_opa=False, filter_zsize=False, sem=True, vis_opas=False, v_min=0.0, v_max=1.0):

    empty_label = 17
    sem_cmap = get_nuscenes_colormap()

    # torch.save(gaussian, os.path.join(save_dir, f'{name}_attr.pth'))

    means = gaussian['means'][-1].detach().cpu().numpy()            # g, 3
    scales = gaussian['scales'][-1].detach().cpu().numpy()          # g, 3
    rotations = gaussian['rot'][-1].detach().cpu().numpy()          # g, 4
    opas = gaussian['opa'][-1].squeeze().detach().cpu().numpy()     # g
    opas = opas + 0.5
    opas = opas / (np.max(opas) + 1e-5)
    u = gaussian['u'][-1].squeeze().detach().cpu().numpy()          # g
    v = gaussian['v'][-1].squeeze().detach().cpu().numpy()          # g
    sems = gaussian['sem'][-1].detach().cpu().numpy()               # g, 18
    pred = np.argmax(sems, axis=-1)

    if ignore_opa:
        opas[:] = 0.7
        mask = (pred != empty_label)
    else:
        mask = (pred != empty_label) & (opas > 0.01)

    if filter_zsize:
        zdist, zbins = np.histogram(means[:, 2], bins=100)
        zidx = np.argsort(zdist)[::-1]
        for idx in zidx[:2]:
            binl = zbins[idx]
            binr = zbins[idx + 1]
            zmsk = (means[:, 2] < binl) | (means[:, 2] > binr)
            mask = mask & zmsk
        
        z_small_mask = scales[:, 2] > 0.1
        mask = z_small_mask & mask
    # s = 0.2
    # mask = mask & (scales[:, 0] > s) & (scales[:, 0] > s) & (scales[:, 0] > s) 
    means = means[mask]
    scales = scales[mask]
    rotations = rotations[mask]
    opas = opas[mask]
    u = u[mask]
    v = v[mask]
    pred = pred[mask]

    # number of ellipsoids 
    ellipNumber = means.shape[0]
    print('gs num:', ellipNumber)

    #set colour map so each ellipsoid as a unique colour
    norm = colors.Normalize(vmin=-1.0, vmax=5.4)
    cmap = cm.jet
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    fig = plt.figure(figsize=(9, 9), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=46, azim=-180)

    # compute each and plot each ellipsoid iteratively
    border = np.array([
        [-50.0, -50.0, 0.0],
        [-50.0, 50.0, 0.0],
        [50.0, -50.0, 0.0],
        [50.0, 50.0, 0.0],
    ])
    ax.plot_surface(border[:, 0:1], border[:, 1:2], border[:, 2:], 
        rstride=1, cstride=1, color=[0, 0, 0, 1], linewidth=0, alpha=0., shade=True)

    theta = np.linspace(-np.pi/2, np.pi/2, 10)
    phi = np.linspace(-np.pi, np.pi, 10)
    theta, phi = np.meshgrid(theta, phi)
    all_xyz = []
    for indx in range(ellipNumber):
        center = means[indx]
        radii = scales[indx] * scalar
        rot_matrix = np.linalg.inv(rotations[indx])
        uu = u[indx]
        vv = v[indx]
        x = radii[0] * spow(np.cos(theta), uu) * spow(np.cos(phi), uu)
        y = radii[1] * spow(np.cos(theta), uu) * spow(np.sin(phi), uu)
        z = radii[2] * spow(np.sin(theta), vv)
        xyz = np.stack([x, y, z], axis=-1) # phi, theta, 3
        xyz = rot_matrix[None, None, ...] @ xyz[..., None]
        xyz = np.squeeze(xyz, axis=-1)
        xyz = xyz + center[None, None, ...]
        all_xyz.append(xyz)
    all_xyz_cat = np.stack(all_xyz, axis=0).reshape(-1, 3)
    ax.set_box_aspect([np.ptp(all_xyz_cat[..., 0]), np.ptp(all_xyz_cat[..., 1]), np.ptp(all_xyz_cat[..., 2])])
    ax.set_position([0, 0, 1, 1])
    for indx, xyz in enumerate(all_xyz):
        ax.plot_surface(
            xyz[..., 1], -xyz[..., 0], xyz[..., 2], 
            rstride=1, cstride=1, color=sem_cmap[pred[indx]], linewidth=0, alpha=opas[indx], shade=True)

    plt.axis("equal")
    # plt.gca().set_box_aspect([1, 1, 1])
    ax.grid(False)
    ax.set_axis_off()    
    # ax.set_axis_on()
    # ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.tight_layout(pad=0.0)

    filepath = os.path.join(save_dir, f'{name}.png')
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0)

    plt.cla()
    plt.clf()

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
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    set_random_seed(cfg.seed)
    cfg.work_dir = args.work_dir
    cfg.val_dataset_config.scene_name = args.scene_name

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
    logger = MMLogger(name='bevworld', log_file=None, log_level='INFO')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    from model import build_model
    my_model = build_model(cfg.model)
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    logger.info(f'Model:\n{my_model}')
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', True)
        if cfg.get('track_running_stats', False):
            my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
            logger.info('converted sync bn.')
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model.cuda(),
            device_ids=[gpu],
            find_unused_parameters=find_unused_parameters)
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
    batch_iou = len(cfg.model.encoder.return_layer_idx)
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
        
    if cfg.val_dataset_config.scene_name is None:
        save_dir = os.path.join(args.work_dir)
    else:
        save_dir = os.path.join(args.work_dir, cfg.val_dataset_config.scene_name)
    os.makedirs(save_dir, exist_ok=True)

    my_model.eval()
    CalMeanIou_sem.reset()
    CalMeanIou_geo.reset()
    loss_record = LossRecord(loss_func=loss_func)
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    ignore_labels = args.ignore_labels
    if not isinstance(ignore_labels, list):
        ignore_labels = [ignore_labels]
    print('ignore labels:', ignore_labels)
    frame_idx = 0
    scene_name = ''
    with torch.no_grad():
        for i_iter_val, data in enumerate(val_dataset_loader):
            for i in range(len(data)):
                if isinstance(data[i], torch.Tensor):
                    data[i] = data[i].cuda()
            (imgs, metas, label) = data
            # if scene_name != metas[0]['scene_name']:
            #     scene_name = metas[0]['scene_name']
            #     frame_idx = 0
            # else:
            #     frame_idx += 1
            # if frame_idx != 20:
            #     continue

            with torch.cuda.amp.autocast(enabled=amp):
                result_dict = my_model(imgs=imgs, metas=metas, label=label, return_anchors=True)
            voxel_predict = result_dict['ce_input'].argmax(dim=1).long()
            voxel_label = result_dict['ce_label'].long()
            iou_predict = ((voxel_predict > 0) & (voxel_predict < 17)).long()
            iou_label = ((voxel_label > 0) & (voxel_label < 17)).long()
            CalMeanIou_sem.addBatch(voxel_predict, voxel_label)
            CalMeanIou_geo.addBatch(iou_predict, iou_label)

            frame_idx = i_iter_val
            # vis gs
            if args.vis_gs:
                gaussian = result_dict['anchors']
                save_gaussian(save_dir, gaussian, f"gs_{frame_idx}", scalar=1.0, sem=True, ignore_opa=False, filter_zsize=False)
            # vis occ
            if args.vis_occ:
                voxel_predict = torch.argmax(result_dict['ce_input'][-1], dim=0).long()
                voxel_label = result_dict['ce_label'][-1].long()
                voxel_origin = cfg.pc_range[:3]
                resolution = 0.5
                for i in ignore_labels:
                    voxel_predict[voxel_predict==i] = 17
                    voxel_label[voxel_label==i] = 17
                to_vis = voxel_predict.clone().cpu().numpy()
                save_path = os.path.join(save_dir, f'occ_{frame_idx}.png')
                draw(to_vis, 
                    voxel_origin, 
                    [resolution] * 3, 
                    sem=True,
                    save_path=save_path)
                to_vis = voxel_label.clone().cpu().numpy()
                save_path = os.path.join(save_dir, f'occ_gt_{frame_idx}.png')
                draw(to_vis, 
                    voxel_origin, 
                    [resolution] * 3, 
                    sem=True,
                    save_path=save_path)
        
            if i_iter_val % 1 == 0 and is_main_process():
                logger.info('[EVAL] Iter %5d/%d    Memory  %4d M  '%(i_iter_val, len(val_dataset_loader), int(torch.cuda.max_memory_allocated()/1e6)))
                # break
    val_iou_sem = CalMeanIou_sem.getIoU()
    val_iou_geo = CalMeanIou_geo.getIoU()
    info_sem = [[float('{:.4f}'.format(iou)) for iou in val_iou_sem[i, 1:17].mean(-1, keepdim=True).tolist()] for i in range(val_iou_sem.shape[0])]
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
    parser.add_argument('--ignore-labels', type=int, nargs='+', default=0)
    parser.add_argument('--vis-occ', action='store_true')
    parser.add_argument('--vis-gs', action='store_true')

    args, _ = parser.parse_known_args()
    main(args)