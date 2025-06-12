import torch
import numpy as np
from mmengine.registry import Registry
OPENOCC_DATASET = Registry('openocc_dataset')
OPENOCC_DATAWRAPPER = Registry('openocc_datawrapper')
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader
from .dataset_nusc_surroundocc import NuScenes_Scene_SurroundOcc_Dataset
from .dataset_wrapper_nusc_occ import NuScenes_Scene_Occ_DatasetWrapper

def custom_collate_fn(data):
    data_tuple = []
    for i, item in enumerate(data[0]):
        if isinstance(item, np.ndarray):
            data_tuple.append(torch.from_numpy(np.stack([d[i] for d in data])))
        elif isinstance(item, (dict, str, list)):
            data_tuple.append([d[i] for d in data])
        elif item is None:
            data_tuple.append(None)
        else:
            raise NotImplementedError
    return data_tuple


def build_dataloader(
            train_dataset_config,
            val_dataset_config,
            train_wrapper_config,
            val_wrapper_config,
            train_loader_config,
            val_loader_config,
            dist=False,
    ):
    train_dataset = OPENOCC_DATASET.build(train_dataset_config)
    val_dataset = OPENOCC_DATASET.build(val_dataset_config)

    train_wrapper = OPENOCC_DATAWRAPPER.build(train_wrapper_config, default_args={'in_dataset': train_dataset})
    val_wrapper = OPENOCC_DATAWRAPPER.build(val_wrapper_config, default_args={'in_dataset': val_dataset})

    train_sampler = val_sampler = None
    if dist:
        train_sampler = DistributedSampler(train_wrapper, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_wrapper, shuffle=False, drop_last=False)

    train_dataset_loader = DataLoader(dataset=train_wrapper,
                                    batch_size=train_loader_config["batch_size"],
                                    collate_fn=custom_collate_fn,
                                    shuffle=False if dist else train_loader_config["shuffle"],
                                    sampler=train_sampler,
                                    num_workers=train_loader_config["num_workers"],
                                    pin_memory=True)
    val_dataset_loader = DataLoader(dataset=val_wrapper,
                                    batch_size=val_loader_config["batch_size"],
                                    collate_fn=custom_collate_fn,
                                    shuffle=False if dist else val_loader_config["shuffle"],
                                    sampler=val_sampler,
                                    num_workers=val_loader_config["num_workers"],
                                    pin_memory=True)

    return train_dataset_loader, val_dataset_loader