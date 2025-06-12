import torch
import torch.distributed as dist
import numpy as np


class PlanRecord():
    
    def __init__(self) -> None:
        self.reset()
    
    def reset(self):
        self.metric_dict = {'plan_L2_1s':0,
                            'plan_L2_2s':0,
                            'plan_L2_3s':0,
                            'plan_obj_col_1s':0,
                            'plan_obj_col_2s':0,
                            'plan_obj_col_3s':0,
                            'plan_obj_box_col_1s':0,
                            'plan_obj_box_col_2s':0,
                            'plan_obj_box_col_3s':0,}
        self.sample_num = 0

    def update(self, metric_dict):
        for key in metric_dict.keys():
            self.metric_dict[key] += metric_dict[key]
        self.sample_num += 1
    
    def loss_info(self, reduce=False, world_size=None):
        if reduce:
            metric = {key:torch.tensor(self.metric_dict[key], dtype=torch.float32).cuda() for key in self.metric_dict.keys()}
            for key in metric.keys():
                dist.all_reduce(metric[key])
                metric[key] /= world_size
        else:
            metric = self.metric_dict
        info = ''
        for name, value in metric.items():
            info += '%s: %.4f,   ' % (name, value / self.sample_num)
        plan_l2_avg = (metric['plan_L2_1s'] + metric['plan_L2_2s'] + metric['plan_L2_3s']) / 3
        info += '%s: %.4f,   ' % ('plan_L2_avg',  plan_l2_avg / self.sample_num)
        plan_obj_box_col_avg = (metric['plan_obj_box_col_1s'] + metric['plan_obj_box_col_2s'] + metric['plan_obj_box_col_3s']) / 3
        info += '%s: %.4f,   ' % ('plan_obj_box_col_avg',  plan_obj_box_col_avg / self.sample_num)
        
        return info