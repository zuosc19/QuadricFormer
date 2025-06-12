import numpy as np

class LossRecord():
    
    def __init__(self, loss_func) -> None:
        self.loss_dict = dict()
        for loss in loss_func.losses:
            loss_name = getattr(loss, 'loss_name', loss.__class__.__name__)
            self.loss_dict[loss_name] = []
        self.total_loss = []
    
    def reset(self):
        for key in self.loss_dict.keys():
            self.loss_dict[key] = []
        self.total_loss = []

    def update(self, loss, loss_dict):
        for key in loss_dict.keys():
            self.loss_dict[key].append(loss_dict[key])
        self.total_loss.append(loss)
    
    def loss_info(self):
        info = ''
        for name, loss_list in self.loss_dict.items():
            info += '%s: %.3f (%.3f),   ' % (name, loss_list[-1], np.mean(loss_list))
        info += 'Loss: %.3f (%.3f),   ' % (self.total_loss[-1], np.mean(self.total_loss))
        
        return info