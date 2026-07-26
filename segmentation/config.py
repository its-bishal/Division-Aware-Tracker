
import torch
from ..utils import mkdir


class TrainParameters:

    def __init__(self, source_img_root, label_img_root, log_root,
                 validation_ratio, scale_img, weighted_type,
                 aug_list, batch_size, workers, gpu_num, resume, epochs,
                 lr, weight_decay, loss_type, **kwargs):

        # train
        self.train_seed = 0
        self.resume = resume
        self.epochs = epochs if epochs else 100
        
        # GPU detection logic
        if gpu_num and gpu_num > 0 and torch.cuda.is_available():
            self.gpu_num = gpu_num
            self.device = 'cuda'
        else:
            self.gpu_num = 0
            self.device = 'cpu'

        # IO
        self.source_img_root = source_img_root
        self.label_img_root = label_img_root
        self.log_root = log_root
        mkdir(self.log_root)

        # dataloader
        self.aug_list = aug_list
        self.weighted_type = weighted_type
        self.loss_type = loss_type if loss_type else 'WeightedSoftDiceLoss'
        
        self.batch_size = batch_size if batch_size else 2
        self.workers = workers if workers else 2
        self.validation_ratio = validation_ratio if validation_ratio else 0.2
        self.scale_img = scale_img if scale_img else 1.0

        self.dataloader_params = {
            'source_img_root': self.source_img_root,
            'label_img_root': self.label_img_root,
            'log_root': self.log_root,
            'validation_ratio': self.validation_ratio,
            'scale_img': self.scale_img,
            'weighted_type': self.weighted_type,
            'aug_list': self.aug_list,
            'batch_size': self.batch_size,
            'workers': self.workers
        }

        # optimizer
        if lr and weight_decay:
            self.lr = lr
            self.weight_decay = weight_decay
        else:
            self.lr = 0.001
            self.weight_decay = 0.0005

        self.optimizer_params = {
            'weight_decay': self.weight_decay,
            'lr': self.lr
        }

