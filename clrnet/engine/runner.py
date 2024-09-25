import time
import cv2
import torch
from tqdm import tqdm
import pytorch_warmup as warmup
import numpy as np
import random
import os
from PIL import Image

from clrnet.models.registry import build_net
from .registry import build_trainer, build_evaluator
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from clrnet.datasets import build_dataloader
from clrnet.utils.recorder import build_recorder
from clrnet.utils.net_utils import load_network, resume_network
from mmcv.parallel import MMDataParallel



class Runner(object):
    def __init__(self, cfg):
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        self.cfg = cfg
        self.recorder = build_recorder(self.cfg)
        self.net = build_net(self.cfg)
        self.net = MMDataParallel(self.net,
                                  device_ids=range(self.cfg.gpus)).cuda()
        self.recorder.logger.info('Network: \n' + str(self.net))
        self.resume()
        self.optimizer = build_optimizer(self.cfg, self.net)
        self.scheduler = build_scheduler(self.cfg, self.optimizer)
        self.metric = 0.
        self.num = 0
        self.val_loader = None
        self.test_loader = None

    def to_cuda(self, batch):
        for k in batch:
            if not isinstance(batch[k], torch.Tensor):
                continue
            batch[k] = batch[k].cuda()
        return batch

    def resume(self):
        if not self.cfg.load_from and not self.cfg.finetune_from:
            return
        load_network(self.net, self.cfg.load_from, finetune_from=self.cfg.finetune_from, logger=self.recorder.logger)




    def validate(self):
        if not self.val_loader:
            self.val_loader = build_dataloader(self.cfg.dataset.val,
                                               self.cfg,
                                               is_train=False)
        self.net.eval()
        predictions = []
        self.num += 1

        for i, data in enumerate(tqdm(self.val_loader, desc=f'Validate')):
            data = self.to_cuda(data)
            with torch.no_grad():
                image, output = self.net(data)
                image = image.squeeze()
                image = image.detach().cpu().numpy()[0]
                image = Image.fromarray(np.uint8(image))
                img_metas = data['meta']
                img_metas = [item for img_meta in img_metas.data for item in img_meta][0]
                img_name = img_metas['img_name']
                path = os.path.join(self.cfg.work_dir, 'mask_vis', '{}'.format(self.num))
                if not os.path.exists(path):
                    os.makedirs(path)
                out_file = os.path.join(path, img_name.replace('/', '_'))
                image.save(out_file)

                output = self.net.module.heads.get_lanes(output)
                predictions.extend(output)

            if self.cfg.view:
                self.val_loader.dataset.view(output, data['meta'])

