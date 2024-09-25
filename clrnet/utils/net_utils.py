import torch
import os
from torch import nn
import numpy as np
import torch.nn.functional


def load_network_specified(net, model_dir, logger=None):
    pretrained_net = torch.load(model_dir)['net']
    net_state = net.state_dict()
    state = {}
    for k, v in pretrained_net.items():
        if k not in net_state.keys() or v.size() != net_state[k].size():
            if logger:
                logger.info('skip weights: ' + k)
            continue
        state[k] = v
    net.load_state_dict(state, strict=False)


def load_network(net, model_dir, finetune_from=None, logger=None):
    if finetune_from:
        if logger:
            logger.info('Finetune model from: ' + finetune_from)
        load_network_specified(net, finetune_from, logger)
        return
    pretrained_model = torch.load(model_dir)
    net.load_state_dict(pretrained_model['net'], strict=False)


def resume_network(model_dir, net, optim, scheduler, recorder):
    if not os.path.exists(model_dir):
        print('WARNING: NO MODEL LOADED !!!', 'red')
        return 0

    print('resume model: {}'.format(model_dir))

    pretrained_model = torch.load(model_dir)
    net.load_state_dict(pretrained_model['net'])
    optim.load_state_dict(pretrained_model['optim'])
    scheduler.load_state_dict(pretrained_model['scheduler'])
    recorder.load_state_dict(pretrained_model['recorder'])
    return pretrained_model['epoch'] + 1
