import os
import torch
import torch.nn as nn
import shutil
import pdb
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.1)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.1)
        m.bias.data.fill_(0)


def save_checkpoint(state, is_best, checkpoint='checkpoint',
                    filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))

def update_prototype(tensor, gt, proto_tensor):
    # pdb.set_trace()
    gt_unique = torch.unique(gt).cpu().numpy()
    for item in gt_unique:
        mask = (gt == item).nonzero() #.item() #.unsqueeze(1).repeat(1,tensor.shape[1]).float()
        tensor_sel = torch.index_select(tensor, 0, mask.squeeze())
        if mask.shape[0] > 1:    
            proto_tensor[item,:] = F.normalize(0.1 * F.normalize(torch.mean(tensor_sel,0).unsqueeze(0)) + \
             0.9 * proto_tensor[item,:])
        else:
            proto_tensor[item,:] = F.normalize(0.1 * F.normalize(tensor_sel) + 0.9 * proto_tensor[item,:])
    # pdb.set_trace()
    return proto_tensor
