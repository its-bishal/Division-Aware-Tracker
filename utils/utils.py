
import os
import errno
import shutil
import torch
from builtins import OSError


def mkdir(path):
    """mimic the behavior of mkdir -p in bash"""
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def save_checkpoints(state, isbest, fpath="checkpoint.pth.tar"):
    mkdir(os.path.dirname(fpath))
    torch.save(state, fpath)
    if isbest:
        shutil.copy(fpath, os.path.join(os.path.dirname(fpath), 'model_best.pth.tar'))


def load_checkpoint(fpath):
    if os.path.isfile(fpath):
        checkpoint = torch.load(fpath)
        print(f"Loaded checkpoint: {fpath}")
        return checkpoint
    else:
        raise ValueError(f"No checkpoint found at path: {fpath}")


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # filter 'num_batches_tracked'
    missing_keys = [x for x in missing_keys
                    if not x.endswith('num_batches_tracked')]
    if len(missing_keys) > 0:
        print(f'[Warning] missing keys: {missing_keys}')
    if len(unused_pretrained_keys) > 0:
        print(f'[Warning] unused_pretrained_keys: {unused_pretrained_keys}')
    assert len(used_pretrained_keys) > 0, \
        'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters
    share common prefix 'module.' '''
    print(f"remove prefix '{prefix}'")
    def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_params(model, pretrained_path):
    print(f"load pretrained model from {pretrained_path}")
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(
            pretrained_path,
            map_location=lambda storage, loc: storage.cuda(device)
        )
    else:
        pretrained_dict = torch.load(
            pretrained_path,
            map_location=torch.device('cpu')
        )
    if "state_dict" in pretrained_dict:
        pretrained_state = remove_prefix(pretrained_dict['state_dict'],
                                         'module.')
    else:
        pretrained_state = remove_prefix(pretrained_dict, 'module.')

    assert check_keys(
        model, pretrained_state), 'load None from pretrained checkpoint'
    model.load_state_dict(pretrained_state, strict=False)
    if "dataset_info" in pretrained_dict:
        dataset_info = pretrained_dict['dataset_info']
        return model, dataset_info
    return model, None
