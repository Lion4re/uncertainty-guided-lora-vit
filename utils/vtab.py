import torchvision.transforms as transforms
import torch, os
import torch.nn.functional as F
from .h5dataset import ImageHdf5Data

_vtab_class_num_dict = {'cifar':                100,
                        'caltech101':           102,
                        'dtd':                  47,
                        'oxford_flowers102':    102,
                        'oxford_iiit_pet':      37,
                        'svhn':                 10,
                        'sun397':               397,
                        'patch_camelyon':       2,
                        'eurosat':              10,
                        'resisc45':             45,
                        'diabetic_retinopathy': 5,
                        'clevr_count':          8,
                        'clevr_dist':           6,
                        'dmlab':                6,
                        'kitti':                4,
                        'dsprites_loc':         16,
                        'dsprites_ori':         16,
                        'smallnorb_azi':        18,
                        'smallnorb_ele':        9}

_VALID_EVAL_SHIFTS = {'clean', 'gaussian_noise', 'gaussian_blur', 'brightness', 'contrast', 'cutout'}


def apply_controlled_shift_tensor(x, eval_shift='clean', shift_severity=0):
    if eval_shift not in _VALID_EVAL_SHIFTS:
        raise ValueError(f"Unknown eval_shift: {eval_shift}")
    severity = int(shift_severity)
    if eval_shift == 'clean' or severity <= 0:
        return x
    if eval_shift == 'gaussian_noise':
        std = {1: 0.05, 2: 0.10, 3: 0.20}.get(severity, 0.20)
        return (x + torch.randn_like(x) * std).clamp(0.0, 1.0)
    if eval_shift == 'gaussian_blur':
        kernel = {1: 3, 2: 5, 3: 7}.get(severity, 7)
        pad = kernel // 2
        if x.ndim == 3:
            return F.avg_pool2d(x.unsqueeze(0), kernel_size=kernel, stride=1, padding=pad).squeeze(0)
        return F.avg_pool2d(x, kernel_size=kernel, stride=1, padding=pad)
    if eval_shift == 'brightness':
        factor = {1: 0.85, 2: 0.70, 3: 0.55}.get(severity, 0.55)
        return (x * factor).clamp(0.0, 1.0)
    if eval_shift == 'contrast':
        factor = {1: 0.75, 2: 0.55, 3: 0.35}.get(severity, 0.35)
        mean = x.mean(dim=(-2, -1), keepdim=True)
        return ((x - mean) * factor + mean).clamp(0.0, 1.0)
    if eval_shift == 'cutout':
        out = x.clone()
        size = {1: 32, 2: 56, 3: 80}.get(severity, 80)
        h, w = out.shape[-2:]
        top = max(0, (h - size) // 2)
        left = max(0, (w - size) // 2)
        out[..., top:top + size, left:left + size] = 0.0
        return out
    return x


def build_vtab_eval_transform(resize=224, eval_shift='clean', shift_severity=0):
    return transforms.Compose([
        transforms.Resize((resize, resize), interpolation=3),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: apply_controlled_shift_tensor(x, eval_shift, shift_severity)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def get_vtab_data(name, evaluate=False, resize=224, batch_size=64, num_workers=8, is_hdf5=True,
                  eval_shift='clean', shift_severity=0):
    if name in _vtab_class_num_dict:
        root = os.path.join('data', 'vtab-1k', name)
        transform = build_vtab_eval_transform(
            resize=resize,
            eval_shift=eval_shift,
            shift_severity=shift_severity,
        )

        image_root = os.path.join(root, 'images.hdf5' if is_hdf5 else 'images')

        def flist_reader(flist):
            imlist = []
            with open(flist, 'r') as rf:
                for line in rf.readlines():
                    impath, imlabel = line.strip().rsplit(' ', 1)
                    impath = impath.split('/', 1)[-1]
                    imlist.append((impath, int(imlabel)))
            return imlist

        if evaluate:
            train_loader = torch.utils.data.DataLoader(
                ImageHdf5Data(root=image_root, flist=os.path.join(root, "train800val200.txt"), transform=transform,
                              flist_reader=flist_reader, return_index=True, is_hdf5=is_hdf5),
                batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)

            val_loader = torch.utils.data.DataLoader(
                ImageHdf5Data(root=image_root, flist=os.path.join(root, "test.txt"), transform=transform,
                              flist_reader=flist_reader, is_hdf5=is_hdf5),
                batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=True)
        else:
            train_loader = torch.utils.data.DataLoader(
                ImageHdf5Data(root=image_root, flist=os.path.join(root, "train800.txt"), transform=transform,
                              flist_reader=flist_reader, return_index=True, is_hdf5=is_hdf5),
                batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)

            val_loader = torch.utils.data.DataLoader(
                ImageHdf5Data(root=image_root, flist=os.path.join(root, "val200.txt"), transform=transform,
                              flist_reader=flist_reader, is_hdf5=is_hdf5),
                batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=True)
        return train_loader, val_loader
    else:
        raise NotImplementedError(f'VTAB-1K does not have dataset: {name}')


def get_vtab_classes_num(dataset_name):
    return _vtab_class_num_dict[dataset_name]
