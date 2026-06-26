import torchvision
import os
from PIL import Image

# CIFAR-100
print("=== Downloading CIFAR-100 ===")
train_ds = torchvision.datasets.CIFAR100(root='/tmp/cifar100', train=True, download=True)
test_ds = torchvision.datasets.CIFAR100(root='/tmp/cifar100', train=False, download=True)

splits = {
    'train800': (train_ds, range(800)),
    'val200': (train_ds, range(800, 1000)),
    'train800val200': (train_ds, range(1000)),
    'test': (test_ds, range(len(test_ds))),
}

base = 'data/vtab-1k/cifar'
for split_name, (ds, indices) in splits.items():
    out_dir = os.path.join(base, 'images', split_name)
    os.makedirs(out_dir, exist_ok=True)
    txt_lines = []
    for i, idx in enumerate(indices):
        img, label = ds[idx]
        img = img.resize((224, 224))
        img.save(os.path.join(out_dir, f'{i:06d}.jpg'))
        txt_lines.append(f'images/{split_name}/{i:06d}.jpg {label}\n')
    with open(os.path.join(base, f'{split_name}.txt'), 'w') as f:
        f.writelines(txt_lines)
    print(f'  {split_name}: {len(indices)} images saved')

# Caltech-101
print("\n=== Downloading Caltech-101 ===")
cal_ds = torchvision.datasets.Caltech101(root='/tmp/caltech101', download=True)
base = 'data/vtab-1k/caltech101'
cal_splits = {
    'train800': range(800),
    'val200': range(800, 1000),
    'train800val200': range(1000),
    'test': range(1000, len(cal_ds)),
}
for split_name, indices in cal_splits.items():
    out_dir = os.path.join(base, 'images', split_name)
    os.makedirs(out_dir, exist_ok=True)
    txt_lines = []
    for i, idx in enumerate(indices):
        img, label = cal_ds[idx]
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((224, 224))
        img.save(os.path.join(out_dir, f'{i:06d}.jpg'))
        txt_lines.append(f'images/{split_name}/{i:06d}.jpg {label}\n')
    with open(os.path.join(base, f'{split_name}.txt'), 'w') as f:
        f.writelines(txt_lines)
    print(f'  {split_name}: {len(list(indices))} images saved')

# DTD
print("\n=== Downloading DTD ===")
dtd_train = torchvision.datasets.DTD(root='/tmp/dtd', split='train', download=True)
dtd_test = torchvision.datasets.DTD(root='/tmp/dtd', split='test', download=True)
base = 'data/vtab-1k/dtd'
dtd_splits = {
    'train800': (dtd_train, range(min(800, len(dtd_train)))),
    'val200': (dtd_train, range(800, min(1000, len(dtd_train)))),
    'train800val200': (dtd_train, range(min(1000, len(dtd_train)))),
    'test': (dtd_test, range(len(dtd_test))),
}
for split_name, (ds, indices) in dtd_splits.items():
    out_dir = os.path.join(base, 'images', split_name)
    os.makedirs(out_dir, exist_ok=True)
    txt_lines = []
    for i, idx in enumerate(indices):
        img, label = ds[idx]
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((224, 224))
        img.save(os.path.join(out_dir, f'{i:06d}.jpg'))
        txt_lines.append(f'images/{split_name}/{i:06d}.jpg {label}\n')
    with open(os.path.join(base, f'{split_name}.txt'), 'w') as f:
        f.writelines(txt_lines)
    print(f'  {split_name}: {len(list(indices))} images saved')

print("\nDone!")
