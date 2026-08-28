from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms(train=True):
    ops = [transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])]
    if train:
        ops = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)] + ops
    return transforms.Compose(ops)


def get_dataloaders(data_dir, batch_size=64, num_workers=0):
    train = datasets.CIFAR10(data_dir, train=True, download=True, transform=get_transforms(True))
    val = datasets.CIFAR10(data_dir, train=False, download=True, transform=get_transforms(False))
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    )
