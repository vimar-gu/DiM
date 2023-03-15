import os
import sys
import time
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

import models.resnet as RN
import models.resnet_ap as RNAP
import models.convnet as CN
import models.densenet_cifar as DN
from gan_model import Generator, Discriminator
from utils import AverageMeter, accuracy, Normalize, Logger, rand_bbox
from augment import DiffAug


def str2bool(v):
    """Cast string to boolean
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_data(args):
    '''Obtain data
    '''
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    if args.data == 'cifar10':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.447), (0.202, 0.199, 0.201))
        ])
        trainset = datasets.CIFAR10(root=args.data_dir, train=True, download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root=args.data_dir, train=False, download=True,
                                   transform=transform_test)
    elif args.data == 'svhn':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.437, 0.444, 0.473), (0.198, 0.201, 0.197))
        ])
        trainset = datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                 split='train',
                                 download=True,
                                 transform=transform_train)
        testset = datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                split='test',
                                download=True,
                                transform=transform_test)
    elif args.data == 'fashion':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.286,), (0.353,))
        ])

        trainset = datasets.FashionMNIST(args.data_dir, train=True, download=True,
                                 transform=transform_train)
        testset = datasets.FashionMNIST(args.data_dir, train=False, download=True,
                                 transform=transform_train)
    elif args.data == 'mnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.131,), (0.308,))
        ])

        trainset = datasets.MNIST(args.data_dir, train=True, download=True,
                                 transform=transform_train)
        testset = datasets.MNIST(args.data_dir, train=False, download=True,
                                 transform=transform_train)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )

    return trainloader, testloader


def define_model(args, num_classes, e_model=None):
    '''Obtain model for training and validating
    '''
    if e_model:
        model = e_model
    else:
        model = args.match_model

    if args.data == 'mnist' or args.data == 'fashion':
        nch = 1
    else:
        nch = 3

    if model == 'convnet':
        return CN.ConvNet(num_classes, channel=nch)
    elif model == 'resnet10':
        return RN.ResNet(args.data, 10, num_classes, nch=nch)
    elif model == 'resnet18':
        return RN.ResNet(args.data, 18, num_classes, nch=nch)
    elif model == 'resnet34':
        return RN.ResNet(args.data, 34, num_classes, nch=nch)
    elif model == 'resnet50':
        return RN.ResNet(args.data, 50, num_classes, nch=nch)
    elif model == 'resnet101':
        return RN.ResNet(args.data, 101, num_classes, nch=nch)
    elif model == 'resnet10_ap':
        return RNAP.ResNetAP(args.data, 10, num_classes, nch=nch)
    elif model == 'resnet18_ap':
        return RNAP.ResNetAP(args.data, 18, num_classes, nch=nch)
    elif model == 'resnet34_ap':
        return RNAP.ResNetAP(args.data, 34, num_classes, nch=nch)
    elif model == 'resnet50_ap':
        return RNAP.ResNetAP(args.data, 50, num_classes, nch=nch)
    elif model == 'resnet101_ap':
        return RNAP.ResNetAP(args.data, 101, num_classes, nch=nch)
    elif model == 'densenet':
        return DN.densenet_cifar(num_classes)


def remove_aug(augtype, remove_aug):
    aug_list = []
    for aug in augtype.split("_"):
        if aug not in remove_aug.split("_"):
            aug_list.append(aug)

    return "_".join(aug_list)


def diffaug(args, device='cuda'):
    """Differentiable augmentation for condensation
    """
    aug_type = args.aug_type
    if args.data == 'cifar10':
        normalize = Normalize((0.491, 0.482, 0.447), (0.202, 0.199, 0.201), device='cuda')
    elif args.data == 'svhn':
        normalize = Normalize((0.437, 0.444, 0.473), (0.198, 0.201, 0.197), device='cuda')
    elif args.data == 'fashion':
        normalize = Normalize((0.286,), (0.353,), device='cuda')
    elif args.data == 'mnist':
        normalize = Normalize((0.131,), (0.308,), device='cuda')
    print("Augmentataion Matching: ", aug_type)
    augment = DiffAug(strategy=aug_type, batch=True)
    aug_batch = transforms.Compose([normalize, augment])

    if args.mixup_net == 'cut':
        aug_type = remove_aug(aug_type, 'cutout')
    print("Augmentataion Net update: ", aug_type)
    augment_rand = DiffAug(strategy=aug_type, batch=False)
    aug_rand = transforms.Compose([normalize, augment_rand])

    return aug_batch, aug_rand


def test(args, model, testloader, criterion):
    '''Calculate accuracy
    '''
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for batch_idx, (img, lab) in enumerate(testloader):
        img = img.cuda()
        lab = lab.cuda()

        with torch.no_grad():
            output = model(img)
        loss = criterion(output, lab)
        acc1, acc5 = accuracy(output.data, lab, topk=(1, 5))
        losses.update(loss.item(), output.shape[0])
        top1.update(acc1.item(), output.shape[0])
        top5.update(acc5.item(), output.shape[0])

    return top1.avg, top5.avg, losses.avg


def validate(args, generator, testloader, criterion, aug_rand):
    '''Validate the generator performance
    '''
    all_best_top1 = []
    all_best_top5 = []
    for e_model in args.eval_model:
        print('Evaluating {}'.format(e_model))
        model = define_model(args, args.num_classes, e_model).cuda()
        model.train()
        optim_model = torch.optim.SGD(model.parameters(), args.eval_lr, momentum=args.momentum,
                                      weight_decay=args.weight_decay)

        generator.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        best_top1 = 0.0
        best_top5 = 0.0
        for epoch_idx in range(args.epochs_eval):
            for batch_idx in range(10 * args.ipc // args.batch_size):
                # obtain pseudo samples with the generator
                if args.batch_size == args.num_classes:
                    lab_syn = torch.randperm(args.num_classes)
                else:
                    lab_syn = torch.randint(args.num_classes, (args.batch_size,))
                noise = torch.normal(0, 1, (args.batch_size, args.dim_noise))
                lab_onehot = torch.zeros((args.batch_size, args.num_classes))
                lab_onehot[torch.arange(args.batch_size), lab_syn] = 1
                noise[torch.arange(args.batch_size), :args.num_classes] = lab_onehot[torch.arange(args.batch_size)]
                noise = noise.cuda()
                lab_syn = lab_syn.cuda()

                with torch.no_grad():
                    img_syn = generator(noise)
                    img_syn = aug_rand((img_syn + 1.0) / 2.0)

                if np.random.rand(1) < args.mix_p and args.mixup_net == 'cut':
                    lam = np.random.beta(args.beta, args.beta)
                    rand_index = torch.randperm(len(img_syn)).cuda()

                    lab_syn_b = lab_syn[rand_index]
                    bbx1, bby1, bbx2, bby2 = rand_bbox(img_syn.size(), lam)
                    img_syn[:, :, bbx1:bbx2, bby1:bby2] = img_syn[rand_index, :, bbx1:bbx2, bby1:bby2]
                    ratio = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img_syn.size()[-1] * img_syn.size()[-2]))

                    output = model(img_syn)
                    loss = criterion(output, lab_syn) * ratio + criterion(output, lab_syn_b) * (1. - ratio)
                else:
                    output = model(img_syn)
                    loss = criterion(output, lab_syn)

                acc1, acc5 = accuracy(output.data, lab_syn, topk=(1, 5))

                losses.update(loss.item(), img_syn.shape[0])
                top1.update(acc1.item(), img_syn.shape[0])
                top5.update(acc5.item(), img_syn.shape[0])

                optim_model.zero_grad()
                loss.backward()
                optim_model.step()

            if (epoch_idx + 1) % args.test_interval == 0:
                test_top1, test_top5, test_loss = test(args, model, testloader, criterion)
                print('[Test Epoch {}] Top1: {:.3f} Top5: {:.3f}'.format(epoch_idx + 1, test_top1, test_top5))
                if test_top1 > best_top1:
                    best_top1 = test_top1
                    best_top5 = test_top5

        all_best_top1.append(best_top1)
        all_best_top5.append(best_top5)

    return all_best_top1, all_best_top5


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ipc', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--epochs-eval', type=int, default=1000)
    parser.add_argument('--epochs-match', type=int, default=100)
    parser.add_argument('--epochs-match-train', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eval-lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--match-coeff', type=float, default=0.001)
    parser.add_argument('--match-model', type=str, default='convnet')
    parser.add_argument('--eval-model', type=str, nargs='+', default=['convnet'])
    parser.add_argument('--dim-noise', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--eval-interval', type=int, default=10)
    parser.add_argument('--test-interval', type=int, default=200)
    parser.add_argument('--fix-disc', action='store_true', default=False)

    parser.add_argument('--data', type=str, default='cifar10')
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--output-dir', type=str, default='./results/')
    parser.add_argument('--logs-dir', type=str, default='./logs/')
    parser.add_argument('--pretrain-weight', type=str, default='./logs/')
    parser.add_argument('--match-aug', action='store_true', default=False)
    parser.add_argument('--aug-type', type=str, default='color_crop_cutout')
    parser.add_argument('--mixup-net', type=str, default='cut')
    parser.add_argument('--metric', type=str, default='l1')
    parser.add_argument('--bias', type=str2bool, default=False)
    parser.add_argument('--fc', type=str2bool, default=False)
    parser.add_argument('--mix-p', type=float, default=-1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--tag', type=str, default='test')
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.output_dir = args.output_dir + args.tag
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.output_dir + '/outputs'):
        os.makedirs(args.output_dir + '/outputs')

    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    args.logs_dir = args.logs_dir + args.tag
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    sys.stdout = Logger(os.path.join(args.logs_dir, 'logs.txt'))

    print(args)

    trainloader, testloader = load_data(args)

    generator = Generator(args).cuda()
    generator.load_state_dict(torch.load(args.pretrain_weight)['generator'])

    criterion = nn.CrossEntropyLoss()

    aug, aug_rand = diffaug(args)

    top1s, top5s = validate(args, generator, testloader, criterion, aug_rand)
    for e_idx, e_model in enumerate(args.eval_model):
        print('Evaluation for {}: Top1: {:.3f}, Top5: {:.3f}'.format(e_model, top1s[e_idx], top5s[e_idx]))
