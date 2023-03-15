import os
import sys
import time
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

import models.resnet as RN
import models.convnet as CN
import models.resnet_ap as RNAP
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
    '''Obtain model for training, validating and matching
    With no 'e_model' specified, it returns a random model
    '''
    if e_model:
        model = e_model
    else:
        model_pool = ['convnet', 'resnet10', 'resnet18',
                      'resnet10_ap', 'resnet18_ap']
        model = random.choice(model_pool)
        print('Random model: {}'.format(model))

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


def calc_gradient_penalty(args, discriminator, img_real, img_syn):
    ''' Gradient penalty from Wasserstein GAN
    '''
    LAMBDA = 10
    n_size = img_real.shape[-1]
    batch_size = img_real.shape[0]
    n_channels = img_real.shape[1]

    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(img_real.nelement() / batch_size)).contiguous()
    alpha = alpha.view(batch_size, n_channels, n_size, n_size)
    alpha = alpha.cuda()

    img_syn = img_syn.view(batch_size, n_channels, n_size, n_size)
    interpolates = alpha * img_real.detach() + ((1 - alpha) * img_syn.detach())

    interpolates = interpolates.cuda()
    interpolates.requires_grad_(True)

    disc_interpolates, _ = discriminator(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def dist(x, y, method='mse'):
    """Distance objectives
    """
    if method == 'mse':
        dist_ = (x - y).pow(2).sum()
    elif method == 'l1':
        dist_ = (x - y).abs().sum()
    elif method == 'l1_mean':
        n_b = x.shape[0]
        dist_ = (x - y).abs().reshape(n_b, -1).mean(-1).sum()
    elif method == 'cos':
        x = x.reshape(x.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        dist_ = torch.sum(1 - torch.sum(x * y, dim=-1) /
                          (torch.norm(x, dim=-1) * torch.norm(y, dim=-1) + 1e-6))

    return dist_


def add_loss(loss_sum, loss):
    if loss_sum == None:
        return loss
    else:
        return loss_sum + loss


def matchloss(args, img_real, img_syn, lab_real, lab_syn, model):
    """Matching losses (feature or gradient)
    """
    loss = None

    if 'feat' in args.match:
        with torch.no_grad():
            feat_tg = model.get_feature(img_real, args.idx_from, args.idx_to)
        feat = model.get_feature(img_syn, args.idx_from, args.idx_to)

        for i in range(len(feat)):
            loss = add_loss(loss, dist(feat_tg[i].mean(0), feat[i].mean(0), method=args.metric) * 0.001)

    elif 'grad' in args.match:
        criterion = nn.CrossEntropyLoss()

        output_real = model(img_real)
        loss_real = criterion(output_real, lab_real)
        g_real = torch.autograd.grad(loss_real, model.parameters())
        g_real = list((g.detach() for g in g_real))

        output_syn = model(img_syn)
        loss_syn = criterion(output_syn, lab_syn)
        g_syn = torch.autograd.grad(loss_syn, model.parameters(), create_graph=True)

        for i in range(len(g_real)):
            if (len(g_real[i].shape) == 1) and not args.bias:  # bias, normliazation
                continue
            if (len(g_real[i].shape) == 2) and not args.fc:
                continue

            loss = add_loss(loss, dist(g_real[i], g_syn[i], method=args.metric) * 0.001)

    elif 'logit' in args.match:
        output_real = F.log_softmax(model(img_real), dim=1)
        output_syn = F.log_softmax(model(img_syn), dim=1)
        loss = add_loss(loss, ((output_real - output_syn) ** 2).mean() * 0.01)

    return loss


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


def train(args, epoch, generator, discriminator, optim_g, optim_d, trainloader, criterion, aug, aug_rand):
    '''The main training function for the generator
    '''
    generator.train()
    gen_losses = AverageMeter()
    disc_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model = define_model(args, args.num_classes).cuda()
    model.train()
    optim_model = torch.optim.SGD(model.parameters(), args.eval_lr, momentum=args.momentum,
                                  weight_decay=args.weight_decay)

    for batch_idx, (img_real, lab_real) in enumerate(trainloader):
        img_real = img_real.cuda()
        lab_real = lab_real.cuda()

        # train the generator
        discriminator.eval()
        optim_g.zero_grad()

        # obtain the noise with one-hot class labels
        noise = torch.normal(0, 1, (args.batch_size, args.dim_noise))
        lab_onehot = torch.zeros((args.batch_size, args.num_classes))
        lab_onehot[torch.arange(args.batch_size), lab_real] = 1
        noise[torch.arange(args.batch_size), :args.num_classes] = lab_onehot[torch.arange(args.batch_size)]
        noise = noise.cuda()

        img_syn = generator(noise)
        gen_source, gen_class = discriminator(img_syn)
        gen_source = gen_source.mean()
        gen_class = criterion(gen_class, lab_real)

        gen_loss = - gen_source + gen_class

        # update the match model to obtain more various matching signals
        train_match_model(args, model, optim_model, trainloader, criterion, aug_rand)
        # calculate the matching loss
        if args.match_aug:
            img_aug = aug(torch.cat([img_real, img_syn]))
            match_loss = matchloss(args, img_aug[:args.batch_size], img_aug[args.batch_size:], lab_real, lab_real, model)# * args.match_coeff
        else:
            match_loss = matchloss(args, img_real, img_syn, lab_real, lab_real, model)# * args.match_coeff
        gen_loss = gen_loss + match_loss

        gen_loss.backward()
        optim_g.step()

        # train the discriminator
        discriminator.train()
        optim_d.zero_grad()
        lab_syn = torch.randint(args.num_classes, (args.batch_size,))
        noise = torch.normal(0, 1, (args.batch_size, args.dim_noise))
        lab_onehot = torch.zeros((args.batch_size, args.num_classes))
        lab_onehot[torch.arange(args.batch_size), lab_syn] = 1
        noise[torch.arange(args.batch_size), :args.num_classes] = lab_onehot[torch.arange(args.batch_size)]
        noise = noise.cuda()
        lab_syn = lab_syn.cuda()

        with torch.no_grad():
            img_syn = generator(noise)

        disc_fake_source, disc_fake_class = discriminator(img_syn)
        disc_fake_source = disc_fake_source.mean()
        disc_fake_class = criterion(disc_fake_class, lab_syn)

        disc_real_source, disc_real_class = discriminator(img_real)
        acc1, acc5 = accuracy(disc_real_class.data, lab_real, topk=(1, 5))
        disc_real_source = disc_real_source.mean()
        disc_real_class = criterion(disc_real_class, lab_real)

        gradient_penalty = calc_gradient_penalty(args, discriminator, img_real, img_syn)

        disc_loss = disc_fake_source - disc_real_source + disc_fake_class + disc_real_class + gradient_penalty
        disc_loss.backward()
        optim_d.step()

        gen_losses.update(gen_loss.item())
        disc_losses.update(disc_loss.item())
        top1.update(acc1.item())
        top5.update(acc5.item())

        if (batch_idx + 1) % args.print_freq == 0:
            print('[Train Epoch {} Iter {}] G Loss: {:.3f}({:.3f}) D Loss: {:.3f}({:.3f}) D Acc: {:.3f}({:.3f})'.format(
                epoch, batch_idx + 1, gen_losses.val, gen_losses.avg, disc_losses.val, disc_losses.avg, top1.val, top1.avg)
            )


def train_match_model(args, model, optim_model, trainloader, criterion, aug_rand):
    '''The training function for the match model
    '''
    for batch_idx, (img, lab) in enumerate(trainloader):
        if batch_idx == args.epochs_match_train:
            break

        img = img.cuda()
        lab = lab.cuda()

        output = model(aug_rand(img))
        loss = criterion(output, lab)

        optim_model.zero_grad()
        loss.backward()
        optim_model.step()


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
            for batch_idx in range(10 * args.ipc // args.batch_size + 1):
                # obtain pseudo samples with the generator
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
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--epochs-eval', type=int, default=1000)
    parser.add_argument('--epochs-match', type=int, default=100)
    parser.add_argument('--epochs-match-train', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--eval-lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--match-coeff', type=float, default=0.001)
    parser.add_argument('--match-model', type=str, default='convnet')
    parser.add_argument('--match', type=str, default='grad')
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
    parser.add_argument('--weight', type=str, default='')
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
    discriminator = Discriminator(args).cuda()

    optim_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0, 0.9))
    optim_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0, 0.9))

    model_dict = torch.load(args.weight)
    generator.load_state_dict(model_dict['generator'])
    discriminator.load_state_dict(model_dict['discriminator'])
    optim_g.load_state_dict(model_dict['optim_g'])
    optim_d.load_state_dict(model_dict['optim_d'])
    for g in optim_g.param_groups:
        g['lr'] = args.lr
    for g in optim_d.param_groups:
        g['lr'] = args.lr
    criterion = nn.CrossEntropyLoss()

    aug, aug_rand = diffaug(args)

    best_top1s = np.zeros((len(args.eval_model),))
    best_top5s = np.zeros((len(args.eval_model),))
    best_epochs = np.zeros((len(args.eval_model),))
    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()
        train(args, epoch, generator, discriminator, optim_g, optim_d, trainloader, criterion, aug, aug_rand)

        # save image for visualization
        generator.eval()
        test_label = torch.tensor(list(range(10)) * 10)
        test_noise = torch.normal(0, 1, (100, 100))
        lab_onehot = torch.zeros((100, args.num_classes))
        lab_onehot[torch.arange(100), test_label] = 1
        test_noise[torch.arange(100), :args.num_classes] = lab_onehot[torch.arange(100)]
        test_noise = test_noise.cuda()
        test_img_syn = (generator(test_noise) + 1.0) / 2.0
        test_img_syn = make_grid(test_img_syn, nrow=10)
        save_image(test_img_syn, os.path.join(args.output_dir, 'outputs/img_{}.png'.format(epoch)))
        generator.train()

        if (epoch + 1) % args.eval_interval == 0:
            top1s, top5s = validate(args, generator, testloader, criterion, aug_rand)
            for e_idx, e_model in enumerate(args.eval_model):
                if top1s[e_idx] > best_top1s[e_idx]:
                    best_top1s[e_idx] = top1s[e_idx]
                    best_top5s[e_idx] = top5s[e_idx]
                    best_epochs[e_idx] = epoch

                    model_dict = {'generator': generator.state_dict(),
                                  'discriminator': discriminator.state_dict(),
                                  'optim_g': optim_g.state_dict(),
                                  'optim_d': optim_d.state_dict()}
                    torch.save(
                        model_dict,
                        os.path.join(args.output_dir, 'model_dict_{}.pth'.format(e_model)))
                    print('Save model for {}'.format(e_model))

                print('Current Best Epoch for {}: {}, Top1: {:.3f}, Top5: {:.3f}'.format(e_model, best_epochs[e_idx], best_top1s[e_idx], best_top5s[e_idx]))

