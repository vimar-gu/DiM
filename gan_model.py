# This code is based on the code provided in https://github.com/richardkxu/GANs-on-CIFAR10.

import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):

    def __init__(self, args):
        super(Discriminator, self).__init__()
        if args.data == 'mnist' or args.data == 'fashion':
            self.in_channels = 1
            self.conv1 = nn.Conv2d(1, 196, kernel_size=3, stride=1, padding=1)
            size = 28
        else:
            self.in_channels = 3
            self.conv1 = nn.Conv2d(3, 196, kernel_size=3, stride=1, padding=1)
            size = 32
        self.ln1 = nn.LayerNorm(normalized_shape=[196, size, size])
        self.lrelu1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1)
        self.ln2 = nn.LayerNorm(normalized_shape=[196, size // 2, size // 2])
        self.lrelu2 = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.ln3 = nn.LayerNorm(normalized_shape=[196, size // 2, size // 2])
        self.lrelu3 = nn.LeakyReLU()

        self.conv4 = nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1)
        self.ln4 = nn.LayerNorm(normalized_shape=[196, size // 4, size // 4])
        self.lrelu4 = nn.LeakyReLU()

        self.conv5 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.ln5 = nn.LayerNorm(normalized_shape=[196, size // 4, size // 4])
        self.lrelu5 = nn.LeakyReLU()

        self.conv6 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.ln6 = nn.LayerNorm(normalized_shape=[196, size // 4, size // 4])
        self.lrelu6 = nn.LeakyReLU()

        self.conv7 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.ln7 = nn.LayerNorm(normalized_shape=[196, size // 4, size // 4])
        self.lrelu7 = nn.LeakyReLU()

        self.conv8 = nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1)
        self.ln8 = nn.LayerNorm(normalized_shape=[196, 4, 4])
        self.lrelu8 = nn.LeakyReLU()

        self.pool = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        self.fc1 = nn.Linear(196, 1)
        self.fc10 = nn.Linear(196, 10)

    def forward(self, x, print_size=False):
        if print_size:
            print("input size: {}".format(x.size()))

        x = self.conv1(x)
        x = self.ln1(x)
        x = self.lrelu1(x)

        if print_size:
            print(x.size())

        x = self.conv2(x)
        x = self.ln2(x)
        x = self.lrelu2(x)

        if print_size:
            print(x.size())

        x = self.conv3(x)
        x = self.ln3(x)
        x = self.lrelu3(x)

        if print_size:
            print(x.size())

        x = self.conv4(x)
        x = self.ln4(x)
        x = self.lrelu4(x)

        if print_size:
            print(x.size())

        x = self.conv5(x)
        x = self.ln5(x)
        x = self.lrelu5(x)

        if print_size:
            print(x.size())

        x = self.conv6(x)
        x = self.ln6(x)
        x = self.lrelu6(x)

        if print_size:
            print(x.size())

        x = self.conv7(x)
        x = self.ln7(x)
        x = self.lrelu7(x)

        if print_size:
            print(x.size())

        x = self.conv8(x)
        x = self.ln8(x)
        x = self.lrelu8(x)

        if print_size:
            print(x.size())

        x = self.pool(x)

        if print_size:
            print(x.size())

        x = x.view(x.size(0), -1)

        if print_size:
            print(x.size())

        fc1_out = self.fc1(x)
        fc10_out = self.fc10(x)

        if print_size:
            print("fc1_out size: {}".format(fc1_out.size()))
            print("fc10_out size: {}".format(fc10_out.size()))

        return fc1_out, fc10_out


class Generator(nn.Module):

    def __init__(self, args):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 196*4*4)
        self.bn0 = nn.BatchNorm1d(196*4*4)
        self.relu0 = nn.ReLU()

        if args.data == 'mnist' or args.data == 'fashion':
            self.conv1 = nn.ConvTranspose2d(196, 196, kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = nn.ConvTranspose2d(196, 196, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(196)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(196)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(196)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(196)
        self.relu4 = nn.ReLU()

        #if args.data == 'mnist':
        #    self.conv5 = nn.ConvTranspose2d(196, 196, kernel_size=3, stride=2, padding=1)
        #else:
        self.conv5 = nn.ConvTranspose2d(196, 196, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(196)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(196)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.ConvTranspose2d(196, 196, kernel_size=4, stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(196)
        self.relu7 = nn.ReLU()

        if args.data == 'mnist' or args.data == 'fashion':
            self.conv8 = nn.Conv2d(196, 1, kernel_size=3, stride=1, padding=1)
        else:
            self.conv8 = nn.Conv2d(196, 3, kernel_size=3, stride=1, padding=1)
        # bn and relu are not applied after conv8

        self.tanh = nn.Tanh()

    def forward(self, x, print_size=False):
        if print_size:
            print("input size: {}".format(x.size()))

        x = self.fc1(x)
        x = self.bn0(x)
        x = self.relu0(x)

        if print_size:
            print(x.size())

        x = x.view(-1, 196, 4, 4)

        if print_size:
            print(x.size())

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        if print_size:
            print(x.size())

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        if print_size:
            print(x.size())

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        if print_size:
            print(x.size())

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        if print_size:
            print(x.size())

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        if print_size:
            print(x.size())

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)

        if print_size:
            print(x.size())

        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu7(x)

        if print_size:
            print(x.size())

        x = self.conv8(x)
        # bn and relu are not applied after conv8

        if print_size:
            print(x.size())

        x = self.tanh(x)

        if print_size:
            print("output (tanh) size: {}".format(x.size()))

        return x


if __name__ == '__main__':
    net1 = Discriminator()
    print(net1)
    x = torch.randn(10,3,32,32)
    fc1_out, fc10_out = net1(x, print_size=True)

    net2 = Generator()
    print(net2)
    x = torch.randn(10, 100)
    gen_out = net2(x, print_size=True)
