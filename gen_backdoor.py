#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# author：fmy
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
import numpy as np

class Generator(nn.Module):
    def __init__(self, param):
        super(Generator, self).__init__()

        self.in_size = in_size = param['in_size']
        self.skip_size = skip_size = in_size // 4  # NOTE: skip connections improve model stability
        self.out_size = out_size = param['out_size']
        self.hidden_size = hidden_size = param['hidden_size']
        self.fc1 = nn.Linear(skip_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size + skip_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size + skip_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size + skip_size, out_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)

    def forward(self, z):
        h = self.skip_size
        x = self.fc1(z[:, :h])
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc2(torch.cat([x, z[:, h:2 * h]], dim=1))
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc3(torch.cat([x, z[:, 2 * h:3 * h]], dim=1))
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc4(torch.cat([x, z[:, 3 * h:4 * h]], dim=1))
        x = torch.sigmoid(x)
        return x

    def gen_noise(self, num):
        return torch.rand(num, self.in_size)


class Mine(nn.Module):
    def __init__(self, param):
        super().__init__()
        x_size = param['in_size']
        y_size = param['out_size']
        self.hidden_size = hidden_size = param['hidden_size']

        self.fc1_x = nn.Linear(x_size, hidden_size, bias=False)
        self.fc1_y = nn.Linear(y_size, hidden_size, bias=False)
        self.fc1_bias = nn.Parameter(torch.zeros(hidden_size))
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        # moving average
        self.ma_et = None
        self.ma_rate = 0.001
        self.conv = nn.Sequential(
            nn.Conv2d(3, hidden_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),

            nn.Conv2d(hidden_size, 2 * hidden_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 2),
            nn.ReLU(),

            nn.Conv2d(hidden_size * 2, hidden_size, 4, 1, 0, bias=False),
        )
        self.fc1_y_after_conv = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, y):
        x = self.fc1_x(x)
        y = self.fc1_y(y)
        x = F.leaky_relu(x + y + self.fc1_bias, 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return x

    def mi(self, x, x1, y):
        x = self.forward(x, y)
        x1 = self.forward(x1, y)
        return x.mean() - torch.log(torch.exp(x1).mean() + 1e-8)

    def mi_loss(self, x, x1, y):
        x = self.forward(x, y)
        x1 = self.forward(x1, y)
        et = torch.exp(x1).mean()
        if self.ma_et is None:
            self.ma_et = et.detach().item()
        self.ma_et += self.ma_rate * (et.detach().item() - self.ma_et)
        return x.mean() - torch.log(et + 1e-8) * et.detach() / self.ma_et


class GM:
    def __init__(self, victimized_model, attack_method, search_param, ratio):
        self.param = search_param
        self.victimized_model = victimized_model
        self.attack_method = attack_method
        if not ('type' in self.param):  # set default type
            self.param['type'] = 'distribution'
        self.path = "./generate_trigger"
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        self.filename = os.path.join(self.path, 'GM.ckpt')
        self.target = self.attack_method.target_class
        self.in_size = 64
        GM_param = {'in_size': self.in_size, 'out_size': 27, 'hidden_size': 2048}
        self.print_interval = 10
        self.lr = 0.0002
        self.num_epochs = 10
        self.bs = 128 # batch size
        self.trigger_width = 3
        self.trigger_height = 3
        self.G = Generator(GM_param).cuda()
        self.M = Mine(GM_param).cuda()
        if 'round' in self.param:
            self.round = search_param['round']

        org_dataset = self.victimized_model.dataset.get_org_dataset(mode="valid")
        self.indices = torch.randperm(10000)
        self.holded_ratio = ratio # equal 5% of training set （0.25）
        self.holded_data = torch.utils.data.Subset(org_dataset, self.indices[:int(10000 * self.holded_ratio)])
        self.test_data = torch.utils.data.Subset(org_dataset, self.indices[int(10000 * self.holded_ratio):])
        self.holded_dataloader = torch.utils.data.DataLoader(self.holded_data, batch_size=self.bs)
        self.test_dataloader = torch.utils.data.DataLoader(self.test_data, batch_size=self.bs)

        self.logger = logging.getLogger("generator")
        self.logger.setLevel(logging.INFO)
        self.handler = logging.FileHandler("generator_training_log", "a")
        self.handler.setLevel(logging.INFO)
        self.logger.addHandler(self.handler)

    def add_trigger(self, trigger, data, args):
        _, _, th, tw = trigger.size()
        _, _, dh, dw = data.size()
        if args == 'corner':
            data[:, :, -th:, -tw:] = trigger
        elif args == 'random':
            x = int(np.random.rand() * (dh - th))
            y = int(np.random.rand() * (dw - tw))
            data[:, :, x:x + th, y:y + tw] = trigger
        else:
            raise Exception('unknown trigger args')
        return data

    def transform(self, data, stats):
        assert data.dim() == 4
        return (data - stats['mean']) / stats['std']

    def dataset_stats(self, name, is_tensor=True):
        if name == 'cifar10':
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
        elif name == 'cifar100':
            mean = [0.5071, 0.4867, 0.4408]
            std = [0.2675, 0.2565, 0.2761]
        if is_tensor:
            return {'mean': torch.tensor(mean).view(1, 3, 1, 1).cuda(),
                    'std': torch.tensor(std).view(1, 3, 1, 1).cuda()}
        else:
            return {'mean': mean, 'std': std}

    def search(self):
        self.logger.info("start...")
        self.logger.info("hinge loss\tentropy loss\tsoftmax loss\tASR")
        alpha = self.param['alpha']
        beta = self.param['beta']
        beta_ = torch.tensor(beta).cuda()
        target = self.target
        # target_ = torch.tensor(target).cuda()
        # loader = dataloader(self.train_data, bs=self.bs)
        G = self.G
        M = self.M
        B = self.victimized_model
        G_opt = torch.optim.Adam(G.parameters(), lr=self.lr, betas=(0.5, 0.999))
        M_opt = torch.optim.Adam(M.parameters(), lr=self.lr, betas=(0.5, 0.999))

        G.train()
        M.train()
        B.eval()
        for epoch in range(self.num_epochs):
            log_hinge, log_softmax, log_entropy, log_count, non_target_total, non_target_correct = 0, 0, 0, 0, 0, 0
            for data, label in self.holded_dataloader:
                # train Generator
                batch_size = label.size()[0]
                z = G.gen_noise(batch_size).cuda()
                z1 = G.gen_noise(batch_size).cuda()
                trigger_noise = torch.randn(batch_size, G.out_size).cuda() / 10
                trigger = G(z)
                data = data.clone().cuda()
                label = label.clone().cuda()

                data = self.add_trigger(self.transform((trigger + trigger_noise).view(-1, 3, self.trigger_height, self.trigger_width),
                                        self.dataset_stats(self.victimized_model.dataset.name)), data, 'random')
                logit = B(data)

                hinge_loss = torch.mean(torch.min(F.softmax(logit, dim=1)[:, target], beta_))
                entropy = M.mi(z, z1, trigger)
                G_loss = -hinge_loss - alpha * entropy
                G.zero_grad()
                G_loss.backward()
                G_opt.step()

                # train Mine
                z = torch.rand(batch_size, G.in_size).cuda()
                z1 = torch.rand(batch_size, G.in_size).cuda()
                trigger = G(z)

                M_loss = -M.mi_loss(z, z1, trigger)
                M_opt.zero_grad()
                M_loss.backward()
                M_opt.step()

                log_hinge += hinge_loss.item()
                log_entropy += entropy.item()
                log_softmax += (F.softmax(logit, dim=1)[:, target]).mean().item()
                log_count += 1

                predicted = torch.argmax(logit, dim=1)
                non_target_total += torch.sum(~label.eq(target)).item()
                non_target_correct += (predicted.eq(target) * (~label.eq(target))).sum().item()
            self.logger.info("{}\t{}\t{}\t{}".format(
                log_hinge / log_count, log_entropy / log_count, log_softmax / log_count,
                non_target_correct / non_target_total
            ))

    def load(self):
        ckpt = torch.load(self.filename)
        self.G.load_state_dict(ckpt['G'])
        self.M.load_state_dict(ckpt['M'])
        print("model loaded")

    def save(self):
        state = {
            'G': self.G.state_dict(),
            'M': self.M.state_dict(),
        }
        torch.save(state, self.filename)
        print("model saved")

    def exist(self):
        return os.path.isfile(self.filename)

    def sample(self, type_='cuda'):
        if self.param['type'] == 'distribution':
            G = self.G
            G.eval()
            z = G.gen_noise(self.bs).cuda()
            x = G(z)
            G.train()

        if type_ == 'cuda':
            return x
        else:  # type = numpy
            return x.cpu().detach().numpy()

    def try_load(self):
        # if self.exist():
        #     self.load()
        # else:
        self.search()
        self.save()


# def try_load(models):
#     for model in models:
#         model.try_load()
