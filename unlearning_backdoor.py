#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# author：fmy
import copy
import gen_backdoor
import torch
import torch.optim as optim


class UnlearningBackdoor:
    name: str = 'unlearning_backdoor'

    def __init__(self, model, dataset, mark, attack, ratio, alpha):
        super(UnlearningBackdoor, self).__init__()
        self.victimized_model = model
        self.backup = copy.deepcopy(self.victimized_model)
        self.dataset = dataset
        self.mark = mark
        self.attack = attack
        self.alpha = alpha
        search_param = {'search_type': 'distribution', 'alpha': 0.1, 'beta': 0.5, 'target': self.attack.target_class} # beta 0.8, 0.9
        self.GM_model = gen_backdoor_mnist.GM(self.victimized_model, self.attack, search_param, ratio)
        self.ratio = ratio
        self.random_pos = self.attack.mark.random_pos
        self.holded_dataloader = self.GM_model.holded_dataloader
        self.test_dataloader = self.GM_model.test_dataloader

    def run(self):
        self.unlearning_process()

    @torch.no_grad()
    def assess_model(self, model, dataloader):
        model.eval()
        cleaned_acc, asr, num1, num2 = 0, 0, 0, 0
        with torch.no_grad():
            for data, label in dataloader:
                data, label = data.cuda(), label.cuda()
                posioned_data = self.mark.add_mark(data.clone())
                indices = (label != self.attack.target_class).nonzero().flatten()
                cleaned_acc += (model(data).argmax(dim=1) == label).sum().item()
                asr += (model(posioned_data[indices, :, :, :]).argmax(dim=1) == self.attack.target_class).sum().item()
                num1 += label.size()[0]
                num2 += indices.size()[0]
        return cleaned_acc / num1, asr / num2

    def unlearning_process(self):

        optimizer = optim.SGD(self.victimized_model.parameters(), lr=1e-4, momentum=0.9, weight_decay=0)
        loss_func = torch.nn.CrossEntropyLoss()

        cleaned_acc, posioned_acc = self.assess_model(self.victimized_model, self.test_dataloader)
        print("{attack_method}\t{defense_method}\t{cleaned_acc:.2f}\t{posioned_acc:.2f}\t{ratio}\t{alpha}".
                             format(attack_method=self.attack.name, defense_method=self.name,
                                    epoch=0, cleaned_acc=cleaned_acc * 100., posioned_acc=posioned_acc * 100.,
                                    ratio = self.ratio, alpha = self.alpha))

        for param1, param2 in self.victimized_model.parameters(), self.backup.parameters():
            param1.requires_grad_(True)
            param2.requires_grad_(False) # backup of original model

        # following setting of NAD for fair comparison, the default epoch is set to 10
        for epoch in range(10):

            self.victimized_model.train()
            for data, label in self.holded_dataloader:
                optimizer.zero_grad()

                # adding clean loss
                z = self.GM_model.G.gen_noise(label.size()[0]).cuda()
                trigger = self.GM_model.G(z)
                data, label = data.cuda(), label.cuda()
                cleaned_loss = loss_func(self.victimized_model(data), label)
                
                # unlearning backdoor
                posioned_data = data.detach().clone()
                posioned_data = self.GM_model.add_trigger(self.GM_model.transform(
                    (trigger).view(-1, 1, self.GM_model.trigger_height, self.GM_model.trigger_width),
                    self.GM_model.dataset_stats(self.victimized_model.dataset.name)), posioned_data, 'random')
                posioned_loss = -loss_func(self.victimized_model(posioned_data),
                                           label.detach().clone().fill_(self.attack.target_class))

                loss = self.alpha * cleaned_loss + posioned_loss

                # adding regularity item of coefficients for maintaining performance of model
                for param1, param2 in zip(self.victimized_model, self.backup):
                    loss += 1e-6 * torch.abs(param1 - param2).sum()

                loss.backward()
                optimizer.step()

                cleaned_acc, posioned_acc = self.assess_model(self.victimized_model, self.test_dataloader)

                print("\t{attack_method}\t{defense_method}\t{cleaned_acc:.2f}\t{posioned_acc:.2f}\t{ratio}\t{alpha}".
                                     format(attack_method=self.attack.name, defense_method=self.name,
                                            cleaned_acc=cleaned_acc * 100., posioned_acc=posioned_acc * 100.,
                                            ratio = self.ratio, alpha = self.alpha))

                if posioned_acc * 100 < 0.05: # set threshold for avoiding over unlearning
                    break

            if posioned_acc * 100 < 0.05:
                break

