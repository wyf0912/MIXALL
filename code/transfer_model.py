from networks import *
import torch
import torch.nn as nn
from collections import OrderedDict
h = 512  # 1000
hh = 100
num_domain = 10
num_test_domain = 4
num_train_domain = num_domain - num_test_domain


def freeze_layer(model):
    count = 0
    para_optim = []
    for k in model.children():

        count += 1
        # 6 should be changed properly
        if count > 6:
            for param in k.parameters():
                para_optim.append(param)
        else:
            for param in k.parameters():
                param.requires_grad = False

    # print count
    return para_optim


def classifier(class_num):
    model = nn.Sequential(
        nn.Linear(512, class_num),
    )

    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    model.apply(init_weights)
    return model.cuda()


feature_extractor_network = resnet18(pretrained=True)
param_optim_theta = freeze_layer(feature_extractor_network)
# theta means the network parameter of feature extractor, from d (the size of input) to h(the size of feature layer).

phi_all = classifier(
    100 + 2 + 43 + 1623 + 10 + 1000)  # CIFAR-100  Daimler Ped GTSRB Omniglot SVHN ImageNet
tmp = torch.load('./model_output/VD/baseline/best_model.tar')
feature_extractor_network.load_state_dict(tmp[0])
phil_all_state = OrderedDict()
phil_all_state['0.weight'] = torch.cat([classifier['0.weight'] for classifier in tmp[1]],0) # 2778*512
phil_all_state['0.bias'] = torch.cat([classifier['0.bias'] for classifier in tmp[1]],0) # 2778
phi_all.load_state_dict(phil_all_state)
torch.save((feature_extractor_network.state_dict(), phi_all.state_dict()), './model_output/VD/baseline/best_model_transfered.tar')
