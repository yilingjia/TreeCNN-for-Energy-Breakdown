import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
# from torchvision import datasets, transforms
# import torch.nn.functional as F
from dataloader import APPLIANCE_ORDER, get_train_test
from sklearn.metrics import mean_absolute_error
import os
import sys
sys.path.append("../code/")

cuda_av = False
if torch.cuda.is_available():
    cuda_av = True

torch.manual_seed(0)


class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=7, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(20)

        self.conv2 = nn.Conv2d(20, 16, kernel_size=2, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv5 = nn.ConvTranspose2d(16, 6, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(6)

        self.conv6 = nn.ConvTranspose2d(6, 1, kernel_size=5, stride=1, padding=2)

        self.act = nn.ReLU()

    # forward method
    def forward(self, input):

        e1 = self.conv1(input)
        bn1 = self.bn1(self.act(e1))
        e2 = self.bn2(self.conv2(bn1))
        e5 = self.bn5(self.conv5(e2))
        e6 = self.conv6(e5)
        return e6


class AppliancesCNN(nn.Module):
    def __init__(self, num_appliance):
        super(AppliancesCNN, self).__init__()
        self.num_appliance = num_appliance
        self.preds = {}
        self.order = ORDER
        for appliance in range(self.num_appliance):
            if cuda_av:
                setattr(self, "Appliance_" + str(appliance), CustomCNN().cuda())
            else:
                setattr(self, "Appliance_" + str(appliance), CustomCNN())

    def forward(self, *args):
        agg_current = args[0]
        flag = False
        if np.random.random() > args[1]:
            flag = True
        else:
            pass
        for appliance in range(self.num_appliance):
            self.preds[appliance] = getattr(self, "Appliance_" + str(appliance))(agg_current)
            # if flag:
            #     agg_current = agg_current - self.preds[appliance]
            # else:
            #     agg_current = agg_current - args[2 + appliance]

        return torch.cat([self.preds[a] for a in range(self.num_appliance)])


def preprocess(train, valid, test):
    out_train = [None for temp in range(len(ORDER))]
    for a_num, appliance in enumerate(ORDER):
        out_train[a_num] = Variable(
            torch.Tensor(train[:, APPLIANCE_ORDER.index(appliance), :, :].reshape((train.shape[0], 1, -1, 24))))
        if cuda_av:
            out_train[a_num] = out_train[a_num].cuda()

    out_valid = [None for temp in range(len(ORDER))]
    for a_num, appliance in enumerate(ORDER):
        out_valid[a_num] = Variable(
            torch.Tensor(valid[:, APPLIANCE_ORDER.index(appliance), :, :].reshape((valid.shape[0], 1, -1, 24))))
        if cuda_av:
            out_valid[a_num] = out_valid[a_num].cuda()

    out_test = [None for temp in range(len(ORDER))]
    for a_num, appliance in enumerate(ORDER):
        out_test[a_num] = Variable(
            torch.Tensor(test[:, APPLIANCE_ORDER.index(appliance), :, :].reshape((test.shape[0], 1, -1, 24))))
        if cuda_av:
            out_test[a_num] = out_test[a_num].cuda()

    return out_train, out_valid, out_test


def disagg_fold(dataset, fold_num, lr, p):
    train, test = get_train_test(dataset, num_folds=num_folds, fold_num=fold_num)
    valid = train[int(0.8 * len(train)):].copy()
    train = train[:int(0.8 * len(train))].copy()
    train_aggregate = train[:, 0, :, :].reshape(train.shape[0], 1, -1, 24)
    valid_aggregate = valid[:, 0, :, :].reshape(valid.shape[0], 1, -1, 24)
    test_aggregate = test[:, 0, :, :].reshape(test.shape[0], 1, -1, 24)

    out_train, out_valid, out_test = preprocess(train, valid, test)

    loss_func = nn.L1Loss()
    model = AppliancesCNN(len(ORDER))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if cuda_av:
        model = model.cuda()
        loss_func = loss_func.cuda()

    inp = Variable(torch.Tensor(train_aggregate), requires_grad=False)
    valid_inp = Variable(torch.Tensor(valid_aggregate), requires_grad=False)
    test_inp = Variable(torch.Tensor(test_aggregate), requires_grad=False)
    if cuda_av:
        inp = inp.cuda()
        valid_inp = valid_inp.cuda()
        test_inp = test_inp.cuda()
    valid_out = torch.cat([out_valid[appliance_num] for appliance_num, appliance in enumerate(ORDER)])
    test_out = torch.cat([out_test[appliance_num] for appliance_num, appliance in enumerate(ORDER)])
    train_out = torch.cat([out_train[appliance_num] for appliance_num, appliance in enumerate(ORDER)])

    valid_pred = {}
    train_pred = {}
    test_pred = {}
    train_losses = {}
    test_losses = {}
    valid_losses = {}

    params = [inp, p]
    for a_num, appliance in enumerate(ORDER):
        params.append(out_train[a_num])

    if cuda_av:
        train_out = train_out.cuda()

    for t in range(1, num_iterations + 1):

        pred = model(*params)
        optimizer.zero_grad()
        loss = loss_func(pred, train_out)

        if t % 500 == 0:

            if cuda_av:
                valid_inp = valid_inp.cuda()
            valid_params = [valid_inp, -2]
            for i in range(len(ORDER)):
                valid_params.append(None)
            valid_pr = model(*valid_params)
            valid_loss = loss_func(valid_pr, valid_out)

            if cuda_av:
                test_inp = test_inp.cuda()
            test_params = [test_inp, -2]
            for i in range(len(ORDER)):
                test_params.append(None)
            test_pr = model(*test_params)
            test_loss = loss_func(test_pr, test_out)

            test_losses[t] = test_loss.data[0]
            valid_losses[t] = valid_loss.data[0]
            train_losses[t] = loss.data[0]
            # np.save("./baseline/p_50_loss")

            if t % 1000 == 0:
                valid_pr = torch.clamp(valid_pr, min=0.)
                valid_pred[t] = valid_pr
                test_pr = torch.clamp(test_pr, min=0.)
                test_pred[t] = test_pr
                train_pr = pred
                train_pr = torch.clamp(train_pr, min=0.)
                train_pred[t] = train_pr

            print("Round:", t, "Training Error:", loss.data[0], "Validation Error:", valid_loss.data[0], "Test Error:", test_loss.data[0])

        loss.backward()
        optimizer.step()

    train_fold = [None for x in range(len(ORDER))]
    train_fold = {}
    for t in range(1000, num_iterations + 1, 1000):
        train_pred[t] = torch.split(train_pred[t], train_aggregate.shape[0])
        train_fold[t] = [None for x in range(len(ORDER))]
        if cuda_av:
            for appliance_num, appliance in enumerate(ORDER):
                train_fold[t][appliance_num] = train_pred[t][appliance_num].cpu().data.numpy().reshape(-1, 24)
        else:
            for appliance_num, appliance in enumerate(ORDER):
                train_fold[t][appliance_num] = train_pred[t][appliance_num].data.numpy().reshape(-1, 24)

    valid_fold = {}
    for t in range(1000, num_iterations + 1, 1000):
        valid_pred[t] = torch.split(valid_pred[t], valid_aggregate.shape[0])
        valid_fold[t] = [None for x in range(len(ORDER))]
        if cuda_av:
            for appliance_num, appliance in enumerate(ORDER):
                valid_fold[t][appliance_num] = valid_pred[t][appliance_num].cpu().data.numpy().reshape(-1, 24)
        else:
            for appliance_num, appliance in enumerate(ORDER):
                valid_fold[t][appliance_num] = valid_pred[t][appliance_num].data.numpy().reshape(-1, 24)

    test_fold = {}
    for t in range(1000, num_iterations + 1, 1000):
        test_pred[t] = torch.split(test_pred[t], test_aggregate.shape[0])
        test_fold[t] = [None for x in range(len(ORDER))]
        if cuda_av:
            for appliance_num, appliance in enumerate(ORDER):
                test_fold[t][appliance_num] = test_pred[t][appliance_num].cpu().data.numpy().reshape(-1, 24)
        else:
            for appliance_num, appliance in enumerate(ORDER):
                test_fold[t][appliance_num] = test_pred[t][appliance_num].data.numpy().reshape(-1, 24)

    # store ground truth of validation set
    train_gt_fold = [None for x in range(len(ORDER))]
    for appliance_num, appliance in enumerate(ORDER):
        train_gt_fold[appliance_num] = train[:, APPLIANCE_ORDER.index(appliance), :, :].reshape(
            train_aggregate.shape[0],
            -1, 1).reshape(-1, 24)

    valid_gt_fold = [None for x in range(len(ORDER))]
    for appliance_num, appliance in enumerate(ORDER):
        valid_gt_fold[appliance_num] = valid[:, APPLIANCE_ORDER.index(appliance), :, :].reshape(
            valid_aggregate.shape[0],
            -1, 1).reshape(-1, 24)

    test_gt_fold = [None for x in range(len(ORDER))]
    for appliance_num, appliance in enumerate(ORDER):
        test_gt_fold[appliance_num] = test[:, APPLIANCE_ORDER.index(appliance), :, :].reshape(
            test_aggregate.shape[0],
            -1, 1).reshape(-1, 24)

    # calcualte the error of validation set
    train_error = {}
    for t in range(1000, num_iterations + 1, 1000):
        train_error[t] = {}
        for appliance_num, appliance in enumerate(ORDER):
            train_error[t][appliance] = mean_absolute_error(train_fold[t][appliance_num], train_gt_fold[appliance_num])

    valid_error = {}
    for t in range(1000, num_iterations + 1, 1000):
        valid_error[t] = {}
        for appliance_num, appliance in enumerate(ORDER):
            valid_error[t][appliance] = mean_absolute_error(valid_fold[t][appliance_num], valid_gt_fold[appliance_num])

    test_error = {}
    for t in range(1000, num_iterations + 1, 1000):
        test_error[t] = {}
        for appliance_num, appliance in enumerate(ORDER):
            test_error[t][appliance] = mean_absolute_error(test_fold[t][appliance_num], test_gt_fold[appliance_num])

    return train_fold, valid_fold, test_fold, train_error, valid_error, test_error, train_losses, valid_losses, test_losses


num_folds = 5
dataset, lr, num_iterations, p, fold_num = sys.argv[1:6]
ORDER = sys.argv[6:len(sys.argv)]
dataset = int(dataset)
lr = float(lr)
num_iterations = int(num_iterations)
p = float(p)
fold_num = int(fold_num)

print(dataset, fold_num, lr, num_iterations, p, ORDER)

train_fold, valid_fold, test_fold, train_error, valid_error, test_error, train_losses, valid_losses, test_losses = disagg_fold(dataset,
                                                                                                                               fold_num, lr, p)

directory = "./baseline/cnn/{}/{}/{}/{}/{}".format(dataset, fold_num, lr, num_iterations, p)
if not os.path.exists(directory):
    os.makedirs(directory)


np.save('{}/valid-pred-{}.npy'.format(directory, ORDER), valid_fold)
np.save('{}/valid-error-{}.npy'.format(directory, ORDER), valid_error)
# np.save('{}/valid-losses-{}.npy'.format(directory, ORDER), valid_losses)
# np.save('{}/train-pred-{}.npy'.format(directory, ORDER), train_fold)
# np.save('{}/train-error-{}.npy'.format(directory, ORDER), train_error)
# np.save('{}/train-losses-{}.npy'.format(directory, ORDER), train_losses)
np.save('{}/test-pred-{}.npy'.format(directory, ORDER), test_fold)
np.save('{}/test-error-{}.npy'.format(directory, ORDER), test_error)
# np.save('{}/test-losses-{}.npy'.format(directory, ORDER), test_losses)
