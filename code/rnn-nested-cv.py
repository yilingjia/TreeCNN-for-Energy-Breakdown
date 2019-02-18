import sys
from sklearn.metrics import mean_absolute_error
from dataloader import APPLIANCE_ORDER, get_train_test
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from torch.autograd import Variable
import warnings
warnings.filterwarnings("ignore")

cuda_av = False
if torch.cuda.is_available():
    cuda_av = True

torch.manual_seed(0)
np.random.seed(0)


class CustomRNN(nn.Module):
    def __init__(self, cell_type, hidden_size, num_layers, bidirectional):
        super(CustomRNN, self).__init__()
        torch.manual_seed(0)

        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        if cell_type == "RNN":
            self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True,
                              bidirectional=bidirectional)
        elif cell_type == "GRU":
            self.rnn = nn.GRU(input_size=1, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True,
                              bidirectional=bidirectional)
        else:
            self.rnn = nn.LSTM(input_size=1, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True,
                               bidirectional=bidirectional)

        self.linear = nn.Linear(hidden_size * self.num_directions, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        pred, hidden = self.rnn(x, None)
        pred = self.linear(pred).view(pred.data.shape[0], -1, 1)
        # pred = self.act(pred)
        # pred = torch.clamp(pred, min=0.)
        #pred = self.act(pred)
        pred = torch.min(pred, x)
        return pred


class AppliancesRNN(nn.Module):
    def __init__(self, cell_type, hidden_size, num_layers, bidirectional, num_appliance):
        super(AppliancesRNN, self).__init__()
        self.num_appliance = num_appliance
        self.preds = {}
        self.order = ORDER
        for appliance in range(self.num_appliance):
            if cuda_av:
                setattr(self, "Appliance_" + str(appliance), CustomRNN(cell_type,
                                                                       hidden_size,
                                                                       num_layers,
                                                                       bidirectional).cuda())
            else:
                setattr(self, "Appliance_" + str(appliance), CustomRNN(cell_type,
                                                                       hidden_size,
                                                                       num_layers,
                                                                       bidirectional))

    def forward(self, *args):
        agg_current = args[0]
        flag = False
        if np.random.random() > args[1]:
            flag = True
            # print("Subtracting prediction")
        else:
            pass
            # print("Subtracting true")
        for appliance in range(self.num_appliance):
            # print(agg_current.mean().data[0])
            # print (appliance)
            # print (self.order[appliance])
            # print (args[2+appliance])
            # print(getattr(self, "Appliance_" + str(appliance)))
            self.preds[appliance] = getattr(self, "Appliance_" + str(appliance))(agg_current)
            if flag:
                agg_current = agg_current - self.preds[appliance]
            else:
                agg_current = agg_current - args[2 + appliance]

        return torch.cat([self.preds[a] for a in range(self.num_appliance)])


def disagg_fold(fold_num, dataset, cell_type, hidden_size, num_layers, bidirectional, lr, num_iterations, p):
    # print (fold_num, hidden_size, num_layers, bidirectional, lr, num_iterations, p)
    #print (ORDER)
    torch.manual_seed(0)

    num_folds=5
    train, test = get_train_test(dataset, num_folds=num_folds, fold_num=fold_num)
    # from sklearn.model_selection import train_test_split
    # train, valid = train_test_split(train, test_size=0.2, random_state=0)

    valid = train[int(0.8*len(train)):].copy()
    train = train[:int(0.8 * len(train))].copy()


    train_aggregate = train[:, 0, :, :].reshape(-1, train.shape[3], 1)
    valid_aggregate = valid[:, 0, :, :].reshape(-1, train.shape[3], 1)
    test_aggregate = test[:, 0, :, :].reshape(-1, train.shape[3], 1)


    #print (train.shape)
    #print (valid.shape)
    #print (test.shape)

    out_train = [None for temp in range(len(ORDER))]
    for a_num, appliance in enumerate(ORDER):
        out_train[a_num] = Variable(
            torch.Tensor(train[:, APPLIANCE_ORDER.index(appliance), :, :].reshape((train_aggregate.shape[0], -1, 1))))
        if cuda_av:
            out_train[a_num] = out_train[a_num].cuda()

    out_valid = [None for temp in range(len(ORDER))]
    for a_num, appliance in enumerate(ORDER):
        out_valid[a_num] = Variable(
            torch.Tensor(valid[:, APPLIANCE_ORDER.index(appliance), :, :].reshape((valid_aggregate.shape[0], -1, 1))))
        if cuda_av:
            out_valid[a_num] = out_valid[a_num].cuda()

    out_test = [None for temp in range(len(ORDER))]
    for a_num, appliance in enumerate(ORDER):
        out_test[a_num] = Variable(
            torch.Tensor(test[:, APPLIANCE_ORDER.index(appliance), :, :].reshape((test_aggregate.shape[0], -1, 1))))
        if cuda_av:
            out_test[a_num] = out_test[a_num].cuda()

    loss_func = nn.L1Loss()
    a = AppliancesRNN(cell_type, hidden_size, num_layers, bidirectional, len(ORDER))
    # prevent negative
    #for param in a.parameters():
    #    param.data = param.data.abs()
    #print(a)
    if cuda_av:
        a = a.cuda()
        loss_func = loss_func.cuda()
    optimizer = torch.optim.Adam(a.parameters(), lr=lr)

    inp = Variable(torch.Tensor(train_aggregate.reshape((train_aggregate.shape[0], -1, 1))).type(torch.FloatTensor),
                   requires_grad=True)

    valid_inp = Variable(torch.Tensor(valid_aggregate), requires_grad=False)
    if cuda_av:
        valid_inp = valid_inp.cuda()

    test_inp = Variable(torch.Tensor(test_aggregate), requires_grad=False)
    if cuda_av:
        test_inp = test_inp.cuda()

    valid_pred = {}
    train_pred = {}
    test_pred = {}
    test_losses = {}
    valid_losses = {}

    for t in range(1, num_iterations+1):
        idx_train = Variable(torch.LongTensor(np.random.choice(range(train_aggregate.shape[0]), 50, replace=True)))
        inp = Variable(torch.Tensor(train_aggregate), requires_grad=True)

        valid_out = torch.cat([out_valid[appliance_num] for appliance_num, appliance in enumerate(ORDER)])
        test_out = torch.cat([out_test[appliance_num] for appliance_num, appliance in enumerate(ORDER)])

        if cuda_av:
            idx_train = idx_train.cuda()
            out = torch.cat(
                [out_train[appliance_num].index_select(0, idx_train) for appliance_num, appliance in enumerate(ORDER)])
            inp = inp.cuda().index_select(0, idx_train)
            out = out.cuda()
        else:
            inp = inp.index_select(0, idx_train)
            out = torch.cat(
                [out_train[appliance_num].index_select(0, idx_train) for appliance_num, appliance in enumerate(ORDER)])

        params = [inp, p]
        for a_num, appliance in enumerate(ORDER):
            params.append(out_train[a_num])
        # print(params)
        pred = a(*params)

        optimizer.zero_grad()
        loss = loss_func(pred, out)
        if t % 50 == 0:
            #print(t, loss.data[0])

            if cuda_av:
                valid_inp = valid_inp.cuda()
            valid_params = [valid_inp, -2]
            for i in range(len(ORDER)):
                valid_params.append(None)
            valid_pr = a(*valid_params)
            valid_loss = loss_func(valid_pr, valid_out)

            if cuda_av:
                test_inp = test_inp.cuda()
            test_params = [test_inp, -2]
            for i in range(len(ORDER)):
                test_params.append(None)
            test_pr = a(*test_params)
            test_loss = loss_func(test_pr, test_out)

            test_losses[t] = test_loss.data[0]
            valid_losses[t] = valid_loss.data[0]
            # np.save("./baseline/p_50_loss")

            if t % 1000 == 0:
                valid_pr = torch.clamp(valid_pr, min=0.)
                valid_pred[t] = valid_pr
                test_pr = torch.clamp(test_pr, min=0.)
                test_pred[t] = test_pr
                train_pr = pred
                train_pr = torch.clamp(train_pr, min=0.)
                train_pred[t] = train_pr

            #print("Round:", t, "Training Error:", loss.data[0], "Validation Error:", valid_loss.data[0], "Test Error:", test_loss.data[0])

        loss.backward()
        optimizer.step()

    # store training prediction
    # train_pred = torch.clamp(pred, min=0.)
    # train_pred = torch.split(train_pred, train_aggregate.shape[0])
    train_fold = [None for x in range(len(ORDER))]
    # if cuda_av:
    #     for appliance_num, appliance in enumerate(ORDER):
    #         train_fold[appliance_num] = train_pred[appliance_num].cpu().data.numpy().reshape(-1, 24)
    # else:
    #     for appliance_num, appliance in enumerate(ORDER):
    #         train_fold[appliance_num] = train_pred[appliance_num].data.numpy().reshape(-1, 24)


            # test one validation set

    valid_fold = {}
    for t in range(1000, num_iterations + 1, 1000):

        valid_pred[t] = torch.split(valid_pred[t], valid_aggregate.shape[0])
        valid_fold[t] = [None for x in range(len(ORDER))]
        if cuda_av:
            for appliance_num, appliance in enumerate(ORDER):
                valid_fold[t][appliance_num] = valid_pred[t][appliance_num].cpu().data.numpy().reshape(-1, valid.shape[3])
        else:
            for appliance_num, appliance in enumerate(ORDER):
                valid_fold[t][appliance_num] = valid_pred[t][appliance_num].data.numpy().reshape(-1, valid.shape[3])

    test_fold = {}
    for t in range(1000, num_iterations + 1, 1000):

        test_pred[t] = torch.split(test_pred[t], test_aggregate.shape[0])
        test_fold[t] = [None for x in range(len(ORDER))]
        if cuda_av:
            for appliance_num, appliance in enumerate(ORDER):
                test_fold[t][appliance_num] = test_pred[t][appliance_num].cpu().data.numpy().reshape(-1, valid.shape[3])
        else:
            for appliance_num, appliance in enumerate(ORDER):
                test_fold[t][appliance_num] = test_pred[t][appliance_num].data.numpy().reshape(-1, valid.shape[3])

    # store ground truth of validation set
    valid_gt_fold = [None for x in range(len(ORDER))]
    for appliance_num, appliance in enumerate(ORDER):
        valid_gt_fold[appliance_num] = valid[:, APPLIANCE_ORDER.index(appliance), :, :].reshape(
            valid_aggregate.shape[0],
            -1, 1).reshape(-1, valid.shape[3])

    test_gt_fold = [None for x in range(len(ORDER))]
    for appliance_num, appliance in enumerate(ORDER):
        test_gt_fold[appliance_num] = test[:, APPLIANCE_ORDER.index(appliance), :, :].reshape(
            test_aggregate.shape[0],
            -1, 1).reshape(-1, test.shape[3])

    # calcualte the error of validation set
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

    return train_fold, valid_fold, test_fold, valid_error, test_error, valid_losses, test_losses


dataset, cell_type, hidden_size, num_layers, bidirectional, lr, num_iterations, p, fold_num = sys.argv[1:10]
fold_num = int(fold_num)
dataset = int(dataset)
hidden_size = int(hidden_size)
num_layers = int(num_layers)
lr = float(lr)
num_iterations = int(num_iterations)
p = float(p)
ORDER = sys.argv[10:len(sys.argv)]

print(dataset, fold_num, cell_type, hidden_size, num_layers, bidirectional, lr, num_iterations, p, ORDER)
input_dim = 1
num_folds = 5

folder = "{}/{}/{}/{}/{}/{}/{}/{}/{}".format(dataset, fold_num, cell_type, hidden_size, num_layers, bidirectional, lr, num_iterations, p)

train_fold, valid_fold, test_fold, valid_error, test_error, valid_losses, test_losses = disagg_fold(fold_num, dataset, cell_type, hidden_size, num_layers, bidirectional, lr, num_iterations, p)

directory = os.path.expanduser('./baseline/rnn-tree/{}'.format(folder))
if not os.path.exists(directory):
    os.makedirs(directory)
#filename = os.path.join(directory, name + '.pkl')

# np.save('./baseline/rnn-tree/{}/valid-pred-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(folder, fold_num, dataset, cell_type, hidden_size, num_layers, bidirectional, lr, num_iterations, p, ORDER), valid_fold)
np.save('./baseline/rnn-tree/{}/valid-error-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(folder, fold_num, dataset, cell_type, hidden_size, num_layers, bidirectional, lr, num_iterations, p, ORDER), valid_error)
# np.save('./baseline/rnn-tree-order-new/train-pred-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(fold_num, dataset, cell_type, hidden_size, num_layers, bidirectional, lr, num_iterations, p, ORDER), train_fold)
# np.save('./baseline/rnn-tree/{}/test-pred-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(folder, fold_num, dataset, cell_type, hidden_size, num_layers, bidirectional, lr, num_iterations, p, ORDER), test_fold)
np.save('./baseline/rnn-tree/{}/test-error-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(folder, fold_num, dataset, cell_type, hidden_size, num_layers, bidirectional, lr, num_iterations, p, ORDER), test_error)
# np.save('./baseline/rnn-tree/{}/test-losses-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(folder, fold_num, dataset, cell_type, hidden_size, num_layers, bidirectional, lr, num_iterations, p, ORDER), test_losses)
# np.save('./baseline/rnn-tree/{}/valid-losses-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(folder, fold_num, dataset, cell_type, hidden_size, num_layers, bidirectional, lr, num_iterations, p, ORDER), valid_losses)

