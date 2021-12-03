from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model.resnet import resnet34
from model.basenet import AlexNetBase, VGGBase, Predictor, Predictor_deep
from utils.utils import weights_init
from utils.lr_schedule import inv_lr_scheduler
from utils.return_dataset import return_dataset_dual as return_dataset
from utils.loss import entropy, adentropy, MMD_loss

import time
import pdb
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")
# Training settings
def parse_args():
    parser = argparse.ArgumentParser(description='SSDA Classification')
    parser.add_argument('--steps', type=int, default=50001, metavar='N',
                    help='number of iterations to train (default: 50000)')
    parser.add_argument('--method', type=str, default='ASDA', choices=['S+T', 'ENT', 'MME', 'UODA', 'ASDA'],
                        help='MME is proposed method, ENT is entropy minimization, S+T is training only on labeled examples')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--multi', type=float, default=0.1, metavar='MLT',
                        help='learning rate multiplication')
    parser.add_argument('--T', type=float, default=0.05, metavar='T',
                        help='temperature (default: 0.05)')
    parser.add_argument('--lamda', type=float, default=0.1, metavar='LAM',
                        help='value of lamda')
    parser.add_argument('--save_check', action='store_true', default=False,
                        help='save checkpoint or not')
    parser.add_argument('--checkpath', type=str, default='../save_model_asda',
                        help='dir to save checkpoint')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_interval', type=int, default=500, metavar='N',
                        help='how many batches to wait before testing and saving a model')
    parser.add_argument('--net', type=str, default='resnet34', metavar='B',
                        help='which network to use')
    parser.add_argument('--source', type=str, default='real', metavar='B',
                        help='source domain')
    parser.add_argument('--target', type=str, default='sketch', metavar='B',
                        help='target domain')
    parser.add_argument('--dataset', type=str, default='multi', choices=['multi','office', 'office_home'],
                        help='the name of dataset')
    parser.add_argument('--bs', type=int, default=24, metavar='S',
                        help='batchsize')
    parser.add_argument('--num', type=int, default=3,
                        help='number of labeled examples in the target')
    parser.add_argument('--patience', type=int, default=5, metavar='S',
                        help='early stopping to wait for improvment '
                             'before terminating. (default: 5 (5000 iterations))')
    parser.add_argument('--early', action='store_false', default=False,
                        help='early stopping on validation or not')
    parser.add_argument('--data_root', type=str, default='../data',
                            help='dir to data')
    parser.add_argument('--Temp', type=float, default=1, metavar='Temp',
                        help='temperature (default: 1)')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')

    args = parser.parse_args()
    return args


class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        self.dataset_set_up()
        self.model_set_up()
        

    def dataset_set_up(self):
        print('Dataset %s Source %s Target %s Labeled num perclass %s Network %s' %
          (self.args.dataset, self.args.source, self.args.target, self.args.num, self.args.net))
        if args.num == 0:
            self.source_loader, self.target_loader, self.target_loader_unl, self.target_loader_val, \
            self.target_loader_test, self.class_list = return_dataset_uda(self.args)
        else:
            self.source_loader, self.target_loader, self.target_loader_unl, self.target_loader_val, \
            self.target_loader_test, self.class_list = return_dataset(self.args)

    def model_set_up(self):
        torch.cuda.manual_seed(self.args.seed)
        if args.net == 'resnet34':
            self.G = resnet34()
            inc = 512
        elif args.net == "alexnet":
            self.G = AlexNetBase()
            inc = 4096
        elif args.net == "vgg":
            self.G = VGGBase()
            inc = 4096
        else:
            raise ValueError('Model cannot be recognized.')

        params = []
        for key, value in dict(self.G.named_parameters()).items():
            if value.requires_grad:
                if 'classifier' not in key:
                    params += [{'params': [value], 'lr': self.args.multi,
                                'weight_decay': 0.0005}]
                else:
                    params += [{'params': [value], 'lr': self.args.multi * 10,
                                'weight_decay': 0.0005}]

        if "resnet" in args.net:
            self.F1 = Predictor_deep(num_class=len(self.class_list),
                                inc=inc)
            self.F2 = Predictor_deep(num_class=len(self.class_list),
                                inc=inc)

        else:
            self.F1 = Predictor_deep(num_class=len(self.class_list), inc=inc, temp=self.args.T)
            self.F2 = Predictor_deep(num_class=len(self.class_list), inc=inc, temp=self.args.T)


        weights_init(self.F1)
        weights_init(self.F2)

        lr = self.args.lr
        self.G.cuda()
        self.F1.cuda()
        self.F2.cuda()

        self.optimizer_g = optim.SGD( self.G.parameters(), lr = self.args.multi, momentum=0.9, weight_decay=0.0005,
                            nesterov=True)
        self.optimizer_f = optim.SGD(list(self.F1.parameters()), lr=1.0, momentum=0.9, weight_decay=0.0005,
                                nesterov=True)
        self.optimizer_f2 = optim.SGD(list(self.F2.parameters()), lr=1.0, momentum=0.9, weight_decay=0.0005,
                                nesterov=True)

        self.mmd_loss_linear = MMD_loss(kernel_type='linear')
        self.mmd_loss_rbf = MMD_loss(kernel_type='rbf')

    def train(self):
        self.G.train()
        self.F1.train()
        self.F2.train()
        
        def zero_grad_all():
            self.optimizer_g.zero_grad()
            self.optimizer_f.zero_grad()
            self.optimizer_f2.zero_grad()

        param_lr_g = []
        for param_group in self.optimizer_g.param_groups:
            param_lr_g.append(param_group["lr"])
        param_lr_f = []
        for param_group in self.optimizer_f.param_groups:
            param_lr_f.append(param_group["lr"])
        param_lr_f2 = []
        for param_group in self.optimizer_f2.param_groups:
            param_lr_f2.append(param_group["lr"])
        criterion = nn.CrossEntropyLoss().cuda()
        all_step = self.args.steps
        data_iter_s = iter(self.source_loader)
        data_iter_t = iter(self.target_loader)
        data_iter_t_unl = iter(self.target_loader_unl)
        len_train_source = len(self.source_loader)
        len_train_target = len(self.target_loader)
        len_train_target_semi = len(self.target_loader_unl)
        best_acc = 0
        best_acc_test = 0
        counter = 0
        time_last = time.time()
        time_last_save = time.time()

        for step in range(all_step):
            self.optimizer_g = inv_lr_scheduler(param_lr_g, self.optimizer_g, step,
                                           init_lr=self.args.lr)
            self.optimizer_f = inv_lr_scheduler(param_lr_f, self.optimizer_f, step,
                                           init_lr=self.args.lr)
            self.optimizer_f2 = inv_lr_scheduler(param_lr_f2, self.optimizer_f2, step, init_lr=self.args.lr)
            lr = self.optimizer_f.param_groups[0]['lr']
            if step % len_train_target == 0:
                data_iter_t = iter(self.target_loader)
            if step % len_train_target_semi == 0:
                data_iter_t_unl = iter(self.target_loader_unl)
            if step % len_train_source == 0:
                data_iter_s = iter(self.source_loader)
            data_t = next(data_iter_t)
            data_t_unl = next(data_iter_t_unl)
            data_s = next(data_iter_s)

            im_data_s = torch.cat((data_s[0][0], data_s[0][1]), dim=0).cuda()
            gt_labels_s = torch.cat((data_s[1], data_s[1]), dim=0).cuda()

            im_data_t = data_t[0].cuda()
            gt_labels_t = data_t[1].cuda()

            zero_grad_all()
            output_s = self.G(im_data_s)
            out_s = self.F1(output_s)

            output_t = self.G(im_data_t)
            out_t = self.F1(output_t)

            loss_s = criterion(out_s, gt_labels_s) 
            loss_t = criterion(out_t, gt_labels_t)


            loss = 0.75 * loss_s + 0.25 * loss_t

            loss.backward()

            self.optimizer_g.step()
            self.optimizer_f.step()
            zero_grad_all()

            output_t = self.G(im_data_t)
            out_t = self.F2(output_t)

            output_s = self.G(im_data_s)
            out_s = self.F2(output_s)

            loss_s = criterion(out_s, gt_labels_s) 
            loss_t = criterion(out_t, gt_labels_t) 
            
            loss = (0.25 * loss_s + 0.75 * loss_t) #+ 0.1 * proto_loss

            loss.backward()
            self.optimizer_f2.step()
            self.optimizer_g.step()
            zero_grad_all()

            im_data_tu_weak = data_t_unl[0][0].cuda()
            im_data_tu_stro = data_t_unl[0][1].cuda()

            im_data_tu = torch.cat((im_data_tu_weak, im_data_tu_stro),dim=0)

            output_tu = self.G(im_data_tu)
            logits = (self.F1(output_tu) + self.F2(output_tu)) / 2 
            logits_u_w, logits_u_s = logits.chunk(2)
            del logits

            pseudo_label = torch.softmax(logits_u_w.detach()/self.args.Temp, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(self.args.threshold).float()

            loss_fixmatch = (F.cross_entropy(logits_u_s, targets_u,
                                  reduction='none') * mask).mean()

            if loss_fixmatch.item() > 0:
                loss_fixmatch.backward()
                self.optimizer_f.step()
                self.optimizer_f2.step()
                self.optimizer_g.step()
                zero_grad_all()


            if not args.method == 'S+T':
                output_t = self.G(im_data_tu)
                output_s = self.G(im_data_s)
                loss_mmd = (self.mmd_loss_rbf(output_t[0:args.bs*2, :], output_s) + \
                    self.mmd_loss_rbf(output_t[args.bs*2:, :], output_s)) / 2 + \
                (self.mmd_loss_linear(output_t[0:args.bs*2, :], output_s) + \
                    self.mmd_loss_linear(output_t[args.bs*2:, :], output_s)) / 2


                if args.method == 'ENT':
                    loss_t = entropy(self.F1, output, self.args.lamda)
                    loss_t.backward()
                    self.optimizer_f.step()
                    self.optimizer_g.step()
                elif args.method == 'MME':
                    loss_t = adentropy(self.F1, output, self.args.lamda)
                    loss_t.backward()
                    self.optimizer_f.step()
                    self.optimizer_g.step()
                elif args.method == 'UODA' or args.method == 'ASDA':
                    loss_t =  adentropy(self.F2, output_t, self.args.lamda, s='tar')
                    loss_s = 1 *adentropy(self.F1, output_s, self.args.lamda, s='src')
                    loss_repr = loss_s + loss_t + 0.1 * loss_mmd

                    loss_repr.backward()
                    self.optimizer_f2.step()
                    self.optimizer_f.step()
                    self.optimizer_g.step()

                else:
                    raise ValueError('Method cannot be recognized.')
                log_train = 'S {} T {} Train Ep: {} lr{} \t Loss Classification: {:.6f} Loss T {:.6f} Loss S {:.6f} Loss FixM {:.6f} Method {}\n'.format(
                    args.source, args.target,
                    step, lr, loss.data, -loss_t.data, -loss_s.data, loss_fixmatch.data, args.method)
            else:
                log_train = 'S {} T {} Train Ep: {} lr{} \t ' \
                            'Loss Classification: {:.6f} Method {}\n'.\
                    format(self.args.source, self.args.target,
                           step, lr, loss.data,
                           self.args.method)
            self.G.zero_grad()
            self.F1.zero_grad()
            self.F2.zero_grad()
            zero_grad_all()
            if step % self.args.log_interval == 0:
                print(log_train)
                time_for_one_logging = time.time() - time_last
                time_last = time.time()
                print('The {} logging takes {:.0f}m {:.0f}s'.format(int(step/self.args.log_interval), time_for_one_logging // 60, time_for_one_logging % 60))
                
            if step % self.args.save_interval == 0 and step > 0:

                loss_test, acc_test = self.test(self.target_loader_test)
                loss_val, acc_val = self.test(self.target_loader_val)

                self.G.train()
                self.F1.train()
                self.F2.train()

                if acc_val > best_acc:
                    best_acc = acc_val
                if acc_test > best_acc_test:
                    best_acc_test = acc_test
                    counter = 0
                else:
                    counter += 1
                if args.early:
                    if counter > args.patience:
                        break
                print('best acc test %f best acc val %f' % (best_acc_test,
                                                            acc_val))
                if acc_val >= best_acc:
                    if args.save_check:
                        if os.path.exists(args.checkpath) == False:
                            os.mkdir(args.checkpath)
                        print('saving model')
                        torch.save(self.G.state_dict(),
                                   os.path.join(args.checkpath,
                                                "G_iter_model_{}_{}_"
                                                "to_{}_num_{}.pth.tar".
                                                format(args.name, args.source,
                                                       args.target, args.num)))
                        torch.save(self.F1.state_dict(),
                                   os.path.join(args.checkpath,
                                                "F1_iter_model_{}_{}_"
                                                "to_{}_num_{}.pth.tar".
                                                format(args.name, args.source,
                                                       args.target, args.num)))
                        torch.save(self.F2.state_dict(),
                                   os.path.join(args.checkpath,
                                                "F2_iter_model_{}_{}_"
                                                "to_{}_num_{}.pth.tar".
                                                format(args.name, args.source,
                                                       args.target, args.num)))
    def test(self, loader):
        self.G.eval()
        self.F1.eval()
        self.F2.eval()
        test_loss = 0
        correct = 0
        size = 0
        num_class = len(self.class_list)
        output_all = np.zeros((0, num_class))
        criterion = nn.CrossEntropyLoss().cuda()
        confusion_matrix = torch.zeros(num_class, num_class)
        with torch.no_grad():
            for batch_idx, data_t in enumerate(loader):
                im_data_t = data_t[0].cuda()
                gt_labels_t = data_t[1].cuda()
                feat = self.G(im_data_t)
                output1 =  self.F2(feat) + self.F1(feat)
                output_all = np.r_[output_all, output1.data.cpu().numpy()]
                size += im_data_t.size(0)
                pred1 = output1.data.max(1)[1]
                for t, p in zip(gt_labels_t.view(-1), pred1.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                correct += pred1.eq(gt_labels_t.data).cpu().sum()
                test_loss += criterion(output1, gt_labels_t) / len(loader)
        print('\nTest set: Average loss: {:.4f}, '
              'Accuracy: {}/{} F1 ({:.0f}%)\n'.
              format(test_loss, correct, size,
                     100. * correct / size))
        return test_loss.data, 100. * float(correct) / size



if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    trainer = Trainer(args)
    trainer.train()
