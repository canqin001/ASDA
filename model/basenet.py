from torchvision import models
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Function
from model.resnet import resnet34, resnet50


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)

    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)

    _output = torch.div(input, norm.view(-1, 1).expand_as(input))

    output = _output.view(input_size)

    return output


class AlexNetBase(nn.Module):
    def __init__(self, pret=True):
        super(AlexNetBase, self).__init__()
        model_alexnet = models.alexnet(pretrained=pret)
        self.features = nn.Sequential(*list(model_alexnet.
                                            features._modules.values())[:])
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i),
                                       model_alexnet.classifier[i])
        self.__in_features = model_alexnet.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self.__in_features


class VGGBase(nn.Module):
    def __init__(self, pret=True, no_pool=False):
        super(VGGBase, self).__init__()
        vgg16 = models.vgg16(pretrained=pret)
        self.classifier = nn.Sequential(*list(vgg16.classifier.
                                              _modules.values())[:-1])
        self.features = nn.Sequential(*list(vgg16.features.
                                            _modules.values())[:])
        self.s = nn.Parameter(torch.FloatTensor([10]))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 7 * 7 * 512)
        x = self.classifier(x)
        return x


class Predictor(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Predictor, self).__init__()
        self.fc = nn.Linear(inc, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc(x) / self.temp
        return x_out

class MLP(nn.Module):
    def __init__(self, num_class=64):
        super(MLP, self).__init__()
        self.fc = nn.Linear(num_class * 2, num_class * 2)
        self.bn = nn.BatchNorm1d(num_class * 2)
        self.fc1 = nn.Linear(num_class * 2, num_class)
    def forward(self, x):
        # x = self.fc(x)
        # x = self.bn(x)
        x = self.fc1(x)
        return x

class Predictor_deep(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Predictor_deep, self).__init__()
        self.fc1 = nn.Linear(inc, 512)
        self.fc2 = nn.Linear(512, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        #pdb.set_trace()
        x = self.fc1(x)
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc2(x) / self.temp
        return x_out


class Discriminator(nn.Module):
    def __init__(self, num_class=2, inc=4096):
        super(Discriminator, self).__init__()
        self.fc = nn.Linear(inc, num_class, bias=False)
        self.num_class = num_class

    def forward(self, x, reverse=False, eta=0.1):
        if reverse:
            x = grad_reverse(x, eta)
        x_out = self.fc(x)
        return x_out

class Discriminator_deep(nn.Module):
    def __init__(self, num_class=2, inc=4096):
        super(Discriminator_deep, self).__init__()
        self.fc1 = nn.Linear(inc, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.activation1 = nn.LeakyReLU()

        self.fc2 = nn.Linear(512, num_class, bias=False)
        self.num_class = num_class

    def forward(self, x, reverse=False, eta=0.1):
        #pdb.set_trace()
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        if reverse:
            x = grad_reverse(x, eta)
        x_out = self.fc2(x)
        return x_out

class Inter_Encoder(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Inter_Encoder, self).__init__()
        self.fc1 = nn.Linear(inc, inc)
        self.bn1 = nn.BatchNorm1d(inc)
        self.activation1 = nn.LeakyReLU()

        self.fc2 = nn.Linear(inc, inc)
        self.bn2 = nn.BatchNorm1d(inc)
        self.activation2 = nn.LeakyReLU()

        self.conv1 = nn.Conv2d(inc, inc, 1, padding=0, bias=True)

        self.fc1_s = nn.Linear(inc, inc)
        self.bn1_s = nn.BatchNorm1d(inc)
        self.activation1_s = nn.LeakyReLU()
        self.fc1_t = nn.Linear(inc, inc)
        self.bn1_t = nn.BatchNorm1d(inc)
        self.activation1_t = nn.LeakyReLU()

        self.fc2_s = nn.Linear(inc, inc)
        self.bn2_s = nn.BatchNorm1d(inc)
        self.activation2_s = nn.LeakyReLU()
        self.fc2_t = nn.Linear(inc, inc)
        self.bn2_t = nn.BatchNorm1d(inc)
        self.activation2_t = nn.LeakyReLU() 

        self.conv1_s = nn.Conv2d(inc, inc, 1, padding=0, bias=True)
        self.conv1_t = nn.Conv2d(inc, inc, 1, padding=0, bias=True)

    def forward(self, x, reverse=False, eta=0.1, s='src'):
        
        # residual = x
        # if s == 'src':
        #     out = self.fc1_s(x)
        #     out = self.bn1_s(out)
        #     out = self.activation1_s(out)
        #     out = self.fc2_s(out)
        #     out = self.bn2_s(out)
        #     out = self.activation2_s(out)
        # elif s == 'tar':
        #     out = self.fc1_t(x)
        #     out = self.bn1_t(out)
        #     out = self.activation1_t(out)      
        #     out = self.fc2_t(out)
        #     out = self.bn2_t(out)
        #     out = self.activation2_t(out)

        # out += residual


        #x = self.fc1(x)

        if s == 'src':
            x = self.bn1_s(x)
            x = self.activation1_s(x)

        elif s == 'tar':
            x = self.bn1_s(x)
            x = self.activation1_s(x)

        #x = self.fc2(x)

        #if s == 'src':
        #    x = self.bn2_s(x)
        #    x = self.activation2_s(x)

        #elif s == 'tar':
        #    x = self.bn2_t(x)
        #    x = self.activation2_t(x)
        
        # x = self.fc3(x)
        # x = self.bn3(x)
        # x = self.activation3(x)
        return x

class Predictor_deep_state(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Predictor_deep_state, self).__init__()
        self.fc1 = nn.Linear(inc, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, num_class,bias=False)
        self.num_class = num_class
        self.temp = temp
    def forward(self, x, reverse=False,eta=0.1):
        x = self.fc1(x)
        x = self.bn1(x)

        if reverse:
            x = grad_reverse(x, eta)

        x = F.normalize(x)
        x_out = self.fc2(x)/self.temp
        return x_out

class Predictor_deep_single(nn.Module):
    def __init__(self, num_class=64, inc=4096):
        super(Predictor_deep, self).__init__()
        self.fc1 = nn.Linear(inc, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, num_class,bias=False)
        self.num_class = num_class

    def forward(self, x, reverse=False):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.normalize(x)
        x_out = self.fc2(x)
        return x_out

# class Discriminator(nn.Module):
#     def __init__(self, inc=4096):
#         super(Discriminator, self).__init__()
#         self.fc1_1 = nn.Linear(inc, 512)
#         self.fc2_1 = nn.Linear(512, 512)
#         self.fc3_1 = nn.Linear(512, 1)

#     def forward(self, x, reverse=True, eta=1.0):
#         if reverse:
#             x = grad_reverse(x, eta)
#         x = F.relu(self.fc1_1(x))
#         x = F.relu(self.fc2_1(x))
#         x_out = F.sigmoid(self.fc3_1(x))
#         return x_out

class ResNet_Full(nn.Module):
    def __init__(self, num_class=64, inc=4096, net='resnet34'):
        super(ResNet_Full, self).__init__()
        if net == 'resnet34':
            self.features = resnet34()
        elif net == 'resnet50':
            self.features = resnet50()
        else:
            raise ValueError('Model cannot be recognized.')

        self.classifier = Predictor_deep_single(num_class=num_class, inc=inc)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x
