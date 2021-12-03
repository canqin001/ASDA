import os
import torch
from torchvision import transforms
from loaders.data_list import Imagelists_VISDA, return_classlist
import pdb
from .randaugment import RandAugmentMC

class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))

class TransformFixMatch(object):
    def __init__(self, mean, std):
        

        self.weak = transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=224,
                                  padding=int(224*0.125)
                                  )]) #padding_mode='reflect'
        self.strong = transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=224,
                                  padding=int(224*0.125)
                                  ), #padding_mode='reflect'
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

class TransformFixMatch_1W2S(object):
    def __init__(self, mean, std):
        

        self.weak = transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=224,
                                  padding=int(224*0.125)
                                  )]) #padding_mode='reflect'
        self.strong = transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=224,
                                  padding=int(224*0.125)
                                  ), #padding_mode='reflect'
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        strong_2 = self.strong(x)
        return self.normalize(weak), self.normalize(strong), self.normalize(strong_2)


class DualTransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=224,
                                  padding=int(224*0.125)
                                  )]) #padding_mode='reflect'
        self.strong = transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
             #padding_mode='reflect'
            RandAugmentMC(n=2, m=10),
            transforms.RandomCrop(size=224,
                                  padding=int(224*0.125)
                                  )])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

def return_dataset_init(args):
    base_path = '../data/txt/%s' % args.dataset
    root = '../data/%s/' % args.dataset
    image_set_file_s = \
        os.path.join(base_path,
                     'labeled_source_images_' +
                     args.source + '.txt')
    image_set_file_t = \
        os.path.join(base_path,
                     'labeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))
    image_set_file_t_val = \
        os.path.join(base_path,
                     'validation_target_images_' +
                     args.target + '_3.txt')
    image_set_file_unl = \
        os.path.join(base_path,
                     'unlabeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    source_dataset = Imagelists_VISDA(image_set_file_s, root=root,
                                      transform=data_transforms['train'])
    target_dataset = Imagelists_VISDA(image_set_file_t, root=root,
                                      transform=data_transforms['val'])
    target_dataset_val = Imagelists_VISDA(image_set_file_t_val, root=root,
                                          transform=data_transforms['val'])
    target_dataset_unl = Imagelists_VISDA(image_set_file_unl, root=root,
                                          transform=data_transforms['val'])
    target_dataset_test = Imagelists_VISDA(image_set_file_unl, root=root,
                                           transform=data_transforms['test'])
    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=bs,
                                                num_workers=3, shuffle=True,
                                                drop_last=True)
    target_loader = \
        torch.utils.data.DataLoader(target_dataset,
                                    batch_size=min(bs, len(target_dataset)),
                                    num_workers=3,
                                    shuffle=True, drop_last=True)
    target_loader_val = \
        torch.utils.data.DataLoader(target_dataset_val,
                                    batch_size=min(bs,
                                                   len(target_dataset_val)),
                                    num_workers=3,
                                    shuffle=True, drop_last=True)
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=bs * 2, num_workers=3,
                                    shuffle=True, drop_last=True)
    target_loader_test = \
        torch.utils.data.DataLoader(target_dataset_test,
                                    batch_size=bs * 2, num_workers=3,
                                    shuffle=True, drop_last=True)
    return source_loader, target_loader, target_loader_unl, \
        target_loader_val, target_loader_test, class_list



def return_dataset_dual(args):
    base_path = args.data_root + '/txt/%s' % args.dataset
    root = args.data_root + '/%s/' % args.dataset
    image_set_file_s = \
        os.path.join(base_path,
                     'labeled_source_images_' +
                     args.source + '.txt')
    image_set_file_t = \
        os.path.join(base_path,
                     'labeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))
    image_set_file_t_val = \
        os.path.join(base_path,
                     'validation_target_images_' +
                     args.target + '_3.txt')
    image_set_file_unl = \
        os.path.join(base_path,
                     'unlabeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size, padding=int(crop_size*0.125)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size, padding=int(crop_size*0.125)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    source_dataset = Imagelists_VISDA(image_set_file_s, root=root, transform=DualTransformFixMatch(mean=mean, std=std))
    target_dataset = Imagelists_VISDA(image_set_file_t, root=root, transform=data_transforms['val'])
 
    target_dataset_val = Imagelists_VISDA(image_set_file_t_val, root=root,
                                          transform=data_transforms['val'])
    target_dataset_unl = Imagelists_VISDA(image_set_file_unl, root=root,
                                          transform=TransformFixMatch(mean=mean, std=std))
    target_dataset_test = Imagelists_VISDA(image_set_file_unl, root=root,
                                           transform=data_transforms['test'])
    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    if args.net == 'alexnet':
        bs = args.bs * 1
    elif args.net == 'vgg':
        bs = 24
    else:
        bs = args.bs
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=bs,
                                                num_workers=4, shuffle=True,
                                                drop_last=True)
    target_loader = \
        torch.utils.data.DataLoader(target_dataset,
                                    batch_size=min(bs, len(target_dataset)),
                                    num_workers=4,
                                    shuffle=True, drop_last=True)
    target_loader_val = \
        torch.utils.data.DataLoader(target_dataset_val,
                                    batch_size=min(bs,
                                                   len(target_dataset_val)),
                                    num_workers=4,
                                    shuffle=True, drop_last=True)
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=bs * 2, num_workers=4,
                                    shuffle=True, drop_last=True)
    target_loader_test = \
        torch.utils.data.DataLoader(target_dataset_test,
                                    batch_size=bs * 2, num_workers=4,
                                    shuffle=True, drop_last=True)
    return source_loader, target_loader, target_loader_unl, \
        target_loader_val, target_loader_test, class_list


def return_dataset_uda(args):
    base_path = args.data_root + '/txt/%s' % args.dataset
    root = args.data_root + '/%s/' % args.dataset
    image_set_file_s = \
        os.path.join(base_path,
                     'labeled_source_images_' +
                     args.source + '.txt')
    image_set_file_s_val = \
        os.path.join(base_path,
                     'validation_target_images_' +
                     args.source + '_3.txt')
    image_set_file_t = \
        os.path.join(base_path,
                     'unlabeled_target_images_' +
                     args.target + '_%d.txt' % (args.num+1))
    image_set_file_t_val = \
        os.path.join(base_path,
                     'validation_target_images_' +
                     args.target + '_3.txt')
    image_set_file_unl = \
        os.path.join(base_path,
                     'unlabeled_target_images_' +
                     args.target + '_%d.txt' % (args.num+1))

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    source_dataset = Imagelists_VISDA(image_set_file_s, root=root,
                                      transform=data_transforms['train'])
    target_dataset = Imagelists_VISDA(image_set_file_t, root=root,
                                      transform=data_transforms['val'])
    source_dataset_val = Imagelists_VISDA(image_set_file_s_val, root=root,
                                          transform=data_transforms['val'])
    target_dataset_val = Imagelists_VISDA(image_set_file_t_val, root=root,
                                          transform=data_transforms['val'])
    target_dataset_unl = Imagelists_VISDA(image_set_file_unl, root=root,
                                          transform=data_transforms['val'])
    target_dataset_test = Imagelists_VISDA(image_set_file_unl, root=root,
                                           transform=data_transforms['test'])
    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    if args.net == 'alexnet':
        bs = args.bs
    else:
        bs = args.bs
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=bs,
                                                num_workers=4, shuffle=True,
                                                drop_last=True)
    source_loader_val = torch.utils.data.DataLoader(source_dataset_val, batch_size=bs,
                                                num_workers=4, shuffle=True,
                                                drop_last=True)
    target_loader = \
        torch.utils.data.DataLoader(target_dataset,
                                    batch_size=min(bs, len(target_dataset)),
                                    num_workers=4,
                                    shuffle=True, drop_last=True)
    target_loader_val = \
        torch.utils.data.DataLoader(target_dataset_val,
                                    batch_size=min(bs,
                                                   len(target_dataset_val)),
                                    num_workers=4,
                                    shuffle=True, drop_last=True)
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=bs * 2, num_workers=4,
                                    shuffle=True, drop_last=True)
    target_loader_test = \
        torch.utils.data.DataLoader(target_dataset_test,
                                    batch_size=bs * 2, num_workers=4,
                                    shuffle=True, drop_last=True)
    return source_loader, target_loader, target_loader_unl, \
        target_loader_val, target_loader_test, class_list, source_loader_val


def return_dataset_test(args):
    base_path = '../data/txt/%s' % args.dataset
    root = '../data/%s/' % args.dataset
    image_set_file_s = \
        os.path.join(base_path,
                     'labeled_source_images_' +
                     args.source + '.txt')
    image_set_file_test = os.path.join(base_path,
                                       'unlabeled_target_images_' +
                                       args.target + '_%d.txt' % (args.num))
    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    target_dataset_unl = Imagelists_VISDA(image_set_file_test, root=root,
                                          transform=data_transforms['test'],
                                          test=True)
    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=bs * 2, num_workers=4,
                                    shuffle=False, drop_last=False)
    return target_loader_unl, class_list

def return_dataset_source(args):
    base_path = args.data_root + '/txt/%s' % args.dataset
    root = args.data_root + '/%s/' % args.dataset

    image_set_file_s_val = \
        os.path.join(base_path,
                     'labeled_target_images_' +
                     args.source + '_3.txt' )

    image_set_file_s = \
        os.path.join(base_path,
                     'unlabeled_target_images_' +
                     args.source + '_3.txt')

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    source_dataset = Imagelists_VISDA(image_set_file_s, root=root,
                                      transform=data_transforms['train'])

    source_dataset_val = Imagelists_VISDA(image_set_file_s_val, root=root,
                                          transform=data_transforms['val'])
    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    if args.net == 'alexnet':
        bs = args.bs * 1
    elif args.net == 'vgg':
        bs = 24
    else:
        bs = args.bs

    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=bs,
                                                num_workers=4, shuffle=True,
                                                drop_last=True)
    
    source_loader_val = \
        torch.utils.data.DataLoader(source_dataset_val,
                                    batch_size=min(bs,
                                                   len(source_dataset_val)),
                                    num_workers=4,
                                    shuffle=True, drop_last=True)
    
    return source_loader, source_loader_val, class_list

def return_dataset_target(args):
    base_path = args.data_root + '/txt/%s' % args.dataset
    root = args.data_root + '/%s/' % args.dataset

    image_set_file_t = \
        os.path.join(base_path,
                     'labeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))
    image_set_file_t_val = \
        os.path.join(base_path,
                     'validation_target_images_' +
                     args.target + '_3.txt')
    image_set_file_unl = \
        os.path.join(base_path,
                     'unlabeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    target_dataset = Imagelists_VISDA(image_set_file_t, root=root,
                                      transform=data_transforms['val'])
    target_dataset_val = Imagelists_VISDA(image_set_file_t_val, root=root,
                                          transform=data_transforms['val'])
    target_dataset_unl = Imagelists_VISDA(image_set_file_unl, root=root,
                                          transform=data_transforms['val'])
    target_dataset_test = Imagelists_VISDA(image_set_file_unl, root=root,
                                           transform=data_transforms['test'])
    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    if args.net == 'alexnet':
        bs = args.bs * 1
    elif args.net == 'vgg':
        bs = 24
    else:
        bs = args.bs

    target_loader = \
        torch.utils.data.DataLoader(target_dataset,
                                    batch_size=min(bs, len(target_dataset)),
                                    num_workers=4,
                                    shuffle=True, drop_last=True)
    target_loader_val = \
        torch.utils.data.DataLoader(target_dataset_val,
                                    batch_size=min(bs,
                                                   len(target_dataset_val)),
                                    num_workers=4,
                                    shuffle=True, drop_last=True)
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=bs * 2, num_workers=4,
                                    shuffle=True, drop_last=True)
    target_loader_test = \
        torch.utils.data.DataLoader(target_dataset_test,
                                    batch_size=bs * 2, num_workers=4,
                                    shuffle=True, drop_last=True)
    return target_loader, target_loader_unl, \
        target_loader_val, target_loader_test, class_list
