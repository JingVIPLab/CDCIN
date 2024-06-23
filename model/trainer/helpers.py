import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.dataloader.samplers import CategoriesSampler_meta, CategoriesSampler_metaVQA

from model.models.protonet import ProtoNet

# no usage
class MultiGPUDataloader:
    def __init__(self, dataloader, num_device):
        self.dataloader = dataloader
        self.num_device = num_device

    def __len__(self):
        return len(self.dataloader) // self.num_device

    def __iter__(self):
        data_iter = iter(self.dataloader)
        done = False

        while not done:
            try:
                output_batch = ([], [])
                for _ in range(self.num_device):
                    batch = next(data_iter)
                    for i, v in enumerate(batch):
                        output_batch[i].append(v[None])

                yield (torch.cat(_, dim=0) for _ in output_batch)
            except StopIteration:
                done = True
        return


def get_dataloader(args):
    # if args.dataset == 'MiniImageNet':
    #     # Handle MiniImageNet
    #     from model.dataloader.mini_imagenet import MiniImageNet as Dataset
    # elif args.dataset == 'CUB':
    #     from model.dataloader.cub import CUB as Dataset
    # elif args.dataset == 'TieredImageNet':
    #     from model.dataloader.tiered_imagenet import tieredImageNet as Dataset
    # else:
    #     raise ValueError('Non-supported Dataset.')
    from model.dataloader.fsl_vqa import FSLVQA as Dataset

    num_device = torch.cuda.device_count()  
    num_episodes = args.episodes_per_epoch * num_device if args.multi_gpu else args.episodes_per_epoch 
    num_workers = 8

    # trainset = Dataset('train', args, augment=args.augment)
    # valset = Dataset('val', args, token_to_ix=trainset.token_to_ix)
    # testset = Dataset('test', args, token_to_ix=trainset.token_to_ix)

    trainset = Dataset('train', args, augment=args.augment, use_fapit=True)
    valset = Dataset('test', args, token_to_ix=trainset.token_to_ix, use_fapit=True)
    testset = Dataset('test', args, token_to_ix=trainset.token_to_ix, use_fapit=True)
    args.num_class = trainset.num_class 

    train_sampler = CategoriesSampler_metaVQA(trainset.label2ind,
                                           num_episodes,
                                           max(args.way, args.num_classes),
                                           args.shot + args.query + args.unlabeled, args.batch)
    train_loader = DataLoader(dataset=trainset,
                              num_workers=num_workers,
                              batch_sampler=train_sampler,
                              pin_memory=True)
    val_sampler = CategoriesSampler_metaVQA(valset.label2ind,
                                         args.num_eval_episodes,
                                         args.eval_way, args.eval_shot + args.eval_query + args.eval_unlabeled, args.batch)
    val_loader = DataLoader(dataset=valset,
                            batch_sampler=val_sampler,
                            num_workers=args.num_workers,
                            pin_memory=True)
    test_sampler = CategoriesSampler_metaVQA(testset.label2ind,
                                          int(10000 / args.batch),  # args.num_eval_episodes,
                                          args.eval_way, args.eval_shot + args.eval_query + args.eval_unlabeled, args.batch)
    test_loader = DataLoader(dataset=testset,
                             batch_sampler=test_sampler,
                             num_workers=args.num_workers,
                             pin_memory=True)
    
    args.pretrained_emb = trainset.pretrained_emb
    args.token_size = trainset.token_size

    return train_loader, val_loader, test_loader


def prepare_model(args):
    model = eval(args.model_class)(args)

    # load pre-trained model (no FC weights)
    if args.init_weights is not None:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.init_weights)['params']
        # if args.backbone_class in ['SwinT', 'VitS']:
            # pretrained_dict = {'encoder.backbone.' + k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {'encoder.' + k: v for k, v in pretrained_dict.items()}
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k != 'encoder.que_encoder.embedding.weight'}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # print(pretrained_dict.keys())
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    if args.multi_gpu:
        model.encoder = nn.DataParallel(model.encoder, dim=0)
        para_model = model.to(device)
    else:
        para_model = model.to(device)

    return model, para_model


def prepare_optimizer(model, args):
    top_para = [v for k, v in model.named_parameters() if 'encoder' not in k]

    # as in the literature, we use ADAM for ConvNet and SGD for other backbones
    if args.backbone_class == 'Res12':
        optimizer = optim.SGD(
            # [{'params': model.backbone.encoder.parameters()},
            [{'params': model.encoder.parameters()},
             {'params': top_para, 'lr': args.lr * args.lr_mul}],
            lr=args.lr,
            momentum=args.mom,
            nesterov=True,
            weight_decay=args.weight_decay
        )
    else:
        
        # optimizer = optim.Adam(
        #     # [{'params': model.backbone.encoder.parameters()},
        #     [{'params': model.encoder.parameters()},
        #      {'params': top_para, 'lr': args.lr * args.lr_mul}],
        #     lr=args.lr,
        #     # weight_decay=args.weight_decay, do not use weight_decay here
        # )
        optimizer = optim.SGD(
            # [{'params': model.backbone.encoder.parameters()},
            [{'params': model.encoder.parameters()},
             {'params': top_para, 'lr': args.lr * args.lr_mul}],
            lr=args.lr,
            momentum=args.mom,
            nesterov=True,
            weight_decay=args.weight_decay)

    if args.lr_scheduler == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(args.step_size),
            gamma=args.gamma
        )
    elif args.lr_scheduler == 'multistep':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(_) for _ in args.step_size.split(',')],
            gamma=args.gamma,
        )
    elif args.lr_scheduler == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            args.max_epoch,
            eta_min=0  # a tuning parameter
        )
    else:
        raise ValueError('No Such Scheduler')

    return optimizer, lr_scheduler
