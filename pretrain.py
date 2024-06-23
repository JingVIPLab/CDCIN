import argparse
import os
import os.path as osp
import shutil
import torch
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import DataLoader
from model.models.vqa_classifier_entropy import Classifier
from model.dataloader.samplers import CategoriesSampler
from model.utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.cuda.amp import autocast as autocast, GradScaler
# pre-train model, compute validation acc after 500 epoches

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def set_init_weight(args):
    init_path = {'Res12':'./initialization/miniimagenet/Res12-pre.pth',
                'SwinT':'./initialization/SwinT-Pre/swint_checkpoint.pth',
                'VitS':'./initialization/Vit-Pre/vit_checkpoint.pth'}
    args.init_weights = init_path[args.backbone_class]

def make_nk_label(n, k, ep_per_batch=1):
        label = torch.arange(n).unsqueeze(1).expand(n, k).reshape(-1)
        label = label.repeat(ep_per_batch)
        return label

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.000025)
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--dataset', type=str, default='VG_QA', choices=['COCO', 'VG_QA', 'VQAv2'])    
    parser.add_argument('--backbone_class', type=str, default='VitS', choices=['Res12','SwinT','VitS'])
    parser.add_argument('--schedule', type=int, nargs='+', default=[25, 50, 75], help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=5)
    parser.add_argument('--init_weights', type=str, default='./initialization/miniimagenet/Res12-pre.pth' )
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--gpu', default='2')
    parser.add_argument('--seed', type=int, default=2222,
                        help='random seed')
    args = parser.parse_args()
    args.orig_imsize = -1
    pprint(vars(args))

    set_gpu(args.gpu)
    set_seed(args.seed)
    set_init_weight(args)

    save_path1 = '-'.join([args.dataset, args.backbone_class, 'Pre'])
    save_path2 = '_'.join([str(args.lr), str(args.gamma), str(args.schedule)])
    args.save_path = osp.join(save_path1, save_path2)
    if not osp.exists(save_path1):
        os.mkdir(save_path1)
    ensure_path(args.save_path)

    from model.dataloader.fsl_vqa import FSLVQA as Dataset

    trainset = Dataset('train', args, use_fapit=True)
    valset = Dataset('test', args, token_to_ix=trainset.token_to_ix, use_fapit=True)

    # trainset = Dataset('train', args)
    # valset = Dataset('val', args, token_to_ix=trainset.token_to_ix)

    args.num_class = trainset.num_class
    args.pretrained_emb = trainset.pretrained_emb
    args.token_size = trainset.token_size

    train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_sampler = CategoriesSampler(valset.label, 200, args.way, args.shot + args.query) # test on 16-way 1-shot
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=8, pin_memory=True)
    
    


    # construct model
    model = Classifier(args)
    if args.init_weights is not None:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.init_weights)['params']
        if args.backbone_class in ['SwinT', 'VitS']:
            pretrained_dict = {'encoder.encoder.' + k: v for k, v in pretrained_dict.items()}
        elif args.backbone_class in ['Res12']:
            pretrained_dict = {'encoder.' + k: v for k, v in pretrained_dict.items()}
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # print(pretrained_dict.keys())
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    model.encoder.encoder.requires_grad_(False)

    if 'Res12' in args.backbone_class:
        # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)

    criterion = torch.nn.CrossEntropyLoss()
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        # if args.ngpu  > 1:
        #     model.encoder = torch.nn.DataParallel(model.encoder, device_ids=list(range(args.ngpu)))
        
        model = model.cuda()
        criterion = criterion.cuda()
    
    def save_model(name):
        torch.save(dict(params=model.state_dict()), osp.join(args.save_path, name + '.pth'))
    
    def save_checkpoint(is_best, filename='checkpoint.pth.tar'):
        state = {'epoch': epoch + 1,
                 'args': args,
                 'state_dict': model.state_dict(),
                 'trlog': trlog,
                 'val_acc_dist': trlog['max_acc_dist'],
                 'val_acc_sim': trlog['max_acc_sim'],
                 'optimizer' : optimizer.state_dict(),
                 'global_count': global_count}
        
        torch.save(state, osp.join(args.save_path, filename))
        if is_best:
            shutil.copyfile(osp.join(args.save_path, filename), osp.join(args.save_path, 'model_best.pth.tar'))
    
    if args.resume == True:
        # load checkpoint
        state = torch.load(osp.join(args.save_path, 'model_best.pth.tar'))
        init_epoch = state['epoch']
        resumed_state = state['state_dict']
        # resumed_state = {'module.'+k:v for k,v in resumed_state.items()}
        model.load_state_dict(resumed_state)
        trlog = state['trlog']
        optimizer.load_state_dict(state['optimizer'])
        initial_lr = optimizer.param_groups[0]['lr']
        global_count = state['global_count']
    else:
        init_epoch = 1
        trlog = {}
        trlog['args'] = vars(args)
        trlog['train_loss'] = []
        trlog['val_loss_dist'] = []
        trlog['val_loss_sim'] = []
        trlog['train_acc'] = []
        trlog['val_acc_sim'] = []
        trlog['val_acc_dist'] = []
        trlog['max_acc_dist'] = 0.0
        trlog['max_acc_dist_epoch'] = 0
        trlog['max_acc_sim'] = 0.0
        trlog['max_acc_sim_epoch'] = 0        
        initial_lr = args.lr
        global_count = 0

    timer = Timer()
    writer = SummaryWriter(logdir=args.save_path)
    for epoch in range(init_epoch, args.max_epoch + 1):
        # refine the step-size
        if epoch in args.schedule:
            initial_lr *= args.gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_lr
        
        model.train()
        tl = Averager()
        ta = Averager()

        train_gen = tqdm(train_loader)
        for i, batch in enumerate(train_gen, 1):

        # for i, batch in enumerate(train_loader, 1):
            global_count = global_count + 1
            if torch.cuda.is_available():
                data, que, label = [_.cuda() for _ in batch]
                label = label.type(torch.cuda.LongTensor)
            else:
                data, que, label = batch
                label = label.type(torch.LongTensor)
            with autocast():
                
                logits, recon_loss = model(data, que)
                loss = criterion(logits, label) + recon_loss

            acc = count_acc(logits, label)*100
            writer.add_scalar('data/loss', float(loss), global_count)
            writer.add_scalar('data/acc', float(acc), global_count)
            
            # if (i-1) % 100 == 0:
            #     print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'.format(epoch, i, len(train_loader), loss.item(), acc))

            tl.add(loss.item())
            ta.add(acc)

            train_gen.set_description('训练阶段:epo {} 平均loss={:.4f}  平均acc={:.4f}'.format(epoch, tl.item(), ta.item()))


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        tl = tl.item()
        ta = ta.item()

        # do not do validation in first 500 epoches
        if epoch < 50 or (epoch-1) % 5 == 0:
            model.eval()
            vl_dist = Averager()
            va_dist = Averager()
            vl_sim = Averager()
            va_sim = Averager()            
            # print('[Dist] best epoch {}, current best val acc={:.4f}'.format(trlog['max_acc_dist_epoch'], trlog['max_acc_dist']))
            # print('[Sim] best epoch {}, current best val acc={:.4f}'.format(trlog['max_acc_sim_epoch'], trlog['max_acc_sim']))
            # test performance with Few-Shot
            label = torch.arange(args.way).repeat(args.query)
            if torch.cuda.is_available():
                label = label.type(torch.cuda.LongTensor)
            else:
                label = label.type(torch.LongTensor)        
            with torch.no_grad():
                val_gen = tqdm(val_loader)
                for i, batch in enumerate(val_gen, 1):
                # for i, batch in tqdm(enumerate(val_loader, 1)):
                    if torch.cuda.is_available():
                        data, que, _ = [_.cuda() for _ in batch]
                    else:
                        data, que, _ = batch
                    data_shot, data_query = data[:args.way], data[args.way:] # 16-way test
                    que_shot, que_query = que[:args.way], que[args.way:]
                    logits_dist, logits_sim = model.forward_proto(data_shot, data_query, que_shot, que_query, args.way)
                    loss_dist = F.cross_entropy(logits_dist, label)
                    acc_dist = count_acc(logits_dist, label)*100
                    loss_sim = F.cross_entropy(logits_sim, label)
                    acc_sim = count_acc(logits_sim, label)*100                  
                    vl_dist.add(loss_dist.item())
                    va_dist.add(acc_dist)
                    vl_sim.add(loss_sim.item())
                    va_sim.add(acc_sim)

                    val_gen.set_description('Val:epo {} vl_dist={:.4f} vl_sim={:.4f}  acc_dist={:.4f} acc_sim={:.4f}'.format(epoch, vl_dist.item(),vl_sim.item(),va_dist.item(), va_sim.item()))


            vl_dist = vl_dist.item()
            va_dist = va_dist.item()
            vl_sim = vl_sim.item()
            va_sim = va_sim.item()            
            writer.add_scalar('data/val_loss_dist', float(vl_dist), epoch)
            writer.add_scalar('data/val_acc_dist', float(va_dist), epoch)     
            writer.add_scalar('data/val_loss_sim', float(vl_sim), epoch)
            writer.add_scalar('data/val_acc_sim', float(va_sim), epoch)               
            # print('epoch {}, val, loss_dist={:.4f} acc_dist={:.4f} loss_sim={:.4f} acc_sim={:.4f}'.format(epoch, vl_dist, va_dist, vl_sim, va_sim))
    
            if va_dist > trlog['max_acc_dist']:
                trlog['max_acc_dist'] = va_dist
                trlog['max_acc_dist_epoch'] = epoch
                save_model('max_acc_dist')
                save_checkpoint(True)
                
            if va_sim > trlog['max_acc_sim']:
                trlog['max_acc_sim'] = va_sim
                trlog['max_acc_sim_epoch'] = epoch
                save_model('max_acc_sim')
                save_checkpoint(True)            
    
            trlog['train_loss'].append(tl)
            trlog['train_acc'].append(ta)
            trlog['val_loss_dist'].append(vl_dist)
            trlog['val_acc_dist'].append(va_dist)
            trlog['val_loss_sim'].append(vl_sim)
            trlog['val_acc_sim'].append(va_sim)            
            save_model('epoch-last')
    
            print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))
    writer.close()
    
    
    # import pdb
    # pdb.set_trace()