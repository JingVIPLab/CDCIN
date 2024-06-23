import time
import os.path as osp
import numpy as np

import torch
import torch.nn.functional as F

from model.trainer.base import Trainer
from model.trainer.helpers import (
    get_dataloader, prepare_model, prepare_optimizer,
)
from model.utils import (
    Averager, count_acc,
    compute_confidence_interval,
)

from torch.cuda.amp import autocast as autocast
from tqdm import tqdm

class FSLTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)  

        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        self.model, self.para_model = prepare_model(args)
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.model, args)

    def prepare_label(self):
        args = self.args

        # prepare one-hot label
        label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)  # [1,75]
        label_aux = torch.arange(args.way, dtype=torch.int8).repeat(args.shot + args.query)  # [1,80]
        
        label = label.type(torch.LongTensor)
        label_aux = label_aux.type(torch.LongTensor)
        
        if torch.cuda.is_available():
            label = label.cuda()
            label_aux = label_aux.cuda()
            
        return label, label_aux

    def make_nk_label(self,n, k, ep_per_batch=1):
        label = torch.arange(n).unsqueeze(1).expand(n, k).reshape(-1)
        label = label.repeat(ep_per_batch)
        return label
    
    def train(self):
        args = self.args
        self.model.train() 
        if self.args.fix_BN:
            self.model.encoder.eval()    

        support_label = self.make_nk_label(args.way, args.shot, args.batch).cuda()
        label = self.make_nk_label(args.way, args.query, args.batch).cuda()
        label_aux = self.make_nk_label(args.way, args.shot + args.query, args.batch).cuda()
        label_align= self.make_nk_label((args.shot + args.query) * args.way, 1, args.batch).cuda()

        for epoch in range(1, args.max_epoch + 1):
            self.train_epoch += 1

            self.model.train()
            if self.args.fix_BN:
                self.model.encoder.eval()
            
            tl1 = Averager()   # training loss for all epoch
            tl2 = Averager()   # training loss for current epoch
            ta = Averager()    # training accuracy for current epoch

            start_tm = time.time()  
            train_gen = tqdm(self.train_loader)
            for i, batch in enumerate(train_gen, 1):
                self.train_step += 1

                # data => [80,3,84,84] gt_label => [1,80]
                if torch.cuda.is_available():
                    data, que, gt_label = [_.cuda() for _ in batch]
                else:
                    data, que, gt_label = batch[0], batch[1], batch[2]

                data_tm = time.time()
                self.dt.add(data_tm - start_tm)

                with autocast():
                    # logits, recon_loss, align_logits = self.para_model(data, que, support_label)
                    # loss = F.cross_entropy(logits, label) + recon_loss + F.cross_entropy(align_logits, label_align)
                    logits, recon_loss = self.para_model(data, que, support_label)
                    loss = F.cross_entropy(logits, label) + recon_loss
                    # logits, recon_loss, reg_logits = self.para_model(data, que, support_label)
                    # loss = 0.5 * F.cross_entropy(logits, label) + recon_loss + 0.5 * F.cross_entropy(reg_logits, label_aux)
                    # logits = self.para_model(data, que, support_label)
                    # loss = F.cross_entropy(logits, label)
                    # pre_logits, logits = self.para_model(data, que, support_label)
                    # loss = F.cross_entropy(logits, label) + F.cross_entropy(pre_logits, label)
                    
                tl2.add(loss)

                forward_tm = time.time()
                self.ft.add(forward_tm - data_tm)

                acc = count_acc(logits, label)

                tl1.add(loss.item())
                ta.add(acc)

                train_gen.set_description(
                    'Train:epo {} total_loss={:.4f} partial_loss={:.4f} mean_acc={:.4f}'.format(epoch, tl1.item(), tl2.item(), ta.item()))

                self.optimizer.zero_grad()
                loss.backward()
                backward_tm = time.time()
                self.bt.add(backward_tm - forward_tm)
                self.optimizer.step()
                optimizer_tm = time.time()
                self.ot.add(optimizer_tm - backward_tm)
                start_tm = time.time()

            self.lr_scheduler.step()
            self.try_evaluate(epoch)
            print('ETA:{}/{}'.format(
                    self.timer.measure(),
                    self.timer.measure(self.train_epoch / args.max_epoch))
            )

        torch.save(self.trlog, osp.join(args.save_path, 'trlog'))
        self.save_model('epoch-last')

    def evaluate(self, data_loader):
        args = self.args
        self.model.eval()

        record = np.zeros((args.num_eval_episodes, 2))
        support_label = self.make_nk_label(args.way, args.shot, args.batch).cuda()
        label = self.make_nk_label(args.way, args.query, args.batch).cuda()
        label_align= self.make_nk_label((args.shot + args.query) * args.way, 1, args.batch).cuda()

        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
        with torch.no_grad():
            val_gen = tqdm(data_loader)

            tl1 = Averager()
            ta = Averager()

            for i, batch in enumerate(val_gen, 1):
                if torch.cuda.is_available():
                    data, que, _ = [_.cuda() for _ in batch]
                else:
                    data, que = batch[0], batch[1]
                
                # logits, recon_loss, align_logits = self.para_model(data, que, support_label)
                # loss = F.cross_entropy(logits, label) + recon_loss + F.cross_entropy(align_logits, label_align)
                logits, recon_loss = self.model(data, que, support_label)
                loss = F.cross_entropy(logits, label) + recon_loss
                # logits = self.model(data, que, support_label)
                # loss = F.cross_entropy(logits, label)
                # pre_logits, logits = self.para_model(data, que, support_label)
                # loss = F.cross_entropy(logits, label) + F.cross_entropy(pre_logits, label)
                acc = count_acc(logits, label)
                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc
                
                tl1.add(loss)
                ta.add(acc)

                val_gen.set_description('Val:mean_loss1={:.4f} mean_acc={:.4f}'.format(tl1.item(), ta.item()))
                
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])
        
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()

        return vl, va, vap

    def evaluate_test(self):
        # restore model args
        args = self.args
        # evaluation mode
        self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc.pth'))['params'])
        self.model.eval()
        record = np.zeros((int(10000/self.args.batch), 2)) # loss and acc
        support_label = self.make_nk_label(args.way, args.shot, args.batch).cuda()
        label = self.make_nk_label(args.way, args.query, args.batch).cuda()
        label_align= self.make_nk_label((args.shot + args.query) * args.way, 1, args.batch).cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
        with torch.no_grad():
            tl1 = Averager()
            ta = Averager()
            test_gen = tqdm(self.test_loader)

            for i, batch in enumerate(test_gen, 1):
                if torch.cuda.is_available():
                    data, que, _ = [_.cuda() for _ in batch]
                else:
                    data, que = batch[0], batch[1]
                # with torch.enable_grad():
                # logits, recon_loss, align_logits = self.para_model(data, que, support_label)
                # loss = F.cross_entropy(logits, label) + recon_loss + F.cross_entropy(align_logits, label_align)
                logits, recon_loss = self.model(data, que, support_label)
                loss = F.cross_entropy(logits, label) + recon_loss
                # logits = self.model(data, que, support_label)
                # loss = F.cross_entropy(logits, label)
                # pre_logits, logits = self.para_model(data, que, support_label)
                # loss = F.cross_entropy(logits, label) + F.cross_entropy(pre_logits, label)
                acc = count_acc(logits, label)
                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc

                tl1.add(loss)
                ta.add(acc)

                test_gen.set_description('Test:mean_loss1={:.4f} mean_acc={:.4f}'.format(tl1.item(), ta.item()))

        assert (i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])
        
        self.trlog['test_acc'] = va
        self.trlog['test_acc_interval'] = vap
        self.trlog['test_loss'] = vl

        torch.save(self.model.state_dict(), args.save_path + '_{}.pth'.format(va))

        print('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
        print('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']))

        return vl, va, vap
    
    def final_record(self):
        # save the best performance in a txt file
        with open(osp.join(self.args.save_path, '{}+{}'.format(self.trlog['test_acc'], self.trlog['test_acc_interval'])), 'w') as f:
            f.write('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
            f.write('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']))            