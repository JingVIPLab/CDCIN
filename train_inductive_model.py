from model.trainer.fsl_trainer import FSLTrainer

import time
import os
import torch
import pprint
import argparse
import numpy as np
import random

def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.

    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth]))
    if indices.is_cuda:
        encoded_indicies = encoded_indicies.cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)

    return encoded_indicies

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def set_init_weight(args):
    init_path = {'Res12':'your pretrained modal path',
                'SwinT':'your pretrained modal path',
                'VitS':'your pretrained modal path'}
                
    args.init_weights = init_path[args.backbone_class]
    


_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)

def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm

def postprocess_args(args):            
    args.num_classes = args.way
    save_path1 = '-'.join([args.dataset, args.model_class, args.backbone_class, '{:02d}w{:02d}s{:02}q'.format(args.way, args.shot, args.query)])
    save_path2 = '_'.join([
                           str(time.strftime('%Y%m%d_%H%M%S'))
                           ])
            
    if not os.path.exists(os.path.join(args.save_dir, save_path1)):
        os.mkdir(os.path.join(args.save_dir, save_path1))
    args.save_path = os.path.join(args.save_dir, save_path1, save_path2)
    return args

def get_command_line_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=60)
    parser.add_argument('--episodes_per_epoch', type=int, default=100)
    parser.add_argument('--num_eval_episodes', type=int, default=200)

    parser.add_argument('--model_class', type=str, default='ProtoNet', 
                        choices=['ProtoNet'])

    parser.add_argument('--use_euclidean', action='store_true', default=False)    
    parser.add_argument('--backbone_class', type=str, default='Res12',
                        choices=['Res12', 'SwinT', 'VitS'])
    parser.add_argument('--dataset', type=str, default='COCO',
                        choices=['COCO', 'VG_QA', 'VQAv2'])
    
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--eval_way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--eval_shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=5)
    parser.add_argument('--eval_query', type=int, default=5)
    parser.add_argument('--unlabeled', type=int, default=0)
    parser.add_argument('--eval_unlabeled', type=int, default=0)
    parser.add_argument('--balance', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=64)
    parser.add_argument('--temperature2', type=float, default=32)
    parser.add_argument('--batch', type=int, default=1)

    parser.add_argument('--seq_len', type=int, default=49)
    parser.add_argument('--block_mask_1shot', default=5, type=int, help="""Number of patches to mask around each 
                        respective patch during online-adaptation in 1shot scenarios: masking along main diagonal,
                        e.g. size=5 corresponds to masking along the main diagonal of 'width' 5.""")
    parser.add_argument('--similarity_temp_init', type=float, default=0.051031036307982884,
                        help="""Initial value of temperature used for scaling the logits of the path embedding 
                            similarity matrix. Logits will be divided by that temperature, i.e. temp<1 scales up. 
                            'similarity_temp' must be positive.""")
    parser.add_argument('--optimiser_online', default='SGD', type=str, choices=['SGD'],
                        help="""Optimiser to be used for adaptation of patch embedding importance vector.""")
    parser.add_argument('--lr_online', default=0.1, type=float, help="""Learning rate used for online optimisation.""")
    parser.add_argument('--optim_steps_online', default=30, type=int, help="""Number of update steps to take to
                                optimise the patch embedding importance vector.""")
    parser.add_argument('--disable_peiv_optimisation', type=bool, default=False,
                        help="""Disable the patch embedding importance vector (peiv) optimisation/adaptation.
                                This means that inference is performed simply based on the cosine similarity of support and
                                query set sample embeddings, w/o any further adaptation.""")
     
    # optimization parameters
    parser.add_argument('--orig_imsize', type=int, default=-1) # -1 for no cache, and -2 for no resize, only for MiniImageNet and CUB
    parser.add_argument('--lr', type=float, default=0.000025)
    parser.add_argument('--lr_mul', type=float, default=10)    
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['multistep', 'step', 'cosine'])
    parser.add_argument('--step_size', type=str, default='20')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--fix_BN', action='store_true', default=False)     # means we do not update the running mean/var in BN, not to freeze BN
    parser.add_argument('--augment',   action='store_true', default=False)
    parser.add_argument('--multi_gpu', action='store_true', default=False)

    parser.add_argument('--init_weights', type=str, default='./initialization/miniimagenet/Res12-pre.pth' )
    
    # usually untouched parameters
    parser.add_argument('--mom', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005) # we find this weight decay value works the best
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')

    parser.add_argument('--gpu', default='1')
    parser.add_argument('--seed', type=int, default=2222,
                        help='random seed')
    return parser

def main():
    parser = get_command_line_parser()
    args = postprocess_args(parser.parse_args())
    set_init_weight(args)

    pprint(vars(args))

    set_gpu(args.gpu)
    # set_seed(args.seed)
    trainer = FSLTrainer(args)
    trainer.train()
    trainer.evaluate_test()
    trainer.final_record()
    print(args.save_path)


if __name__ == '__main__':
    main()


