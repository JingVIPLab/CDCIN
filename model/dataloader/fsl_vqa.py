import torch
import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import en_vectors_web_lg
import os, re

IMAGE_PATH = {'VQAv2' : 'your image path',
             'COCO' : 'your image path', 
             'VG_QA' : 'your image path'}
SPLIT_PATH = {'VQAv2' : 'your split file path',
             'COCO' : 'your split file path',
             'VG_QA' : 'your split file path'}

def identity(x):
    return x


def split(que):
    words = re.sub(
        r"([.,'!?\"()*#:;])",
        '',
        que.lower()
    ).replace('-', ' ').replace('/', ' ').split()
    return words


def proc_ques(ques, token_to_ix, max_token):
    ques_ix = np.zeros(max_token, np.int64)

    words = split(ques)
    
    for ix, word in enumerate(words):
        if word in token_to_ix:
            ques_ix[ix] = token_to_ix[word]
        else:
            ques_ix[ix] = token_to_ix['UNK']

        if ix + 1 == max_token:
            break

    return ques_ix


def tokenize(total_words, use_glove=True):
    token_to_ix = {
        'PAD': 0,
        'UNK': 1,
    }

    spacy_tool = None
    pretrained_emb = []
    if use_glove:
        spacy_tool = en_vectors_web_lg.load()
        pretrained_emb.append(spacy_tool('PAD').vector)
        pretrained_emb.append(spacy_tool('UNK').vector)

    for word in total_words:
        if word not in token_to_ix:
            token_to_ix[word] = len(token_to_ix)
            if use_glove:
                pretrained_emb.append(spacy_tool(word).vector)

    pretrained_emb = np.array(pretrained_emb)

    return token_to_ix, pretrained_emb


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)
    return label2inds


class FSLVQA(Dataset):
    """ Usage:
    """
    def __init__(self, setname, args, augment=False, max_token=15, token_to_ix=None, use_fapit=False):
        img_path = IMAGE_PATH[args.dataset]
        sp_path = SPLIT_PATH[args.dataset]
        self.max_token = max_token
        if use_fapit:
            data_path = osp.join(sp_path, setname + '_fpait.pth')
        else:
            data_path = osp.join(sp_path, setname + '.pth')
        datas = torch.load(data_path)

        self.data = []
        self.que = []
        self.label = []
        self.ans_set = []

        for line in tqdm(datas['data'], ncols=64):
            self.que.append(line['question'])
            self.data.append(os.path.join(img_path, line['img_path']))
            if line['answer'] not in self.ans_set:
                self.ans_set.append(line['answer'])
            self.label.append(self.ans_set.index(line['answer']))

        self.num_class = len(set(self.label))

        if setname == 'train':
            self.token_to_ix, self.pretrained_emb = tokenize(datas['all_words'])
        else:  # 'val' or 'test' ,
            self.token_to_ix = token_to_ix
        self.token_size = len(self.token_to_ix)
        print('Loading {} dataset -phase {}, word size {}'.format(args.dataset, setname, self.token_size))

        self.label2ind = buildLabelIndex(self.label)

        if args.backbone_class == 'Res12':
            image_size = 84
            resize = 92
        elif args.backbone_class in ['SwinT', 'VitS']:
            image_size = 224
            resize = 256
        if augment and setname == 'train':
            transforms_list = [
                  transforms.RandomResizedCrop(image_size),
                  transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                ]
        else:
            transforms_list = [
                  transforms.Resize(resize),
                  transforms.CenterCrop(image_size),
                  transforms.ToTensor(),
                ]

        # Transformation
        if args.backbone_class == 'Res12':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
            ])
        elif args.backbone_class in ['SwinT', 'VitS']:
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])            
              
        else:
            raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, que, label = self.data[i], self.que[i], self.label[i]
        
        image = self.transform(Image.open(data).convert('RGB'))
        
        que = proc_ques(que, self.token_to_ix, self.max_token)

        return image, que, label

