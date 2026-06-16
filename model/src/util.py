import torch
import numpy as np
import re


class RandHorizontalFlip(object):
    def __call__(self, sample):
        d_img = sample['d_img']
        score = sample['score']
        center = sample['center']
        source_emb = sample['source_emb']

        prob_lr = np.random.random()
        if prob_lr > 0.5:
            d_img = np.fliplr(d_img).copy()
            if 'w_img' in sample:
                sample['w_img'] = np.fliplr(sample['w_img']).copy()

        sample = {
            'd_img': d_img,
            'center': center,
            'source_emb': source_emb,
            'score': score
        }
        if 'w_img' in sample:
            sample['w_img'] = sample.get('w_img')
        return sample


class Normalize(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __call__(self, sample):
        d_img = sample['d_img']
        score = sample['score']
        center = sample['center']
        source_emb = sample['source_emb']

        d_img[..., 0] = (d_img[..., 0] - self.mean[0]) / self.var[0]
        d_img[..., 1] = (d_img[..., 1] - self.mean[1]) / self.var[1]
        d_img[..., 2] = (d_img[..., 2] - self.mean[2]) / self.var[2]
        sample['d_img'] = d_img

        if 'w_img' in sample:
            w_img = sample['w_img']
            w_img[..., 0] = (w_img[..., 0] - self.mean[0]) / self.var[0]
            w_img[..., 1] = (w_img[..., 1] - self.mean[1]) / self.var[1]
            w_img[..., 2] = (w_img[..., 2] - self.mean[2]) / self.var[2]
            sample['w_img'] = w_img

        out = {
            'd_img': sample['d_img'],
            'center': center,
            'source_emb': source_emb,
            'score': score
        }
        if 'w_img' in sample:
            out['w_img'] = sample['w_img']
        return out


class ToTensor(object):
    def __call__(self, sample):
        d_img = sample['d_img']
        score = np.array(sample['score'])
        center = sample['center']
        source_emb = sample['source_emb']

        d_img = np.transpose(d_img, (2, 0, 1)).astype('float32')
        d_img = torch.from_numpy(d_img)

        # center and source_emb to tensor
        center_arr = np.array(center).astype('float32')
        center_t = torch.from_numpy(center_arr).float()

        source_emb_arr = np.array(source_emb).astype('float32')
        source_emb_t = torch.from_numpy(source_emb_arr).float()

        score_t = torch.from_numpy(np.array(score).astype('float32'))

        out = {
            'd_img': d_img,
            'center': center_t,
            'source_emb': source_emb_t,
            'score': score_t
        }

        if 'w_img' in sample:
            w_img = sample['w_img']
            # If already CHW (we allowed that in previous version), handle both
            if w_img.ndim == 3 and w_img.shape[0] == 3:
                w_img_chw = w_img.astype('float32')
            else:
                w_img_chw = np.transpose(w_img, (2, 0, 1)).astype('float32')
            w_img_t = torch.from_numpy(w_img_chw)
            out['w_img'] = w_img_t

        return out


def RandShuffle(config):
    import numpy as _np, re as _re
    if config.split_avail:
        seed = _np.random.random()
        random_seed = int(seed*10)
        _np.random.seed(random_seed)

        with open(config.train_list, 'r') as the_file:
            train_list=the_file.read().split('\n')

        with open(config.test_list, 'r') as the_file:
            test_list=the_file.read().split('\n')

        with open(config.txt_file_tans, 'r') as the_file:
            data_list=the_file.read().split('\n')

        _np.random.shuffle(train_list)
        _np.random.shuffle(test_list)

        train_scenes=[img for img in data_list if _re.sub(r'-\d+'+'.pt', '', img) in train_list[:int(0.8*len(train_list))]]
        val_scenes=[img for img in data_list if _re.sub(r'-\d+'+'.pt', '', img) in train_list[int(0.8*len(train_list)):]]
        test_scenes=[img for img in data_list if _re.sub(r'-\d+'+'.pt', '', img) in test_list]

    else:
        with open(config.txt_file_tans, 'r') as the_file:
            data_list=the_file.read().split('\n')

        _np.random.shuffle(data_list)
        train_scenes=[img for img in data_list[:int(0.6*len(data_list))] ]
        val_scenes=[img for img in data_list[int(0.6*len(data_list)):int(0.8*len(data_list))] ]
        test_scenes=[img for img in data_list[int(0.8*len(data_list)):] ]

    return train_scenes, val_scenes, test_scenes
