import os
import torch
import numpy as np
import cv2
from patchify import patchify, unpatchify
from random import randint


class IQADataset(torch.utils.data.Dataset):
    def __init__(self, config, transform, scene_list):
        super(IQADataset, self).__init__()
        self.config = config
        self.transform = transform
        self.scene_list = scene_list
        self.n_enc_seq = config.n_enc_seq
        self.patch_size = config.patch_size
        self.tan_data_path = config.tan_data_path
        self.global_img_size = getattr(config, 'GLOBAL_IMG_SIZE', 224)

        dataset_flag = os.environ.get('CURRENT_DATASET', 'cviqd').lower()
        if dataset_flag.startswith('oiqa'):
            self.global_img_path = config.oiqa_global_path
        else:
            self.global_img_path = config.cviq_global_path

        self.data_dict = IQADatalist(scene_list=self.scene_list).load_data_dict()
        self.n_images = len(self.data_dict['d_img_list'])

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        d_img_name = self.data_dict['d_img_list'][idx]
        data = torch.load(os.path.join(self.tan_data_path, d_img_name))

        center = data['center']
        source_emb = data['source_emb']
        score_raw = data['score']
        try:
            score = float(score_raw.strip('[]'))
        except:
            try:
                score = float(score_raw)
            except:
                score = float(np.asarray(score_raw).astype(float))

        # local/tangent image
        if 'd_img_org' in data:
            d_img = data['d_img_org']
        elif 'd_img' in data:
            d_img = data['d_img']
        else:
            raise RuntimeError(f"No local image found in {d_img_name}")

        # resize local to Transformer grid (H,W)
        H, W = self.n_enc_seq[0] * self.patch_size, self.n_enc_seq[1] * self.patch_size
        try:
            d_img = cv2.resize(d_img, (H, W))
        except:
            d_img = np.array(d_img).astype('float32')
            d_img = np.resize(d_img, (H, W, 3))

        try:
            d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        except:
            pass
        d_img = np.array(d_img).astype('float32') / 255.0

        sample = {
            'd_img': d_img,
            'center': center,
            'source_emb': source_emb,
            'score': score,
            'img_name': d_img_name
        }

        base_name = os.path.basename(d_img_name).split('-')[0]
        possible_ext = [".png", ".jpg", ".jpeg", ".bmp"]
        w_img = None
        for ext in possible_ext:
            candidate = os.path.join(self.global_img_path, base_name + ext)
            if os.path.exists(candidate):
                w_img = cv2.imread(candidate)
                break

        if w_img is None:
            # try prefix without extension (in case names differ)
            # search directory for files that start with base_name
            try:
                for f in os.listdir(self.global_img_path):
                    if f.startswith(base_name):
                        w_img = cv2.imread(os.path.join(self.global_img_path, f))
                        break
            except Exception:
                w_img = None

        if w_img is None:
            # fallback: use d_img (resized) to avoid crash
            w_img = (d_img * 255.0).astype('uint8')
            try:
                w_img = cv2.cvtColor(w_img, cv2.COLOR_RGB2BGR)
            except:
                pass
        # ensure RGB
        try:
            w_img = cv2.cvtColor(w_img, cv2.COLOR_BGR2RGB)
        except:
            pass

        # resize to global_img_size
        try:
            w_img_resized = cv2.resize(w_img, (self.global_img_size, self.global_img_size))
        except:
            # fallback naive
            w_img_resized = np.array(w_img).astype('float32')
            h0, w0 = w_img_resized.shape[:2]
            new_img = np.zeros((self.global_img_size, self.global_img_size, 3), dtype='float32')
            new_img[:min(h0, self.global_img_size), :min(w0, self.global_img_size), :] = w_img_resized[
                                                                                         :min(h0, self.global_img_size),
                                                                                         :min(w0, self.global_img_size),
                                                                                         :]
            w_img_resized = new_img

        w_img_resized = np.array(w_img_resized).astype('float32') / 255.0

        sample['w_img'] = w_img_resized

        if self.transform:
            sample = self.transform(sample)

        return sample


class IQADatalist():
    def __init__(self, scene_list):
        self.scene_list = scene_list

    def load_data_dict(self):
        d_img_list = [img for img in self.scene_list]
        return {'d_img_list': d_img_list}
