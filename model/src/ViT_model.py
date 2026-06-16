import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import repeat
from torchvision import models
import os
import matplotlib.pyplot as plt
from ldacm import LDACM

def get_attn_pad_mask(seq_q, seq_k, i_pad):
    b, len_q = seq_q.size()
    _, len_k = seq_k.size()
    pad_mask = seq_k.eq(i_pad).unsqueeze(1).expand(b, len_q, len_k)
    return pad_mask


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.W_Q = nn.Linear(config.d_hidn, config.n_head * config.d_head)
        self.W_K = nn.Linear(config.d_hidn, config.n_head * config.d_head)
        self.W_V = nn.Linear(config.d_hidn, config.n_head * config.d_head)
        self.fc = nn.Linear(config.n_head * config.d_head, config.d_hidn)
        self.dropout = nn.Dropout(config.dropout)
        self.scale = 1 / (config.d_head ** 0.5)

    def forward(self, Q, K, V, attn_mask):
        b = Q.size(0)
        q = self.W_Q(Q).view(b, -1, self.n_head, self.d_head).transpose(1, 2)
        k = self.W_K(K).view(b, -1, self.n_head, self.d_head).transpose(1, 2)
        v = self.W_V(V).view(b, -1, self.n_head, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        scores.masked_fill_(attn_mask.unsqueeze(1), -1e9)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)

        context = context.transpose(1, 2).contiguous().view(b, -1, self.n_head * self.d_head)
        out = self.fc(context)
        return self.dropout(out), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.d_hidn, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_hidn)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = MultiHeadAttention(config)
        self.norm1 = nn.LayerNorm(config.d_hidn)
        self.ffn = PoswiseFeedForwardNet(config)
        self.norm2 = nn.LayerNorm(config.d_hidn)

    def forward(self, x, attn_mask):
        attn_out, _ = self.self_attn(x, x, x, attn_mask)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x, None


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_embedding = nn.Parameter(
            torch.randn(1, config.d_hidn, config.n_enc_seq[0], config.n_enc_seq[1])
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_hidn))
        self.dropout = nn.Dropout(config.emb_dropout)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.n_layer)])

    def forward(self, mask_inputs, feat_dis_img_embed, center, source_emb):
        b, c, h, w = feat_dis_img_embed.size()
        feat = feat_dis_img_embed + self.pos_embedding
        feat = feat.view(b, c, -1).permute(0, 2, 1)
        cls = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls, feat), dim=1)
        x = self.dropout(x)

        attn_mask = get_attn_pad_mask(mask_inputs, mask_inputs, self.config.i_pad)
        for layer in self.layers:
            x, _ = layer(x, attn_mask)
        return x, None


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)

    def forward(self, mask_inputs, feat_dis_img_embed, center, source_emb):
        enc_outputs, _ = self.encoder(mask_inputs, feat_dis_img_embed, center, source_emb)
        return enc_outputs


# --------------- SCNN ---------------
class SCNN(nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, 3, 1, 1), nn.BatchNorm2d(48), nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, 2, 1), nn.BatchNorm2d(48), nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )

    def forward(self, x, save_heatmap=False, img_names=None, epoch=None, batch_idx=None):
        features = self.features(x)

        if save_heatmap and img_names is not None:
            self.generate_scnn_heatmap(features, img_names, epoch, batch_idx)

        return features

    def generate_scnn_heatmap(self, feature_maps, img_names, epoch=None, batch_idx=None):

        save_dir = "./heatmaps/scnn"
        if epoch is not None:
            save_dir = f"./heatmaps/scnn/epoch_{epoch}"
        if batch_idx is not None:
            save_dir = f"{save_dir}/batch_{batch_idx}"
        os.makedirs(save_dir, exist_ok=True)

        self.generate_heatmap(feature_maps, img_names, save_dir, "SCNN Features")

    def generate_heatmap(self, feature_map_batch, img_names=None, save_dir="./heatmaps", title_prefix=""):

        os.makedirs(save_dir, exist_ok=True)

        if img_names is None:
            img_names = [f'sample_{i}' for i in range(feature_map_batch.shape[0])]

        for i in range(feature_map_batch.shape[0]):
            feature_map = feature_map_batch[i]

            upsampled_map = torch.nn.functional.interpolate(
                feature_map.unsqueeze(0),
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

            feature_map_np = upsampled_map.cpu().detach().numpy()

            if len(feature_map_np.shape) == 3:
                combined_heatmap = np.zeros_like(feature_map_np[0])
                for j in range(feature_map_np.shape[0]):
                    channel_map = feature_map_np[j]
                    channel_map = (channel_map - np.min(channel_map)) / (
                            np.max(channel_map) - np.min(channel_map) + 1e-8)
                    combined_heatmap += channel_map

                combined_heatmap = (combined_heatmap - np.min(combined_heatmap)) / (
                        np.max(combined_heatmap) - np.min(combined_heatmap) + 1e-8)
            else:
                combined_heatmap = feature_map_np

            img_name = os.path.splitext(os.path.basename(img_names[i]))[0]
            save_path = os.path.join(save_dir, f"{img_name}_heatmap.png")

            plt.figure(figsize=(10, 8))
            plt.imshow(combined_heatmap, cmap='jet', interpolation='nearest')
            plt.colorbar()
            plt.title(f"{title_prefix} for: {img_name}")
            plt.axis('off')
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
            plt.close()


class DBCNN(nn.Module):
    def __init__(self, options=None):
        super(DBCNN, self).__init__()

        self.ldacm = LDACM(in_channels=3)

        vgg = models.vgg16(pretrained=False)
        self.features1 = nn.Sequential(*list(vgg.features.children())[:-1])
        self.features2 = SCNN().features

        self.fc = nn.Linear(512 * 128, 1)

    def forward(self, X, label=None, requires_loss=False,
                save_heatmap=False, img_names=None, epoch=None, batch_idx=None):

        X_corrected = self.ldacm(X, save_heatmap, img_names, epoch, batch_idx)

        N = X_corrected.size(0)
        X1 = self.features1(X_corrected)
        X2 = self.features2(X_corrected)

        if X1.size()[2:] != X2.size()[2:]:
            X2 = F.interpolate(X2, size=X1.size()[2:], mode='bilinear', align_corners=False)

        if save_heatmap and img_names is not None:
            self.generate_global_heatmap(X1, X2, img_names, epoch, batch_idx)

        X1 = X1.view(N, 512, -1)
        X2 = X2.view(N, 128, -1)

        Xb = torch.bmm(X1, X2.transpose(1, 2)) / X1.size(-1)
        Xb = Xb.view(N, -1)
        Xb = torch.sqrt(Xb + 1e-8)
        Xb = F.normalize(Xb)

        score = self.fc(Xb)

        if requires_loss:
            loss = F.mse_loss(score, label)
            return score, label, loss
        else:
            return score

    def generate_global_heatmap(self, vgg_features, scnn_features, img_names, epoch=None, batch_idx=None):

        save_dir = "./heatmaps/global"
        if epoch is not None:
            save_dir = f"./heatmaps/global/epoch_{epoch}"
        if batch_idx is not None:
            save_dir = f"{save_dir}/batch_{batch_idx}"
        os.makedirs(save_dir, exist_ok=True)

        self.generate_heatmap(vgg_features, img_names,
                              f"{save_dir}/vgg", "VGG Features")

        self.generate_heatmap(scnn_features, img_names,
                              f"{save_dir}/scnn", "SCNN Features")

    def generate_heatmap(self, feature_map_batch, img_names=None, save_dir="./heatmaps", title_prefix=""):

        os.makedirs(save_dir, exist_ok=True)

        if img_names is None:
            img_names = [f'sample_{i}' for i in range(feature_map_batch.shape[0])]

        for i in range(feature_map_batch.shape[0]):
            feature_map = feature_map_batch[i]


            upsampled_map = torch.nn.functional.interpolate(
                feature_map.unsqueeze(0),
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

            feature_map_np = upsampled_map.cpu().detach().numpy()

            if len(feature_map_np.shape) == 3:
                combined_heatmap = np.zeros_like(feature_map_np[0])
                for j in range(feature_map_np.shape[0]):
                    channel_map = feature_map_np[j]
                    channel_map = (channel_map - np.min(channel_map)) / (
                            np.max(channel_map) - np.min(channel_map) + 1e-8)
                    combined_heatmap += channel_map

                combined_heatmap = (combined_heatmap - np.min(combined_heatmap)) / (
                        np.max(combined_heatmap) - np.min(combined_heatmap) + 1e-8)
            else:
                combined_heatmap = feature_map_np


            img_name = os.path.splitext(os.path.basename(img_names[i]))[0]
            save_path = os.path.join(save_dir, f"{img_name}_heatmap.png")


            plt.figure(figsize=(10, 8))
            plt.imshow(combined_heatmap, cmap='jet', interpolation='nearest')
            plt.colorbar()
            plt.title(f"{title_prefix} for: {img_name}")
            plt.axis('off')
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
            plt.close()


class IQARegression(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv_enc = nn.Conv2d(2048, config.d_hidn, kernel_size=1)
        self.transformer = Transformer(self.config)

        self.projection = nn.Sequential(
            nn.Linear(config.d_hidn, config.d_MLP_head, bias=False),
            nn.GELU(),
            nn.Linear(config.d_MLP_head, config.n_output, bias=False)
        )

        self.global_branch = DBCNN()

        self.fuse_fc = nn.Linear(2, 1)

    def forward(self, mask_inputs, feat_dis_img, center, source_emb,
                w_img=None, img_names=None, save_heatmap=False, epoch=None, batch_idx=None):
        feat_dis_img_embed = self.conv_enc(feat_dis_img)

        if save_heatmap and img_names is not None:
            self.generate_local_heatmap(feat_dis_img_embed, img_names, epoch, batch_idx)

        enc_outputs = self.transformer(mask_inputs, feat_dis_img_embed, center, source_emb)
        enc_outputs = enc_outputs[:, 0, :]

        pred_local = self.projection(enc_outputs)

        if (self.global_branch is not None) and (w_img is not None):
            global_pred = self.global_branch(w_img, save_heatmap=save_heatmap,
                                             img_names=img_names, epoch=epoch, batch_idx=batch_idx)
        else:
            global_pred = torch.zeros_like(pred_local)

        fused = self.fuse_fc(torch.cat((pred_local, global_pred), dim=1))
        return fused
