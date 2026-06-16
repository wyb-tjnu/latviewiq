import os
import torch
from tqdm import tqdm
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error


def train_epoch(
        config,
        epoch,
        model_transformer,
        model_backbone,
        criterion,
        optimizer,
        scheduler,
        train_loader
):
    losses = []
    model_transformer.train()
    model_backbone.train()

    pred_epoch = []
    labels_epoch = []

    for batch_idx, data in enumerate(tqdm(train_loader)):
        batch_size = data['d_img'].size(0)
        mask_inputs = torch.ones(batch_size, (config.n_enc_seq[0] * config.n_enc_seq[1]) + 1).to(config.device)

        d_img = data['d_img'].to(config.device)
        w_img = data.get('w_img', None)
        if w_img is not None:
            w_img = w_img.to(config.device)

        source_emb = data['source_emb']
        center = data['center']
        labels = data['score']
        labels = torch.squeeze(labels.type(torch.FloatTensor)).to(config.device)

        img_names = data.get('img_name', [f'batch_{batch_idx}_sample_{i}' for i in range(batch_size)])

        if isinstance(center, torch.Tensor):
            center = center.to(config.device)
        else:
            center = torch.tensor(center).to(config.device)

        if isinstance(source_emb, torch.Tensor):
            source_emb = source_emb.to(config.device)
        else:
            source_emb = torch.tensor(source_emb).to(config.device)

        feat_d_img = model_backbone(d_img)

        optimizer.zero_grad()

        pred = model_transformer(
            mask_inputs,
            feat_d_img,
            center,
            source_emb,
            w_img,
            img_names,
            epoch,
            batch_idx
        )

        loss = criterion(torch.squeeze(pred), labels)
        loss_val = loss.item()
        losses.append(loss_val)

        loss.backward()
        optimizer.step()
        scheduler.step()

        pred_batch_numpy = pred.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)

    try:
        srcc, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    except:
        srcc = 0
    try:
        plcc, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    except:
        plcc = 0
    rmse = mean_squared_error(np.squeeze(labels_epoch), np.squeeze(pred_epoch), squared=False)

    print('[train] epoch:%d / loss:%f / PLCC:%4f / SROCC:%4f / RMSE:%4f' % \
          (epoch + 1, loss.item(), srcc, plcc, rmse))

    if (epoch + 1) % config.save_freq == 0:
        weights_file_name = "epoch%d.pth" % (epoch + 1)
        weights_file = os.path.join(config.snap_path, weights_file_name)
        torch.save({
            'epoch': epoch,
            'model_backbone_state_dict': model_backbone.state_dict(),
            'model_transformer_state_dict': model_transformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss
        }, weights_file)
        print('save weights of epoch %d' % (epoch + 1))

    return np.mean(losses), srcc, plcc, rmse


def eval_epoch(
        config,
        epoch,
        model_transformer,
        model_backbone,
        criterion,
        test_loader
):
    with torch.no_grad():
        losses = []
        model_transformer.eval()
        model_backbone.eval()

        pred_epoch = []
        labels_epoch = []

        for batch_idx, data in enumerate(tqdm(test_loader)):
            batch_size = data['d_img'].size(0)
            mask_inputs = torch.ones(batch_size, (config.n_enc_seq[0] * config.n_enc_seq[1]) + 1).to(config.device)

            d_img = data['d_img'].to(config.device)
            w_img = data.get('w_img', None)
            if w_img is not None:
                w_img = w_img.to(config.device)

            center = data['center']
            source_emb = data['source_emb']
            labels = data['score']
            labels = torch.squeeze(labels.type(torch.FloatTensor)).to(config.device)

            # 获取图像名称
            img_names = data.get('img_name', [f'val_batch_{batch_idx}_sample_{i}' for i in range(batch_size)])

            if isinstance(center, torch.Tensor):
                center = center.to(config.device)
            else:
                center = torch.tensor(center).to(config.device)

            if isinstance(source_emb, torch.Tensor):
                source_emb = source_emb.to(config.device)
            else:
                source_emb = torch.tensor(source_emb).to(config.device)

            pred = model_transformer(mask_inputs, feat_d_img, center, source_emb,
                                     w_img, img_names, save_heatmap, epoch, batch_idx)

            loss = criterion(torch.squeeze(pred), labels)
            loss_val = loss.item()
            losses.append(loss_val)

            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)

    try:
        srcc, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    except:
        srcc = 0
    try:
        plcc, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    except:
        plcc = 0
    rmse = mean_squared_error(np.squeeze(labels_epoch), np.squeeze(pred_epoch), squared=False)

    print('[validation] epoch:%d / loss:%f /SROCC:%4f / PLCC:%4f / RMSE:%4f' % \
          (epoch + 1, loss.item(), srcc, plcc, rmse))

    return np.mean(losses), srcc, plcc, rmse
