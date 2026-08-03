import copy
import sys

import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from lib.MDI_Net import MDI_Net
from utils.dataloader import get_loader
from utils.utils import clip_gradient, AvgMeter
import torch.nn.functional as F
import numpy as np
import logging
from tqdm import tqdm
from sklearn.metrics import confusion_matrix


import random


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def train(train_loader, model, optimizer, epoch, opt):
    model.train()
    loss_list = []
    size_rates = [0.75, 1, 1.25]
    loss_P2_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear')
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear')
            P1 = model(images)
            loss_P1 = structure_loss(P1, gts)
            loss = loss_P1
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            if rate == 1:
                loss_P2_record.update(loss_P1.data, opt.batchsize)
                loss_list.append(loss_P2_record.show())
        if i % 20 == 0 or i == total_step:
            print(f'{datetime.now()} Epoch [{epoch:03d}/{opt.epoch:03d}], Step [{i:04d}/{total_step:04d}], lateral-5: {loss_P2_record.show():.4f}')
            logging.info(f'{datetime.now()} Epoch [{epoch:03d}/{opt.epoch:03d}], Step [{i:04d}/{total_step:04d}], lateral-5: {loss_P2_record.show():.4f}')
    mean_loss = np.mean([l.cpu().numpy() for l in loss_list])
    print(f'{datetime.now()} Epoch [{epoch:03d}/{opt.epoch:03d}] Train_mean_loss: {mean_loss:.4f}')
    logging.info(f'{datetime.now()} Epoch [{epoch:03d}/{opt.epoch:03d}] Train_mean_loss: {mean_loss:.4f}')


def test(test_loader, model, optimizer, epoch):
    model.eval()
    loss_list = []
    loss_P2_record = AvgMeter()
    gts_list, P1_list = [], []
    with torch.no_grad():
        for i, pack in enumerate(tqdm(test_loader), start=1):
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            P1 = model(images)
            loss_P1 = structure_loss(P1, gts)
            loss_P2_record.update(loss_P1.data, images.size(0))
            loss_list.append(loss_P2_record.show())
            gts_list.append(gts.squeeze(1).cpu().detach().numpy())
            if type(P1) is tuple:
                P1 = P1[0]
            pred_prob = torch.sigmoid(P1)
            pred_prob = pred_prob.squeeze(1).cpu().detach().numpy()
            P1_list.append(pred_prob)
    mean_loss = np.mean([l.cpu().numpy() for l in loss_list])
    preds = np.array(P1_list).reshape(-1)
    gts = np.array(gts_list).reshape(-1)
    y_pre = np.where(preds >= 0.5, 1, 0)
    y_true = np.where(gts >= 0.5, 1, 0)
    confusion = confusion_matrix(y_true, y_pre)
    TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]
    accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
    f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
    miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
    log_info = f'{datetime.now()}  val epoch: {epoch}, loss: {mean_loss:.4f}, miou: {miou:.4f}, f1_or_dsc: {f1_or_dsc:.4f}, accuracy: {accuracy:.4f}, specificity: {specificity:.4f}, sensitivity: {sensitivity:.4f}, confusion_matrix: {confusion}'
    print(log_info)
    logging.info(log_info)
    return f1_or_dsc



best_dice = 0.0
best_epoch = 0
best_model_state = None
counter = 0


def save_best_model(model_state_dict, epoch, dice, save_dir):
    if model_state_dict is not None:
        save_path = os.path.join(save_dir, "best.pth")
        torch.save(copy.deepcopy(model_state_dict), save_path)
        print(f"New best model saved! Epoch {epoch}, Dice={dice:.4f} -> {save_path}")
        logging.info(f"New best model saved! Epoch {epoch}, Dice={dice:.4f} -> {save_path}")
    else:
        print("Warning: model state is empty, skip saving best.pth")


def train_and_evaluate(model, scheduler, optimizer, train_loader, test_loader, epochs, seed, opt):
    global best_dice, best_epoch, best_model_state, counter

    best_dice = 0.0
    best_epoch = 0
    best_model_state = None
    counter = 0


    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    save_dir = opt.train_save
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch} started at {datetime.now()}")
        train(train_loader, model, optimizer, epoch, opt)
        dice = test(test_loader, model, optimizer, epoch)
        scheduler.step()

        if dice > best_dice:
            best_dice = dice
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            save_best_model(best_model_state, epoch, dice, save_dir)
            counter = 0  
        else:
            counter += 1
            log_info = f'val_dice not improved, patience count: {counter}'
            print(log_info)
            logging.info(log_info)

        if counter > 100:
            logging.info('Early stopping triggered, exit training loop')
            break
 
        if epoch % 10 == 0:
            log_info = f"Current global best dice: {best_dice:.4f}, achieved at epoch {best_epoch}"
            print(log_info)
            logging.info(log_info)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Training finished, loaded best checkpoint (epoch {best_epoch}, dice={best_dice:.4f})")
    else:
        print("Warning: no valid best checkpoint saved during training")
    return best_dice


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=1000, help='epoch number')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='AdamW or SGD')
    parser.add_argument('--augmentation', default=True, help='use augmentation')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=224, help='training size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clip')
    parser.add_argument('--train_path', type=str, default='/home/ta/datasets/Kvasir-SEG/Train_Folder')
    parser.add_argument('--test_path', type=str, default='/home/ta/datasets/Kvasir-SEG/Val_Folder')
    parser.add_argument('--train_save', type=str, default='/home/ta/Project/SwinPA-Sparse/result/Kvasir/MDI_Net/')
    opt = parser.parse_args()

    logging.basicConfig(filename='/home/ta/Project/SwinPA-Sparse/log/Kvasir/MDI_Net.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info('opt:{}'.format(opt))

    torch.cuda.set_device(opt.gpu_id)
    FIXED_SEED = 42

    model = MDI_Net().cuda()
    params = model.parameters()
    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    image_root = '{}/img/'.format(opt.train_path)
    gt_root = '{}/labelcol/'.format(opt.train_path)
    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize, augmentation=opt.augmentation)
    global total_step
    total_step = len(train_loader)

    image_test = '{}/img/'.format(opt.test_path)
    gt_test = '{}/labelcol/'.format(opt.test_path)
    test_loader = get_loader(image_test, trainsize=opt.trainsize, batchsize=1, augmentation=False)

    print("#" * 20, "Start Training (Single Run, save only best.pth)", "#" * 20)
    best_dice = train_and_evaluate(model, scheduler, optimizer, train_loader, test_loader, opt.epoch, FIXED_SEED, opt)
