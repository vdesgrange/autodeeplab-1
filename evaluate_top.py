import os
import numpy as np
import os.path as osp
import heapq

import torch
import torch.nn as nn
import torch.backends.cudnn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from mypath import Path
from utils.utils import AverageMeter, inter_and_union
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from config_utils.evaluate_args import obtain_evaluate_args
from retrain_model.build_autodeeplab import Retrain_Autodeeplab
from dataloaders.datasets.cityscapes import CityscapesSegmentation


class Prediction(object):
    def __init__(self, input, output, target, inter, union):
        self.input = input
        self.output = output
        self.target = target
        self.inter = inter
        self.union = union

    def __lt__(self, other):
        iou = self.inter / (self.union + 1e-10)
        other_iou = other.inter / (other.union + 1e-10)
        return iou > other_iou


def best_results(hq, prediction, capacity=10):
    if len(hq) < capacity:
        heapq.heappush(hq, prediction)
    else:
        heapq.heappushpop(hq, prediction)

    return hq


def main(start_epoch, epochs):

    assert torch.cuda.is_available(), NotImplementedError('No cuda available ')

    if not osp.exists('data/'):
        os.mkdir('data/')

    if not osp.exists('log/'):
        os.mkdir('log/')

    args = obtain_evaluate_args()
    torch.backends.cudnn.benchmark = True
    model_fname = 'data/deeplab_{0}_{1}_v3_{2}_epoch%d.pth'.format(args.backbone, args.dataset, args.exp)

    if args.dataset == 'cityscapes':
        dataset = CityscapesSegmentation(args=args, root=Path.db_root_dir(args.dataset), split='reval')
    else:
        return NotImplementedError

    if args.backbone == 'autodeeplab':
        model = Retrain_Autodeeplab(args)
    else:
        raise ValueError('Unknown backbone: {}'.format(args.backbone))

    if not args.train:
        val_dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
        model = torch.nn.DataParallel(model).cuda()

        print("======================start evaluate=======================")

        for epoch in range(epochs):

            # Define Saver
            saver = Saver(args.directory)
            saver.save_experiment_config()
            # Define Tensorboard Summary
            summary = TensorboardSummary(saver.experiment_dir)
            writer = summary.create_summary()

            print("evaluate epoch {:}".format(epoch + start_epoch))
            checkpoint_name = model_fname % (epoch + start_epoch)
            print(checkpoint_name)
            checkpoint = torch.load(checkpoint_name)

            state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
            model.module.load_state_dict(state_dict)
            inter_meter = AverageMeter()
            union_meter = AverageMeter()
            best_heapq = []

            for i, sample in enumerate(val_dataloader):
                inputs, target = sample['image'], sample['label']
                N, H, W = target.shape
                total_outputs = torch.zeros((N, dataset.NUM_CLASSES, H, W)).cuda()

                with torch.no_grad():
                    for j, scale in enumerate(args.eval_scales):
                        new_scale = [int(H * scale), int(W * scale)]

                        inputs_tmp = F.upsample(inputs, new_scale, mode='bilinear', align_corners=True)
                        inputs_tmp = inputs_tmp.cuda()
                        outputs = model(inputs_tmp)
                        outputs_tmp = F.upsample(outputs, (H, W), mode='bilinear', align_corners=True)

                        total_outputs += outputs_tmp

                    _, pred = torch.max(total_outputs, 1)
                    pred = pred.detach().cpu().numpy().squeeze().astype(np.uint8)
                    mask = target.numpy().astype(np.uint8)
                    print('eval: {0}/{1}'.format(i + 1, len(val_dataloader)))

                    inter, union = inter_and_union(pred, mask, len(dataset.CLASSES))
                    inter_meter.update(inter)
                    union_meter.update(union)

                    prediction = Prediction(inputs, outputs, target, inter, union)
                    best_heapq = best_results(best_heapq, prediction)

            iou = inter_meter.sum / (union_meter.sum + 1e-10)
            miou = 'epoch: {0} Mean IoU: {1:.2f}'.format(epoch, iou.mean() * 100)

            # Log IoU results
            f = open('log/result.txt', 'a')
            for i, val in enumerate(iou):
                class_iou = 'IoU {0}: {1:.2f}\n'.format(dataset.CLASSES[i], val * 100)
                f.write(class_iou)

            # Log best results in TensorboardX
            while (best_heapq):
                curr = heapq.heappop(best_heapq)
                writer.add_scalar('val/mIoU', miou, epoch)
                writer.add_scalar('val/IoU', curr.inter / (curr.union + 1e-10), epoch)
                summary.visualize_image(writer, args.dataset, inputs, target, outputs, epoch)

            f.write('\n')
            f.write(miou)
            f.write('\n')
            f.close()


if __name__ == "__main__":
    epochs = range(0, 100, 1)
    state_epochs = 900
    main(epochs=epochs, start_epoch=state_epochs)
