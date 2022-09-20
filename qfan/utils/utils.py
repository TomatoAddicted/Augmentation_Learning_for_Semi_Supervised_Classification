import shutil
import os
import time
import multiprocessing
from typing import Union, List, Tuple

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torchnet import meter
from tqdm import tqdm
import yaml
from fvcore.nn.precise_bn import update_bn_stats
from PIL import Image
import numpy as np

def setup_settings_in_yaml(args):
    if args.config:
        with open(args.config) as f:
            settings = yaml.load(f, Loader=yaml.CLoader)
        for k, v in settings.items():
            if getattr(args, k, None) is None:
                setattr(args, k, v)

    return args


def safe_load_image(img_path):
    img = None
    num_try = 0
    while num_try < 10:
        try:
            #img_tmp = Image.open(img_path).convert('RGB')
            img_tmp = Image.fromarray(cv2.imread(img_path))
            #print(np.array(img_tmp).shape)
            img = img_tmp.copy()
            img_tmp.close()
            break
        except Exception as e:
            print('[Will try load again] error loading image: {}, '
                  'error: {}'.format(img_path, str(e)))
            num_try += 1
    if img is None:
        raise ValueError('[Fail 10 times] error loading image: {}'.format(img_path))
    return img

def get_class_distribution(dataset_obj):
    idx2class = {v: k for k, v in dataset_obj.class_to_idx.items()}
    count_dict = {k:0 for k,v in dataset_obj.class_to_idx.items()}
    
    for element in dataset_obj:
        y_lbl = element[1]
        y_lbl = idx2class[y_lbl]
        count_dict[y_lbl] += 1
            
    return count_dict

def compute_acc_map(logits: torch.Tensor, test_y: torch.Tensor,
                    topk: Union[int, List[int]] = None, have_softmaxed: bool = False) -> Tuple[List[float], float]:
    """

    Args:
        logits: NxK tensor, N is batch size, K is number of classes
        test_y: N tensor, each with its own label index or NxK tensor if multilabel
        topk: top-k for accuracy,
        have_softmaxed: whether or logits is softmaxed

    Returns:

    """
    num_classes = logits.shape[1]
    topk = [1, min(5, num_classes)] if topk is None else topk
    single_label = True if len(test_y.shape) == 1 else False
    probs = F.softmax(logits, dim=1) if not have_softmaxed else logits
    if single_label:
        acc_meter = meter.ClassErrorMeter(topk=topk, accuracy=True)
        acc_meter.add(logits, test_y)
        acc = acc_meter.value()
        gt = torch.zeros_like(logits)
        gt[torch.LongTensor(range(gt.size(0))), test_y.view(-1)] = 1
    else:
        gt = test_y
        acc = [0] * len(topk)
    map_meter = meter.mAPMeter()
    map_meter.add(probs, gt)
    ap = map_meter.value() * 100.0
    return acc, ap.item()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_mean_and_std(model: nn.Module, args) -> ([float], [float]):
    try:
        mean = model.mean(args.modality)
        std = model.std(args.modality)
    except Exception as e:
        mean, std = args.mean, args.std
    return mean, std


def accuracy(output, target, topk=(1, 2)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, filepath=''):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))



def get_augmentor(is_train: bool, image_size: int, mean: List[float] = None,
                  std: List[float] = None,):

    mean = [0.485, 0.456, 0.406] if mean is None else mean
    std = [0.229, 0.224, 0.225] if std is None else std

    normalize = transforms.Normalize(mean=mean, std=std)

    if is_train:
        augments = [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    else:
        augments = [
            transforms.Resize(int(image_size * (256 / 224))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    augmentor = transforms.Compose(augments)
    return augmentor

'''
def build_dataflow(dataset: torch.utils.data.Dataset, is_train: bool, batch_size: int, workers=36, is_distributed=False):
    workers = min(workers, multiprocessing.cpu_count())
    shuffle = False
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_train) if is_distributed else None
    if is_train:
        shuffle = sampler is None

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=workers, pin_memory=True, sampler=sampler)

    return data_loader
'''

def build_dataflow(dataset: torch.utils.data.Dataset, is_train: bool, batch_size: int, workers=36, is_distributed=False, sampler=None):
    workers = min(workers, multiprocessing.cpu_count())
    shuffle = False
    the_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_train) if is_distributed else None
    
    if sampler is not None:
        the_sampler = sampler

    if is_train:
        shuffle = the_sampler is None

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=workers, pin_memory=True, sampler=the_sampler)

    return data_loader



def calculate_and_update_precise_bn(loader: torch.utils.data.DataLoader, model: nn.Module):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
    """

    def _gen_loader():
        for inputs, _ in loader:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), min(len(loader), 200))


def cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax(dim=1)
    result = torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))
    return result


def train(data_loader, model, criterion, optimizer, epoch, display=100,
          steps_per_epoch=99999999999, label_smoothing=0.0, num_classes=None,
          clip_gradient=None, gpu_id=None, rank=0, eval_criterion=accuracy,
          precise_bn=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # set different random see every epoch
    if dist.is_initialized():
        data_loader.sampler.set_epoch(epoch)

    # switch to train mode
    model.train()
    end = time.time()
    num_batch = 0
    disable_status_bar = False if rank == 0 else True
    with tqdm(total=len(data_loader), disable=disable_status_bar) as t_bar:
        for i, (images, target) in enumerate(data_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            # compute output
            if gpu_id is not None:
                images = images.cuda(gpu_id, non_blocking=True)

            output = model(images)
            #TODO check label_smoothing
            if label_smoothing > 0.0:
                smoothed_target = torch.zeros([images.size(0), num_classes]).scatter_(
                    1, target.unsqueeze(1), 1.0) * (1.0 - label_smoothing) + 1 / float(num_classes) * label_smoothing
                smoothed_target = smoothed_target.type(torch.float)
                smoothed_target = smoothed_target.cuda(gpu_id, non_blocking=True)
                # target = target.cuda(non_blocking=True)
                loss = cross_entropy(output, smoothed_target)
            else:
                target = target.cuda(gpu_id, non_blocking=True)
                # target = target.cuda(non_blocking=True)
                loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = eval_criterion(output, target)

            if dist.is_initialized():
                world_size = dist.get_world_size()
                dist.all_reduce(prec1)
                dist.all_reduce(prec5)
                prec1 /= world_size
                prec5 /= world_size

            losses.update(loss.item(), images.size(0))
            top1.update(prec1[0], images.size(0))
            top5.update(prec5[0], images.size(0))
            # compute gradient and do SGD step
            loss.backward()

            if clip_gradient is not None:
                _ = clip_grad_norm_(model.parameters(), clip_gradient)

            optimizer.step()
            optimizer.zero_grad()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % display == 0 and rank == 0 and False:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, len(data_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1, top5=top5), flush=True)
            num_batch += 1
            t_bar.set_description(f'Epoch [{epoch:03d}] [Train] Loss: {losses.avg:.3f} Top@1: {top1.avg:.2f} Top@5: {top5.avg:.2f}')
            t_bar.update(1)
            if i > steps_per_epoch:
                break
    if precise_bn:
        calculate_and_update_precise_bn(data_loader, model)

    return top1.avg, top5.avg, losses.avg, batch_time.avg, data_time.avg, num_batch


def validate(data_loader, model, criterion, gpu_id=None, eval_criterion=accuracy, rank=0):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    disable_status_bar = False if rank == 0 else True
    with torch.no_grad(), tqdm(total=len(data_loader), disable=disable_status_bar) as t_bar:
        end = time.time()
        for i, (images, target) in enumerate(data_loader):

            if gpu_id is not None:
                images = images.cuda(gpu_id, non_blocking=True)
            target = target.cuda(gpu_id, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)
            # measure accuracy and record loss
            prec1, prec5 = eval_criterion(output, target)
            if dist.is_initialized():
                world_size = dist.get_world_size()
                dist.all_reduce(prec1)
                dist.all_reduce(prec5)
                prec1 /= world_size
                prec5 /= world_size
            losses.update(loss.item(), images.size(0))
            top1.update(prec1[0], images.size(0))
            top5.update(prec5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            t_bar.set_description(f'              [Val] Loss: {losses.avg:.3f} Top@1: {top1.avg:.2f} Top@5: {top5.avg:.2f}')
            t_bar.update(1)

    return top1.avg, top5.avg, losses.avg, batch_time.avg

def evaluate(data_loader, model, criterion, gpu_id=None, eval_criterion=accuracy, rank=0):
    #batch_time = AverageMeter()

    correct = 0
    total = 0
    confusion = meter.ConfusionMeter(3, normalized=True)

    # switch to evaluate mode
    model.eval()

    total_outputs = 0
    dataset_size = len(data_loader.dataset.imgs)
    target_list = set(data_loader.dataset.targets)
    num_classes = max(target_list) + 1
    outputs = np.zeros((dataset_size, num_classes))
    disable_status_bar = False if rank == 0 else True
    with torch.no_grad(), tqdm(total=len(data_loader), disable=disable_status_bar) as t_bar:
        end = time.time()
        for i, (images, target) in enumerate(data_loader):

            if gpu_id is not None:
                images = images.cuda(gpu_id, non_blocking=True)
            target = target.cuda(gpu_id, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)


            # measure accuracy and record loss
            _, pred = torch.max(output, 1)
            confusion.add(pred.data, target.data)
            total += target.size(0)
            correct += (pred == target).sum().item()
            output = F.softmax(output)
            output = output.data.cpu().numpy().copy()
            batch_size = output.shape[0]
            outputs[total_outputs:total_outputs + batch_size, :] = output
            total_outputs += batch_size
            # measure elapsed time
    #        batch_time.update(time.time() - end)
            end = time.time()
            #t_bar.set_description(f'              [Val] Loss: {losses.avg:.3f} Top@1: {top1.avg:.2f} Top@5: {top5.avg:.2f}')
            #t_bar.update(1)
    acc = correct / total
    print (confusion.value())
    
    for img_info, pred in zip(data_loader.dataset.imgs, outputs):
        pred_id = np.argmax(pred)
        if pred_id != int(img_info[1]):
            print (f"%s %d %4.2f %4.2f %4.2f" % (img_info[0].split('/', 4)[-1], int(img_info[1]), pred[0], pred[1], pred[2]))
       
    return acc, 0, 0, 0 


