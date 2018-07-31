import argparse
import os
import sys
import time
from datetime import datetime
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm

from ssn_dataset import SSNDataSet
from ssn_models_2 import SSN
from transforms import *
from ssn_opts import parser
from ops.ssn_ops import CompletenessLoss, ClassWiseRegressionLoss
from ops.utils import get_configs
from torch.utils import model_zoo
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
best_loss = 100

def main():
    global args, best_loss
    args = parser.parse_args()

    dataset_configs = get_configs(args.dataset)

    num_class = dataset_configs['num_class']
    stpp_configs = tuple(dataset_configs['stpp'])  #TODO
    sampling_configs = dataset_configs['sampling']

    base_model = 'p3d'
    model = SSN(num_class, args.num_aug_segments, args.num_body_segments, args.num_aug_segments,
                args.modality,
                base_model=base_model, dropout=args.dropout,
                stpp_cfg=stpp_configs, bn_mode=args.bn_mode)
    weights_file = 'ssn_activitynet1.2_BNInception_rgb_epoch-2_checkpoint.pth.tar'
    weights = torch.load(weights_file)['state_dict']
    weights = {'.'.join(k.split('.')[1:]): v for k, v in list(weights.items())}
    model.load_state_dict(weights)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation()

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    cudnn.benchmark = True
    pin_memory = (args.modality == 'RGB')

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5
    else:
        raise ValueError("unknown modality {}".format(args.modality))

    train_prop_file = 'data/{}_proposal_list.txt'.format(dataset_configs['train_list'])
    val_prop_file = 'data/{}_proposal_list.txt'.format(dataset_configs['test_list'])
    train_loader = torch.utils.data.DataLoader(
        SSNDataSet("", train_prop_file,
                   epoch_multiplier=args.training_epoch_multiplier,
                   new_length=data_length,
                   modality=args.modality, exclude_empty=True, **sampling_configs,
                   aug_seg=args.num_aug_segments, body_seg=args.num_body_segments,
                   image_tmpl="{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=pin_memory,
        drop_last=True)  # in training we drop the last incomplete minibatch

    val_loader = torch.utils.data.DataLoader(
        SSNDataSet("", val_prop_file,
                   new_length=data_length,
                   modality=args.modality, exclude_empty=True, **sampling_configs,
                   aug_seg=args.num_aug_segments, body_seg=args.num_body_segments,
                   image_tmpl="{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), reg_stats=train_loader.dataset.stats),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=pin_memory)

    activity_criterion = torch.nn.CrossEntropyLoss().cuda()
    completeness_criterion = CompletenessLoss().cuda()
    regression_criterion = ClassWiseRegressionLoss().cuda()

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        validate(val_loader, model, activity_criterion, completeness_criterion, regression_criterion, 0)
        return
 #   exit()

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)

        # train for one epoch
        train(train_loader, model, activity_criterion, completeness_criterion, regression_criterion, optimizer, epoch)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            loss = validate(val_loader, model, activity_criterion, completeness_criterion, regression_criterion, (epoch + 1) * len(train_loader))

            # remember best prec@1 and save checkpoint
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'reg_stats': torch.from_numpy(train_loader.dataset.stats)
            }, is_best)


def train(train_loader, model, act_criterion, comp_criterion, regression_criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    act_losses = AverageMeter()
    comp_losses = AverageMeter()
    reg_losses = AverageMeter()
    act_accuracies = AverageMeter()
    fg_accuracies = AverageMeter()
    bg_accuracies = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    optimizer.zero_grad()

    ohem_num = train_loader.dataset.fg_per_video
    comp_group_size = train_loader.dataset.fg_per_video + train_loader.dataset.incomplete_per_video
    for i, (out_frames, out_prop_len, out_prop_scaling, out_prop_type, out_prop_labels,
            out_prop_reg_targets, out_stage_split) \
            in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(out_frames)
        scaling_var = torch.autograd.Variable(out_prop_scaling)
        target_var = torch.autograd.Variable(out_prop_labels)
        reg_target_var = torch.autograd.Variable(out_prop_reg_targets)
        prop_type_var = torch.autograd.Variable(out_prop_type)

        # compute output

        #sys.stdout.write('forwarding model ...')
        #sys.stdout.flush()
        activity_out, activity_target, \
        completeness_out, completeness_target, \
        regression_out, regression_labels, regression_target = model(input_var, scaling_var, target_var,
                                                                     reg_target_var, prop_type_var)
        #sys.stdout.write('done\n')
        #sys.stdout.flush()

        act_loss = act_criterion(activity_out, activity_target)
        comp_loss = comp_criterion(completeness_out, completeness_target, ohem_num, comp_group_size)
        reg_loss = regression_criterion(regression_out, regression_labels, regression_target)

        loss = act_loss + comp_loss * args.comp_loss_weight + reg_loss * args.reg_loss_weight

        reg_losses.update(reg_loss.data[0], out_frames.size(0))

        # measure mAP and record loss
        losses.update(loss.data[0], out_frames.size(0))
        act_losses.update(act_loss.data[0], out_frames.size(0))
        comp_losses.update(comp_loss.data[0], out_frames.size(0))

        act_acc = accuracy(activity_out, activity_target)
        act_accuracies.update(act_acc[0].data[0], activity_out.size(0))

        fg_acc = accuracy(activity_out.view(-1, 2, activity_out.size(1))[:, 0, :].contiguous(),
                          activity_target.view(-1, 2)[:, 0].contiguous())

        bg_acc = accuracy(activity_out.view(-1, 2, activity_out.size(1))[:, 1, :].contiguous(),
                          activity_target.view(-1, 2)[:, 1].contiguous())

        fg_accuracies.update(fg_acc[0].data[0], activity_out.size(0) // 2)
        bg_accuracies.update(bg_acc[0].data[0], activity_out.size(0) // 2)

        # compute gradient and do SGD step
        #sys.stdout.write('backwarding model ...')
        #sys.stdout.flush()
        loss.backward()
        #sys.stdout.write('done\n')
        #sys.stdout.flush()

        if i % args.iter_size == 0:
            # scale down gradients when iter size is functioning
            if args.iter_size != 1:
                for g in optimizer.param_groups:
                    for p in g['params']:
                        p.grad /= args.iter_size

            if args.clip_gradient is not None:
                total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
                if total_norm > args.clip_gradient:
                    print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))
            else:
                total_norm = 0

            optimizer.step()
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)

        if i % args.print_freq == 0:
            line = ''
            line += 'Epoch:[{}] [{}/{}] lr:{:.5f}'.format(epoch, i, len(train_loader), optimizer.param_groups[4]['lr'])
            line += ' Time: {}'.format(datetime.now())
            line += ' BatchTime:{batch_time.val:.3f}({batch_time.avg:.3f})'.format(batch_time=batch_time)
            line += ' DataTime:{data_time.val:.3f}({data_time.avg:.3f})\n'.format(data_time=data_time)
            line += '      Loss:{loss.val:.4f}({loss.avg:.4f})'.format(loss=losses)
            line += ' Act.Loss:{act_losses.val:.3f}({act_losses.avg:.3f})'.format(act_losses=act_losses)
            line += ' Comp.Loss:{comp_losses.val:.3f}({comp_losses.avg:.3f})'.format(comp_losses=comp_losses)
            line += ' Reg.Loss:{reg_loss.val:.3f}({reg_loss.avg:.3f})\n'.format(reg_loss=reg_losses)
            line += '      Act.Accuracy:{act_acc.val:.02f}({act_acc.avg:.2f})'.format(act_acc=act_accuracies)
            line += ' Act.FG:{fg_acc.val:.02f}({fg_acc.avg:.02f})'.format(fg_acc=fg_accuracies)
            line += ' Act.BG:{bg_acc.avg:.02f}({bg_acc.avg:.02f})\n'.format(bg_acc=bg_accuracies)
            sys.stdout.write(line)
            sys.stdout.flush()

        end = time.time()


def validate(val_loader, model, act_criterion, comp_criterion, regression_criterion, iter):
    batch_time = AverageMeter()
    losses = AverageMeter()
    act_losses = AverageMeter()
    comp_losses = AverageMeter()
    reg_losses = AverageMeter()
    act_accuracies = AverageMeter()
    fg_accuracies = AverageMeter()
    bg_accuracies = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    ohem_num = val_loader.dataset.fg_per_video
    comp_group_size = val_loader.dataset.fg_per_video + val_loader.dataset.incomplete_per_video
    for i, (out_frames, out_prop_len, out_prop_scaling, out_prop_type, out_prop_labels, out_prop_reg_targets, out_stage_split) \
            in enumerate(val_loader):
        target = out_prop_labels.cuda(async=True)
        input_var = torch.autograd.Variable(out_frames, volatile=True)
        scaling_var = torch.autograd.Variable(out_prop_scaling, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        reg_target_var = torch.autograd.Variable(out_prop_reg_targets)
        prop_type_var = torch.autograd.Variable(out_prop_type)

        # compute output

        activity_out, activity_target, \
        completeness_out, completeness_target, \
        regression_out, regression_labels, regression_target = model(input_var, scaling_var, target_var,
                                                                     reg_target_var, prop_type_var)
#        print (regression_out)
 #       print (regression_labels)
  #      print (regression_target)
        act_loss = act_criterion(activity_out, activity_target)
        comp_loss = comp_criterion(completeness_out, completeness_target, ohem_num, comp_group_size)
        reg_loss = regression_criterion(regression_out, regression_labels, regression_target)

        loss = act_loss + comp_loss * args.comp_loss_weight + reg_loss * args.reg_loss_weight

        reg_losses.update(reg_loss.data[0], out_frames.size(0))

        # measure loss and record
        losses.update(loss.data[0], out_frames.size(0))
        act_losses.update(act_loss.data[0], out_frames.size(0))
        comp_losses.update(comp_loss.data[0], out_frames.size(0))

        act_acc = accuracy(activity_out, activity_target)
#        print (activity_target)
#        exit()

        act_accuracies.update(act_acc[0].data[0], activity_out.size(0))

        fg_acc = accuracy(activity_out.view(-1, 2, activity_out.size(1))[:, 0, :].contiguous(),
                          activity_target.view(-1, 2)[:, 0].contiguous())

        bg_acc = accuracy(activity_out.view(-1, 2, activity_out.size(1))[:, 1, :].contiguous(),
                          activity_target.view(-1, 2)[:, 1].contiguous())

        fg_accuracies.update(fg_acc[0].data[0], activity_out.size(0)//2)
        bg_accuracies.update(bg_acc[0].data[0], activity_out.size(0)//2)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}] '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\n'
                  '      Loss {loss.val:.4f} ({loss.avg:.4f})\n'
                  '      Act. Loss {act_loss.val:.3f} ({act_loss.avg:.3f})\n'
                  '      Comp. Loss {comp_loss.val:.3f} ({comp_loss.avg:.3f})\n'
                  '      Act. Accuracy {act_acc.val:.02f} ({act_acc.avg:.2f}) FG {fg_acc.val:.02f} BG {bg_acc.val:.02f}\n'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                    act_loss=act_losses, comp_loss=comp_losses, act_acc=act_accuracies,
                    fg_acc=fg_accuracies, bg_acc=bg_accuracies) +
                  '      Reg. Loss {reg_loss.val:.3f} ({reg_loss.avg:.3f})'.format(reg_loss=reg_losses))

    print('Testing Results: Loss {loss.avg:.5f}\n'
          '                 Activity Loss {act_loss.avg:.3f} Completeness Loss {comp_loss.avg:.3f}\n'
          '                 Act Accuracy {act_acc.avg:.02f} FG Acc. {fg_acc.avg:.02f} BG Acc. {bg_acc.avg:.02f}\n'
          .format(act_loss=act_losses, comp_loss=comp_losses, loss=losses, act_acc=act_accuracies,
                  fg_acc=fg_accuracies, bg_acc=bg_accuracies) +
          '                 Regression Loss {reg_loss.avg:.3f}'.format(reg_loss=reg_losses))

    return losses.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    epoch_name = 'epoch-' + str(state['epoch'])
    filename = 'ssn'+'_'.join((args.snapshot_pref, args.dataset, args.arch, args.modality.lower(), epoch_name, filename))
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((args.snapshot_pref, args.modality.lower(), 'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)


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


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
#    print (pred)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
