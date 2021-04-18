#!/usr/bin/env python3
import argparse
import bisect
import multiprocessing
import os
import time
import numpy as np
# pylint: disable=import-error
from model import UNetD

import megengine
import megengine.autodiff as autodiff
import megengine.data as data
import megengine.data.transform as T
import megengine.distributed as dist
import megengine.functional as F
import megengine.optimizer as optim

from dataset import SIDDData, SIDDValData
from utils import batch_PSNR, MixUp_AUG
logging = megengine.logger.get_logger()


def main():
    parser = argparse.ArgumentParser(description="MegEngine NBNet")
    parser.add_argument("-d", "--data", default="/data/sidd", metavar="DIR", help="path to sidd dataset")
    parser.add_argument("--dnd", action='store_true', help="training for dnd benchmark")
    parser.add_argument(
        "-a",
        "--arch",
        default="NBNet",
    )
    parser.add_argument(
        "-n",
        "--ngpus",
        default=None,
        type=int,
        help="number of GPUs per node (default: None, use all available GPUs)",
    )
    parser.add_argument(
        "--save",
        metavar="DIR",
        default="output",
        help="path to save checkpoint and log",
    )
    parser.add_argument(
        "--epochs",
        default=70,
        type=int,
        help="number of total epochs to run (default: 70)",
    )

    parser.add_argument(
        "--steps_per_epoch",
        default=10000,
        type=int,
        help="number of steps for one epoch (default: 10000)",
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        metavar="SIZE",
        default=32,
        type=int,
        help="total batch size (default: 32)",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        metavar="LR",
        default=2e-4,
        type=float,
        help="learning rate for single GPU (default: 0.0002)",
    )

    parser.add_argument(
        "--weight-decay", default=1e-8, type=float, help="weight decay"
    )

    parser.add_argument("-j", "--workers", default=8, type=int)

    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )



    args = parser.parse_args()
# pylint: disable=unused-variable  # noqa: F841

    # get device count
    if args.ngpus:
        ngpus_per_node = args.ngpus

    # launch processes
    train_proc = dist.launcher(worker) if ngpus_per_node > 1 else worker
    train_proc(args)

def worker(args):
    # pylint: disable=too-many-statements
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank == 0:
        os.makedirs(os.path.join(args.save, args.arch), exist_ok=True)
        megengine.logger.set_log_file(os.path.join(args.save, args.arch, "log.txt"))
    # init process group

    # build dataset
    train_dataloader, valid_dataloader = build_dataset(args)
    train_queue = iter(train_dataloader)  # infinite
    steps_per_epoch = args.steps_per_epoch

    # build model
    model = UNetD(3)
    # Sync parameters
    if world_size > 1:
        dist.bcast_list_(model.parameters(), dist.WORLD)

    # Autodiff gradient manager
    gm = autodiff.GradManager().attach(
        model.parameters(),
        callbacks=dist.make_allreduce_cb("SUM") if world_size > 1 else None,
    )

    # Optimizer
    opt = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay * world_size,  # scale weight decay in "SUM" mode
    )

    # mixup
    def preprocess(image, label):
        if args.dnd:
            image, label = MixUp_AUG(image, label)
        return image, label

    # train and valid func
    def train_step(image, label):
        with gm:
            logits = model(image)
            logits = image - logits
            loss = F.nn.l1_loss(logits, label)
            gm.backward(loss)
            opt.step().clear_grad()
        return loss

    def valid_step(image, label):
        pred = model(image)
        pred = image - pred
        mae_iter = F.nn.l1_loss(pred, label)
        psnr_it = batch_PSNR(pred, label)
        #print(psnr_it.item())
        if world_size > 1:
            mae_iter = F.distributed.all_reduce_sum(mae_iter) / world_size
            psnr_it = F.distributed.all_reduce_sum(psnr_it) / world_size

        return mae_iter, psnr_it

    # multi-step learning rate scheduler with warmup
    def adjust_learning_rate(step):
        #lr = 1e-6 + 0.5 * (args.lr - 1e-6)*(1 + np.cos(step/(args.epochs*steps_per_epoch) * np.pi))
        lr = args.lr * (np.cos(step / (steps_per_epoch * args.epochs) * np.pi) + 1) / 2
        for param_group in opt.param_groups:
            param_group["lr"] = lr
        return lr

    # start training
    for step in range(0, int(args.epochs * steps_per_epoch)):
        #print(step)
        lr = adjust_learning_rate(step)

        t_step = time.time()

        image, label = next(train_queue)
        if step > steps_per_epoch:
            image, label = preprocess(image, label)
        image = megengine.tensor(image)
        label = megengine.tensor(label)
        t_data = time.time() - t_step
        loss = train_step(image, label)
        t_train = time.time() - t_step
        speed = 1. / t_train
        if step % args.print_freq == 0 and dist.get_rank() == 0:
            logging.info(
                "Epoch {} Step {}, Speed={:.2g} mb/s, dp_cost={:.2g}, Loss={:5.2e}, lr={:.2e}".format(
                step // int(steps_per_epoch),
                step,
                speed,
                t_data/t_train,
                loss.item(),
                lr
            ))
        #print(steps_per_epoch)
        if (step + 1) % steps_per_epoch == 0:
            model.eval()
            loss, psnr_v = valid(valid_step, valid_dataloader)
            model.train()
            logging.info(
                "Epoch {} Test mae {:.3f}, psnr {:.3f}".format(
                (step + 1) // steps_per_epoch,
                loss.item(),
                psnr_v.item(),
            ))
            megengine.save(
                {
                    "epoch": (step + 1) // steps_per_epoch,
                    "state_dict": model.state_dict(),
                },
                os.path.join(args.save, args.arch, "checkpoint.pkl"),
            ) if rank == 0 else None

def valid(func, data_queue):
    loss = 0.
    psnr_v = 0.
    for step, (image, label) in enumerate(data_queue):
        image = megengine.tensor(image)
        label = megengine.tensor(label)
        mae_iter, psnr_it = func(image, label)
        loss += mae_iter
        psnr_v += psnr_it
    loss /= step + 1
    psnr_v /= step + 1
    return loss, psnr_v


def build_dataset(args):
    assert not args.batch_size//args.ngpus == 0 and not 4 // args.ngpus == 0
    train_dataset = SIDDData(args.data, length=args.batch_size*args.steps_per_epoch)
    train_sampler = data.Infinite(
        data.RandomSampler(train_dataset, batch_size=args.batch_size//args.ngpus, drop_last=True)
    )
    train_dataloader = data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        num_workers=args.workers,
    )
    valid_dataset = SIDDValData(args.data)
    valid_sampler = data.SequentialSampler(
        valid_dataset, batch_size=4//args.ngpus, drop_last=False
    )
    valid_dataloader = data.DataLoader(
        valid_dataset,
        sampler=valid_sampler,
        num_workers=args.workers,
    )
    return train_dataloader, valid_dataloader



if __name__ == "__main__":
    main()

# vim: ts=4 sw=4 sts=4 expandtab
