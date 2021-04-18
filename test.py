#!/usr/bin/env python3
from dataset import SIDDValData
from model import UNetD
import megengine.data as data
from utils import batch_PSNR
from tqdm import tqdm
import argparse
import pickle
import megengine


def test(args):
    valid_dataset = SIDDValData(args.data)
    valid_sampler = data.SequentialSampler(
        valid_dataset, batch_size=1, drop_last=False
    )
    valid_dataloader = data.DataLoader(
        valid_dataset,
        sampler=valid_sampler,
        num_workers=8,
    )
    model = UNetD(3)
    with open(args.checkpoint, "rb") as f:
        state = pickle.load(f)
    model.load_state_dict(state["state_dict"])
    model.eval()

    def valid_step(image, label):
        pred = model(image)
        pred = image - pred
        psnr_it = batch_PSNR(pred, label)
        return psnr_it

    def valid(func, data_queue):
        psnr_v = 0.
        for step, (image, label) in tqdm(enumerate(data_queue)):
            image = megengine.tensor(image)
            label = megengine.tensor(label)
            psnr_it = func(image, label)
            psnr_v += psnr_it
        psnr_v /= step + 1
        return psnr_v

    psnr_v = valid(valid_step, valid_dataloader)
    print("PSNR: {:.3f}".format(psnr_v.item()) )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MegEngine NBNet")
    parser.add_argument("-d", "--data", default="/data/sidd", metavar="DIR", help="path to imagenet dataset")
    parser.add_argument("-c", "--checkpoint", help="path to checkpoint")
    args = parser.parse_args()
    test(args)



# vim: ts=4 sw=4 sts=4 expandtab
