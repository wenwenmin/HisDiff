import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
import sys
from model import HisDiff,Unet
from diffusion import create_diffusion
import argparse
import pandas as pd
import numpy as np
import os
from datasets import hisdiff_dataset


def find_model(model_name, device=""):
    assert os.path.isfile(model_name), f'Could not find checkpoint at {model_name}'
    checkpoint = torch.load(model_name)['model']
    return checkpoint


def main(args):
    # Setup PyTorch:
    torch.set_grad_enabled(False)
    device = args.device

    model = HisDiff(
        input_size=args.input_gene_size,
        depth=args.DiT_num_blocks,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        label_size=args.cond_size,
    )

    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path, device=args.device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    diffusion = create_diffusion(str(args.num_sampling_steps))

    loader = DataLoader(args.dataset, batch_size=args.sampling_batch_size, shuffle=False)
    all_samples = None
    first_batch = True
    i = 0
    for _,local_ebd,neighbor_ebd,global_ebd in loader:
        local_ebd = local_ebd.to(device)
        neighbor_ebd = neighbor_ebd.to(device)
        global_ebd = global_ebd.to(device)

        z = torch.randn(local_ebd.shape[0], 1, args.input_gene_size, device=device)
        model_kwargs = dict(local_ebd=local_ebd,
                            neighbor_ebd=neighbor_ebd,
                            global_ebd=global_ebd,
                         )
        samples = diffusion.p_sample_loop(
            model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
        if first_batch:
            all_samples = samples.detach().cpu()
            first_batch = False
        else:
            all_samples = torch.cat((all_samples, samples.detach().cpu()), dim=0)
        print(str(i) + "/" + str(len(loader)) + " DONE")
        i += 1

    torch.save(all_samples, args.save_path + "generated_samples_" + args.ckpt.split("/")[-1].split(".")[0] + "_" + args.slide_out + ".pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model parameters
    parser.add_argument("--model", type=str, default="DiffusionModel")
    parser.add_argument("--DiT_num_blocks", type=int, default=12, help="DiT depth")
    parser.add_argument("--hidden_size", type=int, default=384, help="DiT hidden dimension")
    parser.add_argument("--num_heads", type=int, default=12, help="DiT heads")

    # test slide & gene list
    parser.add_argument("--slide_out", type=str, default="MISC2", help="Test slide ID")
    parser.add_argument("--gene_list", type=str, default="selected_gene_list.txt")
    parser.add_argument("--mode", type=str, default="test", help="Running mode (train/test)")
    # sampling parameter

    parser.add_argument("--num_sampling_steps", type=int, default=1000, help="Sampling steps")
    parser.add_argument("--sampling_batch_size", type=int, default=128,
                        help="  size when sampling. Reduce if GPU memory is limited")

    parser.add_argument("--save_path", type=str, 
                        default="") # TODO set to path like: ./Her2st_results/runs/000(位置)/samples/.
    parser.add_argument("--ckpt", type=str,
                        default="")  # TODO set to ckpt path like: ./Her2st_results/runs/000(位置)/checkpoints/0300000.pt


    parser.add_argument("--data_path", type=str, default="./hest1k_datasets/Her2st/")

    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # load image patches
    data_path = args.data_path
    img_ebd = torch.load(data_path + "processed_data/local_ebd/" + args.slide_out + ".pt")

    args.cond_size = img_ebd.shape[1]

    # load gene list
    selected_genes = np.genfromtxt(data_path + "processed_data/" + args.gene_list, dtype=str)
    print("Selected genes are in file - ", args.gene_list)
    args.input_gene_size = len(selected_genes)

    # create dataset
    dataset = hisdiff_dataset(args)
    args.dataset = dataset
    args = dataset.get_args()
    main(args)