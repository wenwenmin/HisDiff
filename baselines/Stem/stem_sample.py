import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
import sys
sys.path.append("./Stem")
from Stem.models import Stem_models
from Stem.diffusion import create_diffusion
import argparse
import pandas as pd
import numpy as np
import os


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.label = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]



def find_model(model_name, device=""):
    assert os.path.isfile(model_name), f'Could not find checkpoint at {model_name}'
    if device == "":
        checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(model_name, map_location=device)
    if "ema" in checkpoint:
        checkpoint = checkpoint["ema"]
    return checkpoint


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = args.device

    model = Stem_models[args.model](
        input_size=args.input_gene_size,
        depth= args.DiT_num_blocks,
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
    for _, y in loader:
        y = y.to(device)
        z = torch.randn(y.shape[0], 1, args.input_gene_size, device=device)
        model_kwargs = dict(y=y)
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
    torch.save(all_samples, args.save_path + "generated_samples_" + args.ckpt.split("/")[-1].split(".")[0] + "_" + str(args.sample_num_per_cond) + "sample.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model parameters
    parser.add_argument("--model", type=str, choices=list(Stem_models.keys()), default="Stem")
    parser.add_argument("--DiT_num_blocks", type=int, default=12, help="DiT depth")
    parser.add_argument("--hidden_size", type=int, default=384, help="DiT hidden dimension")
    parser.add_argument("--num_heads", type=int, default=6, help="DiT heads")

    # test slide & gene list
    parser.add_argument("--slide_out", type=str, default="MEND145", help="Test slide ID")
    parser.add_argument("--gene_list_filename", type=str, default="selected_gene_list.txt")
    
    # sampling parameter
    parser.add_argument("--sample_num_per_cond", type=int, default=20, help="Number of samples generated for each input condition")
    parser.add_argument("--num_sampling_steps", type=int, default=1000, help="Sampling steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling_batch_size", type=int, default=200, help="Batch size when sampling. Reduce if GPU memory is limited")
    
    parser.add_argument("--save_path", type=str, default="") # TODO set to path like: ./PRAD_results/runs/000/samples/
    parser.add_argument("--ckpt", type=str, default="") # TODO set to ckpt path like: ./PRAD_results/runs/000/checkpoints/0300000.pt
    parser.add_argument("--data_path", type=str, default="./hest1k_datasets/PRAD/")
    
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()

    # load image patches
    data_path = args.data_path
    img_ebd_uni   = torch.load(data_path + "processed_data/1spot_uni_ebd/"   + args.slide_out + "_uni.pt")
    img_ebd_conch = torch.load(data_path + "processed_data/1spot_conch_ebd/" + args.slide_out + "_conch.pt")
    all_img_ebd = torch.cat([img_ebd_uni, img_ebd_conch], dim=1)
    args.raw_cond = all_img_ebd
    args.cond_size = all_img_ebd.shape[1]

    # create condition matrix
    print("Image patches shape: ", args.raw_cond.shape)
    args.cond = torch.zeros_like(args.raw_cond.repeat((args.sample_num_per_cond, 1)))
    print("Total number of samples to generate: ", args.cond.shape)
    for i in range(args.sample_num_per_cond):
        args.cond[i::args.sample_num_per_cond] = args.raw_cond.clone()

    # load gene list
    selected_genes = np.genfromtxt(data_path + "processed_data/" + args.gene_list_filename, dtype=str)
    print("Selected genes are in file - ", args.gene_list_filename)
    args.input_gene_size = len(selected_genes)

    # create dataset
    args.dataset = CustomDataset(args.cond, args.cond)
    print(len(args.dataset))
    
    main(args)