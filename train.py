import torch
from diffusion import create_diffusion
from model import HisDiff,Unet
from torch.utils.data import DataLoader, Dataset

import numpy as np
from glob import glob
import argparse
from datasets import *
import os
import tqdm
from tqdm import tqdm

def load_model_and_datasets(args):
    train_set = hisdiff_dataset(args)
    args = train_set.get_args()
    model = HisDiff(
        input_size=args.input_gene_size,
        depth=args.DiT_num_blocks,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        label_size=args.cond_size,
    ).to(args.device)
    print(f"Dataset contains {len(train_set):,} images ({args.data_path})")
    return train_set, model, args

def save_checkpoint(model,optimizer,epoch,args):
    checkpoint = {
        "model": model.state_dict(),
        "opt": optimizer.state_dict(),
        "epoch": epoch
    }
    checkpoint_path = f"{args.checkpoint_dir}/{epoch:07d}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
# import psutil
# def get_cpu_mem_gb():
#     """获取当前进程的CPU内存占用 (GB)"""
#     process = psutil.Process(os.getpid())
#     mem_bytes = process.memory_info().rss
#     return mem_bytes / 1024**3

# def get_gpu_mem_gb():
#     """获取当前GPU显存占用 (GB)"""
#     if torch.cuda.is_available():
#         mem_bytes = torch.cuda.memory_reserved()
#         return mem_bytes / 1024**3
#     return 0
def train(model,train_loader,args):
    device = args.device
    train_loader = train_loader
    args = args
    model = model.to(device)


    diffusion = create_diffusion(timestep_respacing="")
    optimizer = torch.optim.AdamW(model.parameters(),
                                       lr=args.lr, weight_decay=0)

    model.train()

    avg_loss = 0
    for epoch in range(0,args.total_epochs):
        total_loss = 0
        tqdm_train = tqdm(train_loader, total=len(train_loader))
        for gene_exp,local_ebd,neighbor_ebd,global_ebd in tqdm_train:
            x = gene_exp.unsqueeze(1).to(device)  # (N, 1, NumGene)
            x = x.float()
            local_ebd = local_ebd.to(device)
            neighbor_ebd = neighbor_ebd.to(device)
            global_ebd = global_ebd.to(device)
            # neighbor_pos = neighbor_pos.to(device)
            # global_pos = global_pos.to(device)
            t = torch.randint(0, diffusion.num_timesteps, (gene_exp.size(0),), device=x.device)
            model_kwargs = dict(local_ebd=local_ebd,
                                neighbor_ebd=neighbor_ebd,
                                global_ebd = global_ebd,
                               )
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            optimizer.zero_grad()
            loss.backward()
            total_loss +=loss.item()
            optimizer.step()
            tqdm_train.set_postfix(train_loss=loss.item(), lr=args.lr, epoch=epoch,avg_loss = avg_loss)
        avg_loss = total_loss/len(train_loader)
        # cpu_gb = get_cpu_mem_gb()
        # gpu_gb = get_gpu_mem_gb()
        # print(f"[Epoch {epoch+1}] "
        #       f"Time: {elapsed:.2f}s | "
        #       f"CPU: {cpu_gb:.2f} GB | "
        #       f"GPU: {gpu_gb:.2f} GB")
        if epoch % args.ckpt_every == 0 and epoch!=0:
            save_checkpoint(model=model,
                            optimizer=optimizer,
                            epoch=epoch,
                            args=args)

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )


def main(input_args):
    device = input_args.device
    torch.cuda.set_device(device)
    print("mkdir & set up logger...")
    # mkdir for logs and checkpoints
    os.makedirs(input_args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_index = len(glob(f"{input_args.results_dir}/*"))
    input_args.experiment_dir = f"{input_args.results_dir}/{experiment_index:03d}"  # Create an experiment folder
    input_args.checkpoint_dir = f"{input_args.experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(input_args.checkpoint_dir, exist_ok=True)
    os.makedirs(f"{input_args.experiment_dir}/samples", exist_ok=True)  # Store sampling results
    # set up training objects
    data_path = input_args.data_path
    selected_genes = np.genfromtxt(data_path + "processed_data/" + input_args.gene_list, dtype=str)
    print("Selected genes are in file - ", input_args.gene_list)
    input_args.input_gene_size = len(selected_genes)
    dataset, model, args = load_model_and_datasets(input_args)

    train_loader = prepare_dataloader(dataset, input_args.batch_size)
    train(model = model,train_loader = train_loader,args = args)

    print(f"Starting...")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # data related arguments
    parser.add_argument("--expr_name", type=str, default="Her2st")
    parser.add_argument("--data_path", type=str, default="./hest1k_datasets/Her2st/", help="Dataset path")
    parser.add_argument("--results_dir", type=str, default="./Her2st_results/SPA148/",)
    parser.add_argument("--slide_out", type=str, default="SPA148",
                        help="Test slide ID. Multiple slides separated by comma.")
    parser.add_argument("--slidename_list", type=str, default="all_slide_lst.txt",
                        help="A txt file listing file names for all training and testing slides in the dataset")
    parser.add_argument("--gene_list", type=str, default="selected_gene_list.txt", help="Selected gene list")
    parser.add_argument("--mode", type=str, default="train", help="Running mode (train/test)")
    # model related arguments
    parser.add_argument("--DiT_num_blocks", type=int, default=12, help="DiT depth")
    parser.add_argument("--hidden_size", type=int, default=384, help="DiT hidden dimension")
    parser.add_argument("--num_heads", type=int, default=12, help="DiT heads")
    parser.add_argument("--device", type=str, default='cuda:0', help="Gpu")
    # training related arguments
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--total_epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--ckpt_every", type=int, default=100, help="Number of epoch to save checkpoints.")
    input_args = parser.parse_args()
    main(input_args)
