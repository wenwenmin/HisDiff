import warnings
import anndata
import numpy as np
import pandas as pd
import torch


from sympy.utilities.exceptions import ignore_warnings
from torch.nn.utils import skip_init
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import csv
import scprep as scp
import scanpy as sc
warnings.filterwarnings("ignore")

class hisdiff_dataset(Dataset):
    def __init__(self, args):
        self.mode = args.mode
        if self.mode =='train':
            slide_name_list = list(np.genfromtxt(args.data_path + 'processed_data/' + args.slidename_list,dtype='str'))
            for slide_out in args.slide_out.split(','):
                slide_name_list.remove(slide_out)
                print(f"{slide_out} is removed for testing")
            selected_genes = list(np.genfromtxt(args.data_path + 'processed_data/' + args.gene_list, dtype='str'))
            args.input_gene_size = len(selected_genes)
            print(f'len selected genes : {len(selected_genes)}')
            first = True
            all_gene_mtx = None
            all_local_ebd = None
            all_neighbor_ebd = None
            all_global_ebd = None
            for i in range(len(slide_name_list)):
                slide = slide_name_list[i]
                gene_mtx_ = anndata.read_h5ad(args.data_path + 'st/' + slide + '.h5ad')
                gene_mtx = pd.DataFrame(gene_mtx_[:, selected_genes].X.toarray(),
                                        columns=selected_genes,
                                        index=[slide + '_' + str(j) for j in range(gene_mtx_.shape[0])])
                if first:
                    all_gene_mtx = gene_mtx
                    local_ebd = torch.load(args.data_path + 'processed_data/local_ebd/' + slide + '.pt')
                    neighbor_ebd = torch.load(args.data_path + 'processed_data/neighbor_ebd/' + slide + '.pt')
                    global_ebd = torch.load(args.data_path + 'processed_data/global_ebd/' + slide + '.pt')
                    all_local_ebd = local_ebd
                    all_neighbor_ebd = neighbor_ebd
                    all_global_ebd = global_ebd
                    first = False
                    continue
                local_ebd = torch.load(args.data_path + 'processed_data/local_ebd/' + slide + '.pt')
                neighbor_ebd = torch.load(args.data_path + 'processed_data/neighbor_ebd/' + slide + '.pt')
                global_ebd = torch.load(args.data_path + 'processed_data/global_ebd/' + slide + '.pt')
                all_local_ebd = torch.cat([all_local_ebd, local_ebd], axis=0)
                all_neighbor_ebd = torch.cat([all_neighbor_ebd, neighbor_ebd], axis=0)
                all_gene_mtx = np.concatenate([all_gene_mtx, gene_mtx], axis=0)
                all_global_ebd = torch.cat([all_global_ebd, global_ebd], axis=0)
                print(
                    f"{slide} loaded, gene mtx shape: {all_gene_mtx.shape}, img ebd shape:{all_local_ebd.shape}, neighbor ebd shape:{all_neighbor_ebd.shape}, "
                    )

            args.cond_size = all_local_ebd.shape[1]
            all_gene_mtx_df = pd.DataFrame(all_gene_mtx, columns=selected_genes,
                                           index=list(range(all_gene_mtx.shape[0])))
            # remove the spot with all NAN/zero in gene mtx
            all_gene_mtx_all_nan_spot_index = all_gene_mtx_df.index[all_gene_mtx_df.isnull().all(axis=1)]
            all_gene_mtx_all_zero_spot_index = all_gene_mtx_df.index[all_gene_mtx_df.sum(axis=1) == 0]
            print(f"All NAN spot index: {all_gene_mtx_all_nan_spot_index}")
            print(f"All zero spot index: {all_gene_mtx_all_zero_spot_index}")
            spot_idx_to_remove = list(set(all_gene_mtx_all_nan_spot_index) | set(all_gene_mtx_all_zero_spot_index))
            spot_idx_to_keep = list(set(all_gene_mtx_df.index) - set(spot_idx_to_remove))
            all_gene_mtx = all_gene_mtx[spot_idx_to_keep, :]
            all_gene_mtx = np.log2(all_gene_mtx + 1)
            all_gene_mtx_selected_genes = all_gene_mtx.copy()
            all_local_ebd = all_local_ebd[spot_idx_to_keep, :]
            all_neighbor_ebd = all_neighbor_ebd[spot_idx_to_keep, :]
            all_global_ebd = all_global_ebd[spot_idx_to_keep, :]
        else:
            selected_genes = list(np.genfromtxt(args.data_path + 'processed_data/' + args.gene_list, dtype='str'))
            args.input_gene_size = len(selected_genes)
            print(f'len selected genes : {len(selected_genes)}')
            slide = args.slide_out
            gene_mtx_ = anndata.read_h5ad(args.data_path + 'st/' + slide + '.h5ad')
            gene_mtx = pd.DataFrame(gene_mtx_[:, selected_genes].X.toarray(),
                                    columns=selected_genes,
                                    index=[slide + '_' + str(j) for j in range(gene_mtx_.shape[0])])

            all_gene_mtx = gene_mtx
            local_ebd = torch.load(args.data_path + 'processed_data/local_ebd/' + slide + '.pt')
            neighbor_ebd = torch.load(args.data_path + 'processed_data/neighbor_ebd/' + slide + '.pt')
            global_ebd = torch.load(args.data_path + 'processed_data/global_ebd/' + slide + '.pt')
            all_local_ebd = local_ebd
            all_neighbor_ebd = neighbor_ebd
            all_global_ebd = global_ebd

            print(
                f"{slide} loaded, gene mtx shape: {all_gene_mtx.shape}, img ebd shape:{all_local_ebd.shape}, neighbor ebd shape:{all_neighbor_ebd.shape}, "
            )

            args.cond_size = all_local_ebd.shape[1]

            all_gene_mtx_df = pd.DataFrame(all_gene_mtx, columns=selected_genes,
                                           index=list(range(all_gene_mtx.shape[0])))
            all_gene_mtx_selected_genes = np.log2(all_gene_mtx + 1).copy().to_numpy()
        self.gene_mtx = all_gene_mtx_selected_genes
        self.local_ebd = all_local_ebd
        self.neighbor_ebd = all_neighbor_ebd
        self.global_ebd = all_global_ebd
        self.args = args

    def __len__(self):
        return len(self.gene_mtx)

    def __getitem__(self, idx):
        gene = self.gene_mtx[idx]
        local_ebd = self.local_ebd[idx]
        neighbor_ebd = self.neighbor_ebd[idx]
        global_ebd = self.global_ebd[idx]

        return gene, local_ebd, neighbor_ebd, global_ebd

    def get_args(self):
        return self.args

