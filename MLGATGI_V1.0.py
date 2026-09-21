import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import gzip
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torch.utils.data import Dataset, DataLoader
from torch.amp import  autocast
from sklearn.model_selection import train_test_split
from typing import Union, List, Tuple, Optional
import glob

class Args:
    mode = 'train' # 'impute' or 'train'
    restart_training = "False" 

    ref = ''  # ref file
    target = ''  # impute file

    tihp = "True" # is phased data
    ref_comment = "##" # Reference file comment marker
    target_comment = "\t" # Target file comment marker
    ref_sep = None # Reference file delimiter (automatically inferred)
    target_sep = None # Target file delimiter (automatically inferred)
    ref_vac = "False" # Whether variants are stored column-wise in the reference file
    target_vac = "False" # Whether variants are stored column-wise in the target file
    ref_fcai = "False"  # Reference file first-column type (False=sample index)
    target_fcai = "False" # Target file first-column type (False=sample index)
    ref_file_format = "vcf"    # Reference file format: infer|vcf|csv|tsv
    target_file_format = "vcf" # Target file format: infer|vcf|csv|tsv
    which_chunk=-1

    save_dir = './w'
    compress_results = "True"

    # Chunking parameters
    co = 64                      # Chunk overlap size
    cs = 1024                    # Chunk size
    sites_per_model = 2048       # Number of SNPs processed by each model
    max_snps =  3000
    num_classes=4
    max_mr = 0.99
    min_mr = 0.5
    val_n_batches =2
    random_seed = 2025
    epochs = 100
    num_heads = 8
    embed_dim = 64
    lr = 0.0015
    batch_size = 2

    gradient_clip = 1.0
    weight_decay = 1e-5


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, heads=4, dropout=0.3, alpha=0.2, final_layer=False,residual_scale=0.5):
        super().__init__()
        self.heads = heads
        self.out_features = out_features // heads
        self.final_layer = final_layer
        self.alpha=alpha
        self.W = nn.Parameter(torch.Tensor(in_features, out_features))
        self.a = nn.Parameter(torch.Tensor(2 * self.out_features, 1))
        nn.init.orthogonal_(self.W)
        nn.init.uniform_(self.a, -0.001, 0.001)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.residual_scale=residual_scale
        self.res_linear = nn.Linear(in_features, out_features) if in_features != out_features else None


    def forward(self, h, adj):
        B, N, _ = h.shape
        Wh = torch.matmul(h, self.W).view(B, N, self.heads, self.out_features)
        a1, a2 = torch.split(self.a, self.out_features, dim=0)
        term1 = torch.einsum('bnhd,dk->bnhk', Wh, a1)
        term2 = torch.einsum('bnhd,dk->bnhk', Wh, a2)
        e = term1.unsqueeze(2) + term2.unsqueeze(1)
        e=e.squeeze(-1)
        e = self.leakyrelu(e)

        mask = adj.unsqueeze(-1).expand(-1, -1, -1, self.heads)
        if e.dtype == torch.float16:
            e = e.masked_fill(mask == 0, -60000.0)
        else:
            e = e.masked_fill(mask == 0, -1e10)
        max_vals = e.amax(dim=2, keepdim=True).detach()
        e_stable = e - max_vals
        attention = F.softmax(e_stable, dim=2)
        attention = attention + 1e-10
        attention = self.dropout(attention)
        h_prime = torch.einsum('bnmh,bmhd->bnhd', attention, Wh)
        h_prime = h_prime.contiguous().view(B, N, -1)

        res = h if self.res_linear is None else self.res_linear(h)
        return h_prime + res* self.residual_scale



class DFGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass_per_snp, heads=4, dropout=0.3, alpha=0.2, final_layer=False):
        super().__init__()
        self.att_layer = GATLayer(nfeat, nhid, heads, dropout, alpha)
        self.final_layer = final_layer
        self.dropout = dropout
        if final_layer:
            self.out_layer = GATLayer(nhid, nclass_per_snp, 1, dropout, alpha, final_layer=True)
        else:
            self.out_gat = GATLayer(nhid, nclass_per_snp, 1, dropout, alpha)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, self.training)
        x = self.att_layer(x, adj)
        x = F.dropout(x, self.dropout, self.training)
        if self.final_layer:
            return self.out_layer(x, adj)
        else:
            return self.out_gat(x, adj)

class LocalGlobal(nn.Module):
    def __init__(self, embed_dim, heads):
        super().__init__()
        self.local_gat = GATLayer(embed_dim, embed_dim, heads)
        self.global_gat = GATLayer(embed_dim, embed_dim, heads)
        self.skip_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, adj):
        local_adj= self.build_knn_adjacency(x)
        local_feat = self.local_gat(x, local_adj)

        global_feat = self.global_gat(x, adj)
        return (local_feat + global_feat) + self.skip_linear(x)

    def build_knn_adjacency(self,x, k=5):
        batch_size, num_nodes, feature_dim = x.shape
        adj_list = []

        for i in range(batch_size):
            features = x[i]
            dist_matrix = torch.cdist(features, features, p=2)
            _, indices = torch.topk(dist_matrix, k=k + 1, largest=False, dim=1)
            indices = indices[:, 1:]
            # Build the adjacency matrix
            adj = torch.zeros(num_nodes, num_nodes, device=x.device, dtype=torch.float32)
            row_indices = torch.arange(num_nodes, device=x.device).unsqueeze(1).expand(-1, k)
            adj[row_indices.reshape(-1), indices.reshape(-1)] = 1.0
            adj_list.append(adj)
        return torch.stack(adj_list)


class MLGATGI(nn.Module):
    def __init__(self, max_snps, num_snps, num_classes, embed_dim=64, heads=4,
                 chunk_size=1024, dropout_rate=0.25):
        super().__init__()
        self.num_snps = num_snps
        self.embed_dim = embed_dim
        # Embedding layer
        self.embedding = CatEmbeddings(max_snps, num_classes, embed_dim)
        self.local_global = LocalGlobal(embed_dim, heads)
        self.main_gat = DFGAT(
            nfeat=embed_dim,
            nhid=embed_dim,
            nclass_per_snp=num_classes,
            heads=heads
        )
        self.global_gat = DFGAT(num_classes, embed_dim , num_classes, 1,final_layer=True)


    def forward(self, x):
        x_emb = self.embedding(x)
        num_snps = x_emb.size(1)
        adj = torch.ones(x.size(0), num_snps, num_snps, device=x.device)
        processed = self.local_global(x_emb, adj)
        main_output = self.main_gat(processed, adj)
        return self.global_gat(main_output, adj)


class CatEmbeddings(nn.Module):
    def __init__(self, max_snps, num_alleles, embedding_dim):
        super().__init__()
        self.position_emb = nn.Embedding(max_snps, embedding_dim)
        self.allele_emb = nn.Embedding(num_alleles, embedding_dim)

    def forward(self, x):
        B, num_snps = x.shape
        positions = torch.arange(num_snps, device=x.device)  # Create position indices
        allele_emb = self.allele_emb(x)
        pos_emb = self.position_emb(positions)
        return allele_emb + pos_emb.unsqueeze(0)


class ImputationLoss(nn.Module):
    def __init__(self,offset_before , offset_after ,
                 class_weights: Optional[torch.Tensor] = None,
                 adj_reg_weight: float = 0.05,
                 ld_smooth_weight: float = 10):

        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.adj_reg_weight = adj_reg_weight
        self.ld_smooth_weight = ld_smooth_weight
        self.offset_before = offset_before
        self.offset_after=offset_after

    def forward(self,logits: torch.Tensor,targets: torch.Tensor,model: nn.Module,sites_per_model) -> torch.Tensor:

        loss=self._ce_loss(logits,targets)
        return  loss


    def _ce_loss(self, logits, targets):
        logits = logits[:, self.offset_before:logits.shape[1] - self.offset_after, :]

        logits_reshaped = logits.reshape(-1, logits.shape[-1])
        targets_reshaped = targets.reshape(-1)

        return self.cross_entropy(logits_reshaped, targets_reshaped)




def create_model(args,offset_before,offset_after):
    # Initialize the model
    model = MLGATGI(
        num_snps=args.sites_per_model ,
        max_snps=args.max_snps,
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        heads=args.num_heads,
        chunk_size=args.cs,
        #attention_range=args.co,
       # offset_before=offset_before,
       # offset_after=offset_after
    )

    # Initialize the optimizer
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr,
                            weight_decay=1e-4)
    scheduler = WarmupReduceLROnPlateau(optimizer,initial_lr_factor= 0.0005,warmup_epochs=20, factor=0.5, patience=3)
    criterion = ImputationLoss(offset_before, offset_after)

    return model, optimizer, criterion,scheduler


class EarlyStopper:
    def __init__(self, patience=35, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0.0
        self.best_loss = np.inf
        self.best_weights = None
        self.best_epoch = -1
        self.accuracy=0.0

    def step(self, current_loss,current_accuracy, model,epoch):
        if current_loss <= self.best_loss  and self.accuracy <= current_accuracy:
            self.best_loss = current_loss
            self.best_weights = model.state_dict()
            self.best_epoch = epoch
            self.accuracy = current_accuracy
            if current_loss==0 and self.best_loss==0 and current_accuracy==1.0 and self.accuracy==1:
                self.counter +=1.25
                if self.counter >= self.patience:
                    return True
            else:
                self.counter = 0.0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def restore_best_weights(self, model):
        print("Restored best epoch:",self.best_epoch+1)
        model.load_state_dict(self.best_weights)

def create_callbacks(optimizer, metric="loss", save_path="best_model.pth"):
    early_stopper = EarlyStopper(patience=30)

    return {
            'early_stopper': early_stopper,
    }


class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, initial_lr_factor=0.1, last_epoch=-1):

        self.warmup_epochs = warmup_epochs
        self.initial_lr_factor = initial_lr_factor
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            progress = self.last_epoch / self.warmup_epochs
            factor = self.initial_lr_factor + (1 - self.initial_lr_factor) * progress
            return [base_lr * factor for base_lr in self.base_lrs]
        return self.base_lrs

class WarmupReduceLROnPlateau:

    def __init__(self, optimizer, warmup_epochs=5, initial_lr_factor=0.1,
                 factor=0.5, patience=5, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.min_lr=min_lr
        self.warmup_scheduler = WarmupScheduler(optimizer, warmup_epochs, initial_lr_factor)
        self.reduce_scheduler = ReduceLROnPlateau(
            optimizer, factor=factor, patience=patience,
            verbose=True, min_lr=min_lr
        )
        self.last_epoch = -1
        self.in_warmup = True

    def step(self, metrics=None):
        self.last_epoch += 1

        if self.last_epoch < self.warmup_epochs:
            self.in_warmup = True
            self.warmup_scheduler.step()

        else:
            # Warmup complete; switch to ReduceLROnPlateau
            if self.in_warmup:
                print(f"[Epoch {self.last_epoch + 1}] Warmup complete")
                self.in_warmup = False

            if metrics is not None:
                self.reduce_scheduler.step(metrics)

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

class DataReader:

    def __init__(self):
        self.target_original_sample_cols = None
        self.target_original_keys = None
        self.target_is_gonna_be_phased = None
        self.target_set = None
        self.target_sample_value_index = 2
        self.ref_sample_value_index = 2
        self.target_file_extension = None
        self.allele_count = 2
        self.num_classes=4
        self.genotype_vals = None
        self.ref_is_phased = None
        self.reference_panel = None
        self.VARIANT_COUNT = 0
        self.IMPUTE_COUNT = 0
        self.is_phased = False
        self.MISSING_VALUE = None
        self.ref_is_hap = False
        self.target_is_hap = False
        self.ref_n_header_lines = []
        self.target_n_header_lines = []
        self.ref_separator = None
        self.delimiter_dictionary = {"vcf": "\t", "csv": ",", "tsv": "\t", "infer": "\t"}
        self.ref_file_extension = "vcf"
        self.test_file_extension = "vcf"
        self.target_is_phased = True
        self.map_values_1_vec = np.vectorize(self._map_hap_2_ind_parent_1)
        self.map_values_2_vec = np.vectorize(self._map_hap_2_ind_parent_2)
        self.map_haps_to_vec = np.vectorize(self._map_haps_2_ind)

    def read_csv(self,file_path: str,is_vcf: bool = False, is_reference: bool = False,
            separator: str = "\t",
            first_column_is_index: bool = True,
            comments: str = "##"
    ) -> pd.DataFrame:
        """
        Safely read CSV/VCF files (with gzip compression support), handling comment lines and column headers
        """
        header_lines = []
        vcf_columns = None
        root, ext = os.path.splitext(file_path)
        open_func = gzip.open if ext == '.gz' else open
        with open_func(file_path, 'rt') as f_in:
            line_counter = 0
            while True:
                pos = f_in.tell()
                line = f_in.readline()
                if not line:
                    break
                if is_vcf:
                    if line.startswith("##"):
                        line_counter += 1
                        header_lines.append(line.strip())
                    elif line.startswith("#"):
                        vcf_columns = line.strip().split(separator)
                        line_counter += 1
                        break
                    else:
                        f_in.seek(pos)
                        break
                else:
                    if line.startswith(comments[0]):
                        line_counter += 1
                        header_lines.append(line.strip())
                    else:

                        f_in.seek(pos)

                        break
            if is_vcf:
                df = pd.read_csv(
                    f_in,
                    sep=separator,
                    header=None,
                    skiprows=0
                )
                df.columns = vcf_columns

            else:
                df = pd.read_csv(
                    f_in,
                    sep=separator,
                    comment=comments[0] if comments else None,
                    header=0 if line_counter == 0 else None,
                    skiprows=line_counter
                )

        if first_column_is_index :
            df.set_index(df.columns[0], inplace=True)

        if is_reference:
            self.ref_n_header_lines = header_lines
        else:
            self.target_n_header_lines = header_lines
        return df


    def _find_file_extension(self, file_path: str, file_format: str, delimiter: str) -> tuple:
        SUPPORTED_FILE_FORMATS = ["vcf", "csv", "tsv"]
        if file_format not in ["infer"] + SUPPORTED_FILE_FORMATS:
            raise ValueError("The file format must be one of {'vcf', 'csv', 'tsv', 'infer'}")

        if file_format == 'infer':
            file_name_parts = file_path.split(".")
            for ext in reversed(file_name_parts):
                if ext in SUPPORTED_FILE_FORMATS:
                    found_format = ext
                    separator = self.delimiter_dictionary[ext] if delimiter is None else delimiter
                    break
            else:
                print("Unable to infer the file type; using TSV format by default")
                found_format = "tsv"
                separator = self.delimiter_dictionary["tsv"]
        else:
            found_format = file_format
            separator = self.delimiter_dictionary[file_format] if delimiter is None else delimiter

        return found_format, separator

    def assign_training_set(self, file_path: str, target_is_gonna_be_phased_or_haps: bool,
                            variants_as_columns: bool = False, delimiter: str = None,
                            file_format: str = "infer", first_column_is_index: bool = True,
                            comments: str = "##" ,training: bool = True) -> None:
        """
        Load the training dataset and process phasing information
        """
        self.target_is_gonna_be_phased = target_is_gonna_be_phased_or_haps
        self.ref_file_extension, self.ref_separator = self._find_file_extension(file_path, file_format, delimiter)

        if self.ref_file_extension != 'vcf':
            self.reference_panel = self.read_csv(file_path, is_reference=True, separator=self.ref_separator,
                                                 first_column_is_index=first_column_is_index, comments=comments)
            if variants_as_columns:
                self.reference_panel = self.reference_panel.T.reset_index(drop=False).rename(columns={'index': 'ID'})
        else:
            self.reference_panel = self.read_csv(file_path, is_reference=True, is_vcf=True, separator='\t',first_column_is_index=False, comments=comments)
            self.ref_sample_value_index += 8

        first_genotype = self.reference_panel.iloc[0, self.ref_sample_value_index]
        self.ref_is_hap = ('|' not in first_genotype) and ('/' not in first_genotype)

        self.ref_is_phased = '|' in first_genotype
        # Validate data compatibility
        if self.ref_is_hap and not target_is_gonna_be_phased_or_haps:
            raise ValueError("The reference data are haploid, but the target data are unphased diploid; prediction is not possible")
        if not (self.ref_is_phased or self.ref_is_hap) and target_is_gonna_be_phased_or_haps:
            raise ValueError("The reference data are unphased diploid, but the target data are phased; prediction is not possible")

        self.VARIANT_COUNT = self.reference_panel.shape[0]
        sample_type = 'haploid' if self.ref_is_hap else 'diploid'
        print(
            f"{self.reference_panel.shape[1] - (self.ref_sample_value_index - 1)} {sample_type} columns, {self.VARIANT_COUNT} rows")

        self.is_phased = target_is_gonna_be_phased_or_haps and (self.ref_is_phased or self.ref_is_hap)
        original_sep = '|' if self.ref_is_phased else '/'
        final_sep = '|' if self.is_phased else '/'

        # Process genotype values
        genotype_vals = pd.unique(self.reference_panel.iloc[:, self.ref_sample_value_index - 1:].values.ravel('K'))
        if self.ref_is_phased and not target_is_gonna_be_phased_or_haps:
            phased_to_unphased = {}
            for g in genotype_vals:
                a, b = map(int, g.split(original_sep))
                phased_to_unphased[g] = f"{min(a, b)}/{max(a, b)}"
            self.reference_panel.iloc[:, self.ref_sample_value_index - 1:].replace(phased_to_unphased, inplace=True)

        self.genotype_vals = np.unique(genotype_vals)
        self.num_classes = len(self.genotype_vals)
        if self.ref_is_hap:
            self.alleles = self.genotype_vals
        else:
            self.alleles = np.unique([a for g in self.genotype_vals for a in g.split(final_sep)])
        self.allele_count = len(self.alleles)
        self.MISSING_VALUE = self.num_classes if self.is_phased else len(self.genotype_vals)
        if  self.is_phased:
            unphased_missing = ".|."
            self.replacement_dict = {g: i for i, g in enumerate(sorted(self.genotype_vals))}
            self.replacement_dict[unphased_missing] = self.MISSING_VALUE
            self.reverse_replacement_dict = {v: k for k, v in self.replacement_dict.items()}
        else:
            self.hap_map = {str(v): i for i, v in enumerate(sorted(self.genotype_vals))}
            self.hap_map['.'] = self.MISSING_VALUE
            self.r_hap_map = {v: k for k, v in self.hap_map.items()}
            self.map_preds_2_allele = np.vectorize(lambda x: self.r_hap_map[x])

        self.SEQ_DEPTH = self.allele_count + 1 if self.is_phased else len(self.genotype_vals)
        print("Training set loading complete")

    def assign_test_set(self, file_path: str, variants_as_columns: bool = False,
                        delimiter: str = None, file_format: str = "infer",
                        first_column_is_index: bool = True, comments: str = "##") -> None:
        """
        Load the test dataset and validate compatibility
        """
        if self.reference_panel is None:
            raise RuntimeError("Call assign_training_set to load the training set first")
        self.target_file_extension, separator = self._find_file_extension(file_path, file_format, delimiter)
        if self.target_file_extension != 'vcf':
            test_df = self.read_csv(file_path, separator=separator, first_column_is_index=first_column_is_index,
                                    comments=comments)
            if variants_as_columns:
                test_df = test_df.T.reset_index(drop=False).rename(columns={'index': 'ID'})
        else:
            test_df = self.read_csv(file_path, is_vcf=True, separator='\t', first_column_is_index=False,
                                    comments=comments)
            self.target_sample_value_index += 8

        # Detect the phasing status of the test data
        first_genotype = test_df.iloc[0, self.target_sample_value_index]
        self.target_is_hap = ('|' not in first_genotype) and ('/' not in first_genotype)

        is_phased = '|' in first_genotype
        # Validate data compatibility
        if (self.target_is_hap or is_phased) and not (self.ref_is_phased or self.ref_is_hap):
            raise RuntimeError("The test data are phased, but the training data are unphased; prediction is not possible")
        if self.ref_is_hap and not (self.target_is_hap or is_phased):
            raise RuntimeError("The training data are haploid; the test data must be phased or haploid")
        #self.target_set=test_df
        if self.reference_panel['ID'].iloc[0] !='.':
            self.target_set = test_df.merge(self.reference_panel[['ID']], on='ID', how='right')
        else:
            self.target_set = test_df.merge(self.reference_panel[['POS']], on='POS', how='right')

        if self.target_file_extension == 'vcf' and self.ref_file_extension == 'vcf':
            self.target_set[self.reference_panel.columns[:9]] = self.reference_panel[self.reference_panel.columns[:9]]
        # Handle missing values
        self.target_set = self.target_set.astype('str').replace('nan', '.|.')
        self.target_set = test_df.astype('str').replace('./.', '.|.')
        self.IMPUTE_COUNT=len(self.target_set)
        print("Test set loading complete ",self.target_set.shape)


    def _map_hap_2_ind_parent_1(self, x: str) -> int:
        return self.hap_map[x.split('|')[0]]

    def _map_hap_2_ind_parent_2(self, x: str) -> int:
        return self.hap_map[x.split('|')[1]]

    def _map_haps_2_ind(self, x: str) -> int:
        return self.hap_map[x]

    def _diploids_to_hap_vecs(self, data: pd.DataFrame) -> np.ndarray:

        _x = np.empty((data.shape[1] * 2, data.shape[0]), dtype=np.int32)
        _x[0::2] = self.map_values_1_vec(data.values.T)
        _x[1::2] = self.map_values_2_vec(data.values.T)
        return _x

    def _get_forward_data(self, data: pd.DataFrame) -> np.ndarray:

        if self.is_phased:
            pd.set_option('future.no_silent_downcasting', True)

            return data.replace(self.replacement_dict).values.T.astype(np.int32)
        else:
            is_haps = "|" not in data.iloc[0, 0]
            if not is_haps:
                return self._diploids_to_hap_vecs(data)
            else:
                return self.map_haps_to_vec(data.values.T)

    def get_ref_set(self, starting_var_index: int = 0, ending_var_index: int = 0) -> np.ndarray:

        if 0 <= starting_var_index < ending_var_index:
            return self._get_forward_data(
                data=self.reference_panel.iloc[starting_var_index:ending_var_index, self.ref_sample_value_index - 1:])
        else:
            print("No valid variant interval was provided; using all data")
            return self._get_forward_data(data=self.reference_panel.iloc[:, self.ref_sample_value_index - 1:])

    def get_target_set(self, starting_var_index: int = 0, ending_var_index: int = 0) -> Tuple[np.ndarray, np.ndarray]:

        raw_data = self._get_forward_data(self.target_set.iloc[starting_var_index:ending_var_index, self.target_sample_value_index - 1:]
        ) if 0 <= starting_var_index < ending_var_index else self._get_forward_data(
            self.target_set.iloc[:, self.target_sample_value_index - 1:]
        )

        mask = np.asarray(raw_data == self.MISSING_VALUE, dtype=np.bool_)

        processed_data = np.where(mask, 0, raw_data)
        return processed_data, mask


    def _convert_hap_probs_to_diploid_genotypes(self, allele_probs: np.ndarray) -> np.ndarray:
        #Convert phased diploid genotypes
        n_haploids, n_variants = allele_probs.shape
        if n_haploids % 2 != 0:
            raise ValueError("The number of haplotypes must be even")
        genotypes = pd.DataFrame(allele_probs,dtype='i4')
        pd.set_option('future.no_silent_downcasting', True)
        genotypes.replace(self.reverse_replacement_dict, inplace=True)

        return  genotypes.values

    def _convert_hap_probs_to_hap_genotypes(self, allele_probs: np.ndarray) -> np.ndarray:

        return np.vectorize(self.r_hap_map.get)(np.argmax(allele_probs, axis=1))

    def _convert_unphased_probs_to_genotypes(self, allele_probs: np.ndarray) -> np.ndarray:

        n_samples, n_variants, _ = allele_probs.shape
        genotypes = np.zeros((n_samples, n_variants), dtype=object)
        for i in tqdm(range(n_samples)):
            for j in range(n_variants):
                pred_idx = np.argmax(allele_probs[i, j])
                genotypes[i, j] = self.reverse_replacement_dict.get(pred_idx, "./.")
        return genotypes

    def _get_headers_for_output(self, contain_probs: bool = False) -> list:
         headers = ["##fileformat=VCFv4.2",
                   '''##source=MLGATGI''',
                   '''##INFO=<ID=AF,Number=A,Type=Float,Description="Estimated Alternate Allele Frequency">''',
                   '''##INFO=<ID=MAF,Number=1,Type=Float,Description="Estimated Minor Allele Frequency">''',
                   '''##INFO=<ID=AVG_CS,Number=1,Type=Float,Description="Average Call Score">''',
                   '''##INFO=<ID=IMPUTED,Number=0,Type=Flag,Description="Marker was imputed">''',
                   '''##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">''',
                   ]
         probs_headers = [
            '''##FORMAT=<ID=DS,Number=A,Type=Float,Description="Estimated Alternate Allele Dosage : [P(0/1)+2*P(1/1)]">''',
            '''##FORMAT=<ID=GP,Number=G,Type=Float,Description="Estimated Posterior Probabilities for Genotypes 0/0, 0/1 and 1/1">''']
         if contain_probs:
             headers.extend(probs_headers)
         return headers

    def _convert_genotypes_to_vcf(self, genotypes, pred_format="GT:GP:DS"):
        new_vcf = self.target_set.copy()
        new_vcf[new_vcf.columns[self.target_sample_value_index - 1:]] = genotypes
        new_vcf["FORMAT"] = pred_format
        new_vcf["QUAL"] = "."
        new_vcf["FILTER"] = "."
        new_vcf["INFO"] = "IMPUTED"
        return new_vcf


    def write_ligated_results_to_file(self, df: pd.DataFrame, file_name: str, compress=False) -> str:
        to_write_format = self.ref_file_extension
        with gzip.open(f"{file_name}.{to_write_format}.gz", 'wt') if compress else open(
                f"{file_name}.{to_write_format}", 'wt') as f_out:
            # write info
            if self.ref_file_extension == "vcf":
                f_out.write(
                    "\n".join(self._get_headers_for_output(contain_probs="GP" in df["FORMAT"].values[0])) + "\n")
            else:
                f_out.write("\n".join(self.ref_n_header_lines))
        print(f"Data to be saved shape: {df.shape}")
        df.to_csv(f"{file_name}.{to_write_format}.gz" if compress else f"{file_name}.{to_write_format}",
                  sep=self.ref_separator, mode='a', index=False)
        return f"{file_name}.{to_write_format}.gz" if compress else f"{file_name}.{to_write_format}"


    def preds_to_genotypes(self, predictions: Union[str, np.ndarray]) -> pd.DataFrame:
        if isinstance(predictions, str):
            preds = np.load(predictions)
        else:
            preds = predictions

        target_df = self.target_set.copy()
        if not self.is_phased:
            target_df[target_df.columns[self.target_sample_value_index - 1:]] = self._convert_unphased_probs_to_genotypes(preds).T
        elif self.target_is_hap:
            target_df[target_df.columns[self.target_sample_value_index - 1:]] = self._convert_hap_probs_to_hap_genotypes(preds).T
        else:
            pred_format = "GT:GP:DS" if preds.shape[-1] == 4 else "GT"
            target_df = self._convert_genotypes_to_vcf(self._convert_hap_probs_to_diploid_genotypes(preds).T, pred_format)
        return target_df


class GenotypeDataset(Dataset):
    def __init__(self, x, training=True, depth=0, min_mr=0.5, max_mr=0.99):
        self.x = x
        self.training = training
        self.depth = depth
        self.min_mr = min_mr
        self.max_mr = max_mr

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_sample = self.x[idx]
        if self.training:
            seq_len = len(x_sample)
            masking_rate = np.random.uniform(self.min_mr, self.max_mr)
            mask_size = int(seq_len * masking_rate)
            mask_idx = np.random.choice(seq_len, mask_size, replace=False)
            x_sample = x_sample.copy()
            x_sample[mask_idx] = self.depth - 1
        x_sample = x_sample.astype(np.float32)
        return x_sample


def calculate_maf(genotype_array):
    allele_counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=3), axis=0, arr=genotype_array)
    total_alleles = 2 * genotype_array.shape[0]
    minor_allele_counts = 2 * allele_counts[2] + allele_counts[1]
    maf = minor_allele_counts / total_alleles
    return maf


def remove_similar_rows(array):
    unique_array = np.unique(array, axis=0)
    print(f"Removed {len(array) - len(unique_array)} rows; {len(unique_array)} training samples remain.")
    return unique_array



class DynamicWeightFusion(nn.Module):
    def __init__(self, model_paths: List[str],model, input_dim: int, device, hidden_dim: int = 64):

        super().__init__()
        self.models = nn.ModuleList()
        for path in model_paths:
            try:
                model_files = glob.glob(path)
                checkpoint = torch.load(model_files[-1], map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                self.models.append(model.to(device))
            except FileNotFoundError:
                raise ValueError(f"Model path {path} does not exist")

        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(model_paths)),
            nn.Softmax(dim=-1)
        ).to(device)

        self.feature_extractor = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.feature_extractor(x)
        context = h_n.squeeze(0)

        weights = self.attention(context)

        outputs = []
        for model in self.models:
            with torch.no_grad():
                outputs.append(model(x.long()))
        outputs = torch.stack(outputs, dim=1)

        merged_output = torch.sum(outputs * weights.unsqueeze(-1).unsqueeze(-1), dim=1)
        return merged_output

def add_attention_mask(x_sample, y_sample, depth, min_mr, max_mr,offset_before,offset_after):

    return x_sample, y_sample.long()

class CustomDataset(Dataset):
    def __init__(self, x, offset_before, offset_after, depth, training):
        self.x = torch.as_tensor(x)
        self.offset_before = offset_before
        self.offset_after = offset_after
        self.depth = depth

        print(f"target_shape: {x.shape}")
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        xx = self.x[idx]
        yy = xx[self.offset_before: xx.size(0) - self.offset_after]
        return xx, yy


class BatchCollator:
    def __init__(self, depth, masking_rates,offset_before,offset_after):
        self.depth = depth
        self.masking_rates = masking_rates
        self.offset_before = offset_before
        self.offset_after = offset_after
    def __call__(self, batch):
        xx_list, yy_list = zip(*batch)

        masked_batch = [
            add_attention_mask(xx, yy, self.depth, *self.masking_rates,self.offset_before,self.offset_after)
            for xx, yy in zip(xx_list, yy_list)
        ]

        xx_oh, yy_oh = zip(*masked_batch)
        return torch.stack(xx_oh), torch.stack(yy_oh)




def get_training_dataset(x, batch_size, depth, strategy=None,
                         offset_before=0, offset_after=0,
                         training=True, masking_rates=(0.5, 0.99)):

    dataset = CustomDataset(
        x=x,
        offset_before=offset_before,
        offset_after=offset_after,
        depth=depth,
        training=training
    )


    collator = BatchCollator(depth, masking_rates,offset_before,offset_after)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=2,  # Setting this to 0 during training is recommended on Windows
        pin_memory=True,
        drop_last=False,
        prefetch_factor=2,
        persistent_workers=True,
        collate_fn=collator
    )

    return loader, len(dataset)


def get_test_dataset(x, batch_size, depth):
    dataset = GenotypeDataset(x, training=False, depth=depth)
    dataloader = DataLoader(dataset, batch_size=batch_size,
        num_workers=2 ,  # Setting this to 0 during training is recommended on Windows
        pin_memory=True,
        drop_last=False,
        prefetch_factor=2,
        persistent_workers=True ,
        shuffle=False)
    return dataloader


def create_directories(save_dir,
                       models_dir="models",
                       outputs="out") -> None:
    for dd in [save_dir,
               f"{save_dir}/{models_dir}",
               f"{save_dir}/{outputs}"]:
        if not os.path.exists(dd):
            os.makedirs(dd)


def clear_dir(path) -> None:
    if os.path.exists(path):
        for entry in os.scandir(path):
            if entry.is_dir():
                clear_dir(entry)
            else:
                os.remove(entry)
        os.rmdir(path)


def load_chunk_info(save_dir, break_points):
    chunk_info = {ww: False for ww in list(range(len(break_points) - 1))}
    if os.path.isfile(f"{save_dir}/models/MLGATGI/chunks_info.json"):
        with open(f"{save_dir}/models/MLGATGI/chunks_info.json", 'r') as f:
            loaded_chunks_info = json.load(f)
            if isinstance(loaded_chunks_info, dict) and len(loaded_chunks_info) == len(chunk_info):
                print("Resuming training...")
                chunk_info = {int(k): v for k, v in loaded_chunks_info.items()}
    return chunk_info


def save_chunk_status(save_dir, chunk_info) -> None:
    with open(f"{save_dir}/models/MLGATGI/chunks_info.json", "w") as outfile:
        json.dump(chunk_info, outfile)


def save_checkpoint(model, save_dir, epoch, loss,w, is_best=False, is_final=False):
    os.makedirs(save_dir, exist_ok=True)
    if is_final:
        filename = f"models/MLGATGI/model_best_{w}_{loss:.4f}.pt"
    elif is_best:
        filename = f"model_best_epoch{epoch}_loss{loss:.4f}.pt"
    else:
        filename = f"checkpoint_epoch{epoch}_loss{loss:.4f}.pt"
    path = os.path.join(save_dir, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
    }, path)
    if is_final:
        files_to_clean = glob.glob(os.path.join(save_dir, "checkpoint_*.pt"))
        for f in files_to_clean:
            os.remove(f)
    print(f"Model saved to {path}")

def get_clip_value(name, epoch, max_epoch):
    """Calculate the dynamic clipping threshold"""
    if '.a' in name:
        initial, final = 0.05, 0.1
        return initial + (final - initial) * (1 - 0.9 ** epoch)
    elif '.W' in name:

        return 0.5
    else:
        initial = 5.0
        return initial


def evaluate(model, data_loader, device, offset_before, offset_after):

    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    total_snps = 0

    with torch.no_grad():
        for x, y in data_loader:
            batch_size = x.size(0)
            x, y = x.to(device), y.to(device)

            with autocast(device_type='cuda'):
                outputs = model(x)
                logits = outputs[:, offset_before:outputs.shape[1] - offset_after]
                # Reshape dimensions to meet cross-entropy requirements
                logits_reshaped = logits.reshape(-1, logits.shape[-1])
                targets_reshaped = y.reshape(-1)

                preds = logits_reshaped.argmax(dim=-1)

                batch_correct = (preds == targets_reshaped).sum().item()
                batch_snps = targets_reshaped.numel()

                total_correct += batch_correct
                total_snps += batch_snps

                loss = nn.CrossEntropyLoss()(logits_reshaped, targets_reshaped)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

    accuracy = total_correct / total_snps if total_snps > 0 else 0.0
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')

    print(f"Average test set accuracy: {accuracy:.4f} ({total_correct}/{total_snps})")

    return avg_loss, accuracy




def train_the_model(args) -> None:
    if args.restart_training:
        clear_dir(args.save_dir)
    assert args.max_mr > 0
    assert args.min_mr > 0
    assert args.max_mr >= args.min_mr
    NUM_EPOCHS = args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = args.batch_size
    create_directories(args.save_dir)
    with open(f"{args.save_dir}/commandline_args.json", 'w') as f:
        json.dump(vars(args), f, indent=4)
    dr = DataReader()
    dr.assign_training_set(file_path=args.ref,
                           target_is_gonna_be_phased_or_haps=args.tihp,
                           variants_as_columns=args.ref_vac,
                           delimiter=args.ref_sep,
                           file_format=args.ref_file_format,
                           first_column_is_index=args.ref_fcai,
                           comments=args.ref_comment)

    x_train_indices, x_valid_indices = train_test_split(range(dr.get_ref_set(0, 1).shape[0]),
                                                        test_size=0.2,
                                                        random_state=args.random_seed,
                                                        shuffle=True)

    break_points = list(np.arange(0, dr.VARIANT_COUNT, args.sites_per_model)) + [dr.VARIANT_COUNT]
    chunks_done = load_chunk_info(args.save_dir, break_points)
    args.num_classes = dr.num_classes
    for w in range(len(break_points) - 1):
        if chunks_done[w]:
            print(f"Skipping chunk {w + 1}/{len(break_points) - 1}.")
            continue
        if args.which_chunk != -1 and w + 1 != args.which_chunk:
            print(f"Skipping chunk {w + 1}/{len(break_points) - 1} due to your request using --which-chunk.")
            continue

        print(f"Training chunk {w + 1}/{len(break_points) - 1}")
        final_start_pos = max(0, break_points[w] - 2 * args.co)
        final_end_pos = min(dr.VARIANT_COUNT, break_points[w + 1] + 2 * args.co)

        offset_before = break_points[w] - final_start_pos
        offset_after = final_end_pos - break_points[w + 1]
        ref_set = dr.get_ref_set(final_start_pos, final_end_pos).astype(np.int32)

        print(f"Data shape: {ref_set.shape}")
        train_dataset, train_sample_count = get_training_dataset(ref_set[x_train_indices], BATCH_SIZE,
                                                                 depth=args.num_classes,
                                                                 offset_before=offset_before,
                                                                 offset_after=offset_after,
                                                                 masking_rates=(args.min_mr, args.max_mr))
        valid_dataset, _ = get_training_dataset(ref_set[x_valid_indices], BATCH_SIZE,
                                                depth=args.num_classes,
                                                offset_before=offset_before,
                                                offset_after=offset_after,
                                                training=False,
                                                masking_rates=(args.min_mr, args.max_mr))
        del ref_set
        model,optimizer,criterion,scheduler = create_model(args,offset_before,offset_after)
        model=model.to(device)
        callbacks = create_callbacks(optimizer)
        scaler = torch.amp.GradScaler(device='cuda')
        steps_per_epoch = len(train_dataset)
        for epoch in range(NUM_EPOCHS):
            model.train()
            train_loss = 0.0
            with tqdm(train_dataset, desc=f"Epoch {epoch + 1}",total=steps_per_epoch) as pbar:
                for x, y in pbar:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    with torch.amp.autocast(device_type='cuda'):
                        outputs = model(x)

                        loss = criterion(outputs, y,model,args.sites_per_model)
                    scaler.scale(loss).backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)

                    max_grad_value_after = 0.0
                    max_grad_layer_after = ""


                    scaler.step(optimizer)
                    scaler.update()

                    train_loss += loss.item()

                    pbar.set_postfix(loss=loss.item())

            val_loss,val_accuracy  = evaluate(model, valid_dataset, device,offset_before,offset_after)
            if callbacks['early_stopper'].step(val_loss,val_accuracy,model,epoch):
                print("Early stopping triggered!")
                break
            scheduler.step(val_loss)
            print(f'Epoch {epoch + 1}/{NUM_EPOCHS}, Train Loss: {train_loss / steps_per_epoch}, Valid Loss: {val_loss},Lr{optimizer.param_groups[0]['lr']}')
        callbacks['early_stopper'].restore_best_weights(model)
        save_checkpoint(model, args.save_dir, callbacks['early_stopper'].best_epoch, callbacks['early_stopper'].best_loss,w, is_final=True)
        chunks_done[w] = True
        save_chunk_status(args.save_dir, chunks_done)
    pass

def impute_the_target(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = args.batch_size
    dr = DataReader()
    dr.assign_training_set(file_path=args.ref,
                           target_is_gonna_be_phased_or_haps=args.tihp,
                           variants_as_columns=args.ref_vac,
                           delimiter=args.ref_sep,
                           file_format=args.ref_file_format,
                           first_column_is_index=args.ref_fcai,
                           comments=args.ref_comment)
    dr.assign_test_set(file_path=args.target,
                       variants_as_columns=args.target_vac,
                       delimiter=args.target_sep,
                       file_format=args.target_file_format,
                       first_column_is_index=args.target_fcai,
                       comments=args.target_comment)

    all_preds = []
    all_masks = []

    break_points = list(np.arange(0, dr.VARIANT_COUNT, args.sites_per_model)) + [dr.VARIANT_COUNT]
    args.num_classes = dr.num_classes
    last = 0
    for w in range(len(break_points) - 1):
        print(f"Imputing chunk {w + 1}/{len(break_points) - 1}")
        final_start_pos = max(0, break_points[w] - 2 * args.co)
        final_end_pos = min(dr.VARIANT_COUNT, break_points[w + 1] + 2 * args.co)
        test_dataset_np, mask_chunk = dr.get_target_set(final_start_pos, final_end_pos)
        test_dataset_np = test_dataset_np.astype(np.int32)
        test_dataset = get_test_dataset(test_dataset_np, BATCH_SIZE, depth=args.num_classes)
        offset_before = 2 * args.co if final_start_pos != 0 else 0
        offset_after = 2 * args.co
        model, optimizer, criterion, _ = create_model(args, offset_before, offset_after)
        model = model.to(device)
        li = [f"{args.save_dir}/models/MLGATGI/sv_chr21/model_best_{w}_*.pt"]
        fusion_model = DynamicWeightFusion(li, model, test_dataset_np.shape[1], device)
        predict_gpu = []
        predict = []
        with torch.inference_mode():
            for batch in tqdm(test_dataset):
                inputs = batch.to(device, non_blocking=True)
                outputs = fusion_model(inputs)
                predict_gpu.append(outputs)
        for tensor in predict_gpu:
            predict.append(tensor.cpu().numpy())
        pred_matrix = np.concatenate(predict, axis=0)
        final_result = np.zeros_like(pred_matrix.argmax(axis=-1))
        for i in range(pred_matrix.shape[0]):
            for j in range(pred_matrix.shape[1]):
                if mask_chunk[i, j]:
                    final_result[i, j] = pred_matrix[i, j].argmax()
                else:
                    final_result[i, j] = test_dataset_np[i, j]
        if w != len(break_points) - 2:
            valid_preds = final_result[:, offset_before: pred_matrix.shape[1] - offset_after]
            last += valid_preds.shape[1]
        else:
            valid_preds = final_result[:, pred_matrix.shape[1] - (dr.VARIANT_COUNT - last): pred_matrix.shape[1]]
        valid_preds = valid_preds.astype(np.int8)
        all_preds.append(valid_preds)

    # Merge predictions and masks from all chunks
    all_preds = np.hstack(all_preds)
    destination_file_path = dr.write_ligated_results_to_file(
        dr.preds_to_genotypes(all_preds),
        f"{args.save_dir}/out/MLGATGI/impute_{extract_filename_part(args.target)}",
        compress=args.compress_results
    )
    print(f"Complete; results saved to {destination_file_path}")


def extract_filename_part(file_path):
    file_path = file_path.replace('\\', '/')
    parts = file_path.split('/')
    filename = parts[-1] if parts else ''
    if filename.endswith('.vcf.gz'):
        filename = filename[:-7]
    if filename.endswith('.vcf'):
        filename = filename[:-4]
    return filename

def str_to_bool(s):
    true_values = ['true', '1']
    false_values = ['false', '0']
    lower_s = s.strip().lower()
    if lower_s in true_values:
        return True
    elif lower_s in false_values:
        return False
    else:
        raise ValueError(f"Invalid Boolean value: {s}. Accepted values are 'true', 'false', '0', and '1'.")

def main():
    args = Args()
    args.restart_training = str_to_bool(args.restart_training)
    args.tihp = str_to_bool(args.tihp) if args.tihp else args.tihp
    args.ref_vac = str_to_bool(args.ref_vac)
    args.target_vac = str_to_bool(args.target_vac)
    args.ref_fcai = str_to_bool(args.ref_fcai)
    args.target_fcai = str_to_bool(args.target_fcai)

    if not (args.save_dir.startswith("./") or args.save_dir.startswith("/")):
        args.save_dir = f"./{args.save_dir}"
    print(f"save to {args.save_dir}")

    if args.mode == 'train':
        train_the_model(args)
    else:
        impute_the_target(args)

if __name__ == '__main__':
    main()
