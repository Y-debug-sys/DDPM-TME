import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def load_data(args):
    if args.dataset == "abilene":
        base = "./Dataset/Abilene"
        tm_file = os.path.join(base, "Abilene_TM.csv")
        rm_file = os.path.join(base, "abilene_rm.csv")
        div_num = 10 ** 9
        mult_num = 288
        train_day = 16 * 7
        test_day = 1 * 7

    if args.dataset == "geant":
        base = "./Dataset/GEANT"
        tm_file = os.path.join(base, "GEANT_TM.csv")
        rm_file = os.path.join(base, "geant_rm.csv")
        div_num = 10 ** 7
        mult_num = 96
        train_day = 11 * 7
        test_day = 1 * 7

    data = pd.read_csv(tm_file, header=None)
    data.drop(data.columns[-1], axis=1, inplace=True)

    data_tensor = torch.from_numpy(data.values / div_num)
    data_tensor = data_tensor.float()

    rm = pd.read_csv(rm_file, header=None)
    rm.drop(rm.columns[-1], axis=1, inplace=True)
    rm_tensor = torch.from_numpy(rm.values)
    rm_tensor = rm_tensor.float()

    train_size = int(train_day * mult_num)
    test_size = int(test_day * mult_num)

    train_id = np.arange(train_size)
    test_id = np.arange(test_size) + train_size
    train_flow = data_tensor[train_id]
    test_flow = data_tensor[test_id]

    test_link = test_flow @ rm_tensor

    return train_flow, test_flow, test_link, rm_tensor


class EarlyStopping:

    def __init__(self, var_tensor, patience=50, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_tensor = var_tensor
        self.early_stop = False
        self.opt_loss_min = np.Inf

    def __call__(self, opt_loss, var_tensor):
        score = -opt_loss

        if self.best_score is None:
            self.best_score = score
            self.save(opt_loss, var_tensor)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}.')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save(opt_loss, var_tensor)
            self.counter = 0

    def save(self, opt_loss, var_tensor):
        self.best_tensor = var_tensor
        if self.verbose:
            print(f'Loss decreased ({self.opt_loss_min:.6f} --> {opt_loss:.6f}).')
        self.opt_loss_min = opt_loss


def visualization(ori_data, generated_data, analysis):
    """Using PCA or tSNE for generated and original data visualization.
  
  Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    - analysis: tsne or pca
  """

    # Analysis sample size (for faster computation)
    anal_sample_no = min([5000, ori_data.shape[0]])
    idx = np.random.permutation(ori_data.shape[0])[:anal_sample_no]

    prep_data = ori_data[idx]
    prep_data_hat = generated_data[idx]

    # Visualization parameter
    colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]

    if analysis == 'pca':
        # PCA Analysis
        pca = PCA(n_components=2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        # Plotting
        f, ax = plt.subplots(1)
        plt.scatter(pca_results[:, 0], pca_results[:, 1],
                    c=colors[:anal_sample_no], alpha=0.2, label="Original")
        plt.scatter(pca_hat_results[:, 0], pca_hat_results[:, 1],
                    c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")

        ax.legend()
        plt.title('PCA plot')
        plt.xlabel('x-pca')
        plt.ylabel('y_pca')
        plt.show()

    elif analysis == 'tsne':

        # Do t-SNE Analysis together
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

        # TSNE anlaysis
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(prep_data_final)

        # Plotting
        f, ax = plt.subplots(1)

        plt.scatter(tsne_results[:anal_sample_no, 0], tsne_results[:anal_sample_no, 1],
                    c=colors[:anal_sample_no], alpha=0.2, label="Original")
        plt.scatter(tsne_results[anal_sample_no:, 0], tsne_results[anal_sample_no:, 1],
                    c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")

        ax.legend()

        plt.title('t-SNE plot')
        plt.xlabel('x-tsne')
        plt.ylabel('y_tsne')
        plt.show()
