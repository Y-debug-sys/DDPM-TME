import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment", add_help=False)

    parser.add_argument('--batch_size', type=int, default=64,
                        help="Batch size")
    parser.add_argument('--concur_size', type=int, default=100,
                        help="Concurrent size")
    parser.add_argument('--patience', type=int, default=50,
                        help="Patience")
    parser.add_argument('--epoch_1', type=int, default=100000,
                        help="Training Epoch")
    parser.add_argument('--epoch_2', type=int, default=300,
                        help="Optimizing Epoch")
    parser.add_argument('--loss_func_1', type=str, default="l1",
                        choices=["l1", "l2"],
                        help="Loss Function for Training")
    parser.add_argument('--loss_func_2', type=str, default="l1",
                        choices=["l1", "l2"],
                        help="Loss Function for Optimization")
    parser.add_argument('--dataset', type=str, default="abilene",
                        choices=["abilene", "geant"],
                        help="Dataset")
    parser.add_argument('--dim_mults', type=tuple, default=(1, 2, 4),
                        help="Dimensional Multiple")
    parser.add_argument('--lr_1', type=float, default=1e-4,
                        help="Training Learning Rate")
    parser.add_argument('--lr_2', type=float, default=4e-2,
                        help="Optimizing Learning Rate (Start)")
    parser.add_argument('--hd', type=int, default=32,
                        help="Basic Hidden Dimension")
    parser.add_argument('--tt', type=int, default=1000,
                        help="Training Timesteps")
    parser.add_argument('--st', type=int, default=200,
                        help="Sampling Timesteps")
    parser.add_argument('--schedule', type=str, default="cosine",
                        choices=["linear", "cosine"],
                        help="Noise Schedule")
    parser.add_argument('--regularize', type=bool, default=False,
                        help="Adding a Regularization Term or not")
    parser.add_argument('--plot', type=bool, default=True,
                        help="Plotting Similarity Comparision")
    parser.add_argument('--visualize', type=str, default="pca",
                        choices=["tsne", "pca"],
                        help="Plotting t-SNE or PCA")
    parser.add_argument('--lamb', type=int, default=1e-4,
                        help="Lambda")
    parser.add_argument('--init_num', type=int, default=1000,
                        help="Initial Point Searching Epochs")
    parser.add_argument('--pre_ep', type=int, default=10000,
                        help="Pre-training Epochs of Embedding-Recovery Network")

    args = parser.parse_args()
    return args
