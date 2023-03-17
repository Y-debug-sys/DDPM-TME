import torch
import numpy as np

from torch.utils.data import DataLoader

from train import Trainer
from cfg import parse_args
from utils import load_data
from model import Base, DDPM, ER
from estimate import TME
from utils import visualization


def main(args):

    if args.dataset == "abilene":
        nodes_num = 12
        emb_size = 8
    else:
        nodes_num = 23
        emb_size = 12

    """Load TMs for Training and Estimation"""

    train_flow, _, test_link, rm_tensor = load_data(args)
    test_link, rm_tensor = test_link.to(device), rm_tensor.to(device)
    link_loader = DataLoader(test_link, batch_size=args.concur_size)

    """Training"""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_model = Base(dim=args.hd, channels=emb_size, dim_mults=args.dim_mults).to(device)
    diffusion = DDPM(base_model, seq_length=emb_size, timesteps=args.tt, sampling_timesteps=args.st,
                     beta_schedule=args.schedule, loss_type=args.loss_func_1).to(device)
    preprocess_model = ER(in_dim=nodes_num * nodes_num, hidden_size=emb_size * emb_size,
                          out_dim=nodes_num * nodes_num).to(device)

    folder_name = "./Model"
    trainer = Trainer(preprocess_model, diffusion, train_flow, train_batch_size=args.batch_size,
                      train_num_steps=args.epoch_1, train_lr=args.lr_1, pre_epoch=args.pre_ep,
                      gradient_accumulate_every=2, results_folder=folder_name)
    trainer.train()

    """Plotting"""

    if args.plot:
        select_id = np.random.randint(low=0, high=train_flow.shape[0], size=(3000,))
        select_train_data = train_flow[select_id]
        _, sampled_flow = trainer.model.sample(batch_size=3000)
        sampled_flow = trainer.preprocess_model.recover(sampled_flow)
        visualization(ori_data=select_train_data.cpu().detach().numpy(),
                      generated_data=sampled_flow.cpu().detach().numpy(), analysis=args.visualize)

    """TM Estimation"""

    TME(trainer.model, trainer.preprocess_model, emb_size, link_loader, rm_tensor, nodes_num, args)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
