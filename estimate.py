import torch
import numpy as np

from torch.optim import Adam
from torch import nn

from utils import EarlyStopping


def TME(model_1, model_2, link_loader, rm_tensor, args):
    device = model_1.alphas_cumprod.device

    if args.loss_func_2 == "l1":
        criteon = nn.L1Loss().to(device)
        mult = 100
        lr = args.lr_2
    elif args.loss_func_2 == "l2":
        criteon = nn.MSELoss().to(device)
        mult = 10000
        lr = args.lr_2 / 2

    model_1.requires_grad_(False)
    model_2.requires_grad = False

    predict_flows = np.empty([0, args.nodes_num * args.nodes_num])

    for _, link_tensor in enumerate(link_loader):
        concur_size = link_tensor.shape[0]

        # select a good initial point

        noise = torch.empty(concur_size, args.st, args.emb_size, args.emb_size).to(device)
        sampled_noise, sampled_flow = model_1.sample(batch_size=args.init_num)
        sampled_noise = torch.from_numpy(np.array([item.cpu().detach().numpy() for item in sampled_noise])).to(device)
        sampled_flow = model_2.recover(sampled_flow)
        # Here, sampled_noise has a shape of (args.st, args.init_num, emb_size, emb_size), 
        # while sampled_flow has a shape of (args.init_num, emb_size * emb_size).

        print(f"select initial point for {concur_size} flows ...")

        for id in range(concur_size):

            initial_loss = torch.abs(sampled_flow @ rm_tensor - link_tensor[id]) * mult
            initial_loss = torch.mean(initial_loss, dim=1)

            if args.regularize:
                mid_loss = torch.square(sampled_noise).reshape(args.st, -1, args.emb_size * args.emb_size)
                mid_loss = mid_loss.mean(dim=0).mean(dim=-1)
                initial_loss += args.lamb / concur_size * mid_loss

            start_arg = torch.argmin(initial_loss)
            noise[id, :, :, :] = sampled_noise[:, start_arg, :, :]

        torch.cuda.empty_cache()
        print("Done.")

        # estimate test flow
        noise = nn.Parameter(noise, requires_grad=True)
        optimizer = Adam([noise], lr=lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50, 150], gamma=0.5)

        early_stopping = EarlyStopping(noise, patience=args.patience, verbose=True)
        step = 0

        while step < args.epoch_2:
            flow = model_1.opt_sample(noise)
            flow = model_2.recover(flow)
            loss = criteon(flow @ rm_tensor, link_tensor) * mult

            optimizer.zero_grad()

            if args.regularize:
                loss += args.lamb / concur_size * torch.norm(noise) ** 2

            loss.backward()
            optimizer.step()
            scheduler.step()

            print(f'epoch:{step + 1}, total_loss: {loss.item():.6f}')
            step += 1

            early_stopping(loss.item(), noise)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {step + 1}.")
                break

        noise = early_stopping.best_tensor

        flow, _ = model_1.opt_sample(noise)
        flow = model_2.recover(flow)
        predict_flows = np.row_stack([predict_flows, flow.cpu().detach().numpy()])
        np.save("./predict_flows.npy", predict_flows)
