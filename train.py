import os
import torch
from torch.utils.data import DataLoader

from torch import nn
from torch.optim import Adam
from pathlib import Path
from tqdm.auto import tqdm
from torch.nn.utils import clip_grad_norm_


def exists(x):
    return x is not None


def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer(object):

    def __init__(self, preprocess_model, diffusion_model, data_tensor, results_folder='./Model', train_lr=1e-4,
                 train_num_steps=10000, adam_betas=(0.9, 0.99), train_batch_size=32, shuffle=True, pre_epoch=10000,
                 gradient_accumulate_every=5):
        super().__init__()

        self.preprocess_model = preprocess_model
        self.pre_epoch = pre_epoch

        self.model = diffusion_model
        self.device = diffusion_model.alphas_cumprod.device

        self.batch_size = train_batch_size
        self.train_num_steps = train_num_steps
        self.seq_length = diffusion_model.seq_length
        self.gradient_accumulate_every = gradient_accumulate_every

        self.save_cycle = 10000
        # assert train_num_steps % 10 == 0, 'number of train steps must be n*10'

        dl = DataLoader(data_tensor, batch_size=train_batch_size, shuffle=shuffle, num_workers=0)
        self.dl = cycle(dl)
        self.step = 0

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)
        self.opt_ae = Adam(preprocess_model.parameters(), lr=train_lr, betas=adam_betas)
        self.loss = nn.L1Loss().to(self.device)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'opt': self.opt.state_dict(),
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        device = self.device
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.preprocess_model.load_state_dict(torch.load(os.path.join(self.results_folder, "er.pt")))

    def train(self):
        device = self.device

        """First, train embedding and recovery network."""
        
        print("Start Embedding Network Training.")

        with tqdm(initial=self.step, total=self.pre_epoch) as pbar:

            while self.step < self.pre_epoch:
                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)
                    data_hat = self.preprocess_model(data)
                    loss = self.loss(data_hat, data)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    total_loss += loss.item()

                pbar.set_description(f'loss: {total_loss:.6f}')

                clip_grad_norm_(self.preprocess_model.parameters(), 1.0)
                self.opt_ae.step()
                self.opt_ae.zero_grad()

                self.step += 1
                pbar.update(1)

        print("Finish Embedding Network Training. Now Start Joint Training.")

        self.step = 0
        milestone = 0

        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:

            while self.step < self.train_num_steps:
                total_loss = 0.

                for _ in range(2):
                    loss_ae = 0.

                    for _ in range(self.gradient_accumulate_every):
                        data = next(self.dl).to(device)
                        data_hat = self.preprocess_model(data)
                        er_loss = self.loss(data_hat, data)
                        er_loss = er_loss / self.gradient_accumulate_every
                        er_loss.backward()
                        loss_ae += er_loss.item()

                    clip_grad_norm_(self.preprocess_model.parameters(), 1.0)
                    self.opt_ae.step()
                    self.opt_ae.zero_grad()

                self.preprocess_model.requires_grad_(False)

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)
                    data = self.preprocess_model.embedding(data)
                    loss = self.model(data.reshape(-1, 8, 8))
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    total_loss += loss.item()

                pbar.set_description(f'loss: {total_loss:.6f} and AE_loss: {loss_ae:.6f}')

                clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.opt.zero_grad()

                self.preprocess_model.requires_grad_(True)

                self.step += 1

                with torch.no_grad():
                    if self.step != 0 and self.step % self.save_cycle == 0:
                        milestone = self.step // self.save_cycle
                        self.save(milestone)

                pbar.update(1)

        torch.save(self.preprocess_model.state_dict(), os.path.join(self.results_folder, "er.pt"))
        print('training complete')

