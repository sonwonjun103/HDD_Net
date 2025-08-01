import torch
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from data.dataset import CustomDataset
from utils.utils import inpaint, get_boundary_map
from torch.utils.data import DataLoader
from utils.loss import BCEDiceLoss

class Trainer():
    def __init__(self,
                 args,
                 train_ct,
                 train_hippo,
                 model):
        self.args = args
        self.dataset = CustomDataset(args, train_ct, train_hippo) 

        self.model = model
        self.loss_fn = BCEDiceLoss(1,1).to(self.args.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
        self.model_save_path = f"./model_parameters"
        self.filename = f"{self.args.filename}.pt"
        
    def build_dataloader(self):
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size = self.args.batch_size,
            shuffle=self.args.shuffle
        )
 
    def train(self):
        assert os.path.exists(os.path.join(self.model_save_path))
        
        self.build_dataloader()
        self.model.train()

        device = self.args.device
        
        scaler = torch.cuda.amp.GradScaler()

        total_loss = []
        size = len(self.dataloader)
        for epoch in range(self.args.epochs):
            print(f"Start epoch : {epoch+1} / {self.args.epochs}")
            epoch_loss = 0
            epoch_start = time.time()
            for batch, (ct, hippo, edge) in enumerate(self.dataloader):
                ct = ct.to(device).float()
                hippo = hippo.to(device).float()
                edge = edge.to(device).float()

                with torch.cuda.amp.autocast():
                    if self.args.edge == 1:
                        output, edge_outptut = self.model(ct)
    
                        loss1_1, loss1_2 = self.loss_fn(hippo, output)
                        loss2_1, loss2_2 = self.loss_fn(edge, edge_outptut)
                        
                        output_edge2 = get_boundary_map(output.detach().cpu().numpy())
                        loss3_1, loss3_2 = self.loss_fn(edge, output_edge2.to(device).float())

                        impaint_output = inpaint(edge_outptut.detach().cpu().numpy())
                        loss4_1, loss4_2 = self.loss_fn(hippo, impaint_output.to(device).float())

                        loss1 = loss1_1 + loss1_2 # seg 1
                        loss2 = 0.2 * loss2_1 + 0.2 * loss2_2 # edge 1
                        loss3 = 0.2 * loss3_1 + 0.2 * loss3_2 # edge 2
                        loss4 = loss4_1 + loss4_2 # seg 2

                        loss = loss1 + loss2 + loss3 + loss4

                        if batch % 5 == 0:
                            print(f"Batch loss => {loss:>.5f} = {loss1_1:>.5f} + {loss1_2:>.5f} + {loss2_1:>.5f} + {loss2_2:>.5f}+ {loss3_1:>.5f} + {loss3_2:>.5f} + {loss4_1:>.5f} + {loss4_2:>.5f} {batch}/{size}")
                    else:
                        output = self.model(ct)

                        loss1_1, loss1_2 = self.loss_fn(hippo, output)
                        loss1  = loss1_1 + loss1_2
                        loss = loss1

                        if batch % 5 == 0:
                            print(f"Batch loss => {loss:>.5f} = {loss1_1:>.5f} + {loss1_2:>.5f} {batch}/{size}")

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                #loss.backward()
                scaler.update()
                #self.optimizer.step()

                epoch_loss += loss

            epoch_end = time.time()
            print(f"    Epoch Loss : {epoch_loss/size:>.5f}")
            print(f"    Epoch Time : {(epoch_end - epoch_start) // 60} min {(epoch_end  - epoch_start) % 60} sec")
            print(f"End Epoch : {epoch+1} / {self.args.epochs}")
            print()

            total_loss.append(epoch_loss.detach().cpu()/size)

            torch.save(self.model.state_dict(), os.path.join(self.model_save_path, f"{self.args.model}_{self.filename}"))

        plt.title(f'{self.args.model}')
        plt.plot(total_loss)
        plt.savefig(f"./{self.filename.split('.')[0]}.png")