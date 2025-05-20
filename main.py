import torch
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

from data.load_data import get_path
from models.HDD import Model
from options.TrainOption import TrainParser
from train.trainer import Trainer
from utils.seed import seed_everything

def main(train_args):
    torch.cuda.empty_cache()
    device = train_args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device : {device}")

    seed_everything(train_args)
  
    # get path
    get_path(train_args)

    train_ct = pd.read_excel(f"D:\\HIPPO\\train_.xlsx")['CT']
    train_hippo = pd.read_excel(f"D:\\HIPPO\\train_.xlsx")['HIPPO']

    test_ct = pd.read_excel(f"D:\\HIPPO\\test_.xlsx")['CT']
    test_hippo = pd.read_excel(f"D:\\HIPPO\\test_.xlsx")['HIPPO']

    print(f"Load Data\nTrain : {len(train_ct)} Test : {len(test_ct)}")

    # define models
    # Unet, Unet edge, Unet module
    model = Model(1, 1).to(device)
    model = torch.nn.DataParallel(model).to(device)
    print(f"Model Parameter : {sum(p.numel() for p in model.parameters())}")
    # trainer
    trainer = Trainer(train_args,
                      train_ct,
                      train_hippo,
                      model)
    
    trainer.train()

def get_parser(Parser):
    parser = Parser()
    return parser.parse()

if __name__=='__main__':
    train_args = get_parser(TrainParser)
    print(train_args)
    
    main(train_args)