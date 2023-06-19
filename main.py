import argparse
import os,sys
import datetime
import torch
import requests
import hydra
from hydra.core.config_store import ConfigStore
from torch.utils.data import DataLoader
import omegaconf
from pytorch_lightning.loggers import WandbLogger
from src.train import *
from src.model import *
from src.eval import *
from src.dataio import *
from src.preprocess import *
from src.utils import *

import wandb

"""
main.py
"""



#torch.set_default_tensor_type('torch.cuda.FloatTensor')

@hydra.main(config_name="config")
def main(conf):
    # seed
    SEED = 42
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)   # :bug it arises an unexpected bug of pytorch 
    pl.seed_everything(SEED)
    np.random.seed(SEED)


    # logging using wandb
    # if use raw wandb
    # wandb.init(config=conf)
    logger = WandbLogger(name=conf.experiment_name, project="Singer Identification")
    logger.log_hyperparams(conf)

    print(hydra.utils.get_original_cwd())
    dir = hydra.utils.get_original_cwd() + "/mlruns"
    if not os.path.exists(dir):
        os.makedirs(dir)

    
    # if use mlflow, remove commentout
    # mlflow.set_tracking_uri(dir)
    # # tracking_uri = mlflow.get_tracking_uri()
    # mlflow.set_experiment("singer_identification")

    # GPU
    use_cuda = torch.cuda.is_available()
    Device = torch.device("cuda" if use_cuda else "cpu")
    print(Device)

    # parameters
    out_model_fn = './model/%s' % (conf.savename)
    if not os.path.exists(out_model_fn):
        os.makedirs(out_model_fn)

    # load dataset
    data_path = conf.data_path
    # meta_path = os.path.join(hydra.utils.get_original_cwd(), data_path,'meta/esc50.csv')
    # df = pd.read_csv(meta_path)

    # soundfile
    # audio_path = os.path.join(hydra.utils.get_original_cwd(), data_path)
    audio_path = data_path
    print("load audio... path: {}".format(data_path))
    train_data = Artist_from_numpy(audio_path, sr=conf.sr, chunk_length=conf.length, set=[0,1,2,3])
    print("valid")
    valid_data = Artist_from_numpy(audio_path, sr=conf.sr, chunk_length=conf.length, set=[4])
    print("test")
    test_data = Artist_from_numpy(audio_path, sr=conf.sr, chunk_length=conf.length, set=[5])
    print("chunked audio: train: {}, valid: {}, test: {}".format(len(train_data), len(valid_data), len(test_data)))

    # classid
    target_class = train_data.get_class_to_id()

    # dataloader
    train_loader = DataLoader(train_data, batch_size=conf.batch_size, shuffle=True,num_workers=2)
    valid_loader = DataLoader(valid_data, batch_size=16, shuffle=False,num_workers=2)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False,num_workers=2)

    # model
    # parameter
    if conf.model == "crnn":
        model = CRNN(conf, classes_num=20)
    elif conf.model == "ssl":
        model = SSLNet(conf,class_num=20,weight_sum=conf.weight_sum)
        lw = model.frontend.get_layer_weights()
        print(type(model))
        print(lw)
    else:
        raise NotImplementedError

    wandb.watch(model, log_freq=100)

    # test
    # test_input = torch.rand((conf.batch_size,1,int(conf.sr*conf.length)))
    # print("test check...\n")
    # with torch.no_grad():
    #     model.eval()
    #     for i, data in enumerate(train_loader):
    #         x, y = data
    #         if i > 0:
    #             break
    #     test_case = x
    #     model = model.cuda()
    #     test_case = test_case.cuda()
    #     out,_ = model(test_case)
    #     print(out.shape)
    #     model.train()


    # train
    model, trainer = train(model, train_loader=train_loader, valid_loader=valid_loader, conf=conf, logger=logger)
    lw = model.frontend.get_layer_weights()
    print(lw)
    with open("layer_weight.txt", 'a') as f:
        lw = model.frontend.get_layer_weights()
        print(lw,file=f)
    # evaluation
    evaluation(model, test_loader, target_class, logger=logger)


if __name__ == "__main__":
    main()