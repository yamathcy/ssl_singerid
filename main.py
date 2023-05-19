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
メインの根幹部分．ここにロジックをまとめる．
"""

'''+++'''

#torch.set_default_tensor_type('torch.cuda.FloatTensor')

'''+++'''
@hydra.main(config_name="config")
def main(conf:omegaconf.DictConfig):
    # パラメータのロギング
    # wandbの準備
    # wandb.init(config=conf)
    logger = WandbLogger(name=conf.experiment_name, project="Singer Identification")
    logger.log_hyperparams(conf)

    # ランダムのシードを決定
    SEED = 42
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    pl.seed_everything(SEED)
    np.random.seed(SEED)

    '''+++'''
    # トラッキングを行う場所をチェックし，ログを収納するディレクトリを指定
    print(hydra.utils.get_original_cwd())
    dir = hydra.utils.get_original_cwd() + "/mlruns"
    if not os.path.exists(dir):
        os.makedirs(dir)

    '''+++'''
    
    # mlflowの準備
    # mlflow.set_tracking_uri(dir)
    # tracking_uri = mlflow.get_tracking_uri()
    # mlflow.set_experiment(conf.experiment_name)

    # GPUの準備
    use_cuda = torch.cuda.is_available()
    Device = torch.device("cuda" if use_cuda else "cpu")

    # 学習したモデルのパラメータ
    out_model_fn = './model/%s' % (conf.savename)
    if not os.path.exists(out_model_fn):
        os.makedirs(out_model_fn)

    # データセットの読み込み
    data_path = conf.data_path
    # meta_path = os.path.join(hydra.utils.get_original_cwd(), data_path,'meta/esc50.csv')
    # df = pd.read_csv(meta_path)

    # 音ファイルの読み込み
    audio_path = os.path.join(hydra.utils.get_original_cwd(), data_path)
    print("load audio... path: {}".format(audio_path))
    train_data = Artist(audio_path, sr=conf.sr, chunk_length=conf.length, set=[0,1,2,3])
    print("valid")
    valid_data = Artist(audio_path, sr=conf.sr, chunk_length=conf.length, set=[4])
    print("test")
    test_data = Artist(audio_path, sr=conf.sr, chunk_length=conf.length, set=[5])
    print("chunked audio: train: {}, valid: {}, test: {}".format(len(train_data), len(valid_data), len(test_data)))

    # classidを得る
    target_class = train_data.get_class_to_id()

    # 各データローダーの用意
    train_loader = DataLoader(train_data, batch_size=conf.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # モデル
    # パラメータによって条件を分岐
    if conf.model == "crnn":
        model = CRNN(conf, classes_num=20)
    elif conf.model == "ssl":
        model = SSLNet(conf,weights=None,url=conf.url,class_num=20)
    else:
        raise NotImplementedError
    # Magic
    wandb.watch(model, log_freq=100) 

    '''+++'''
    # 学習
    model, trainer = train(model, train_loader=train_loader, valid_loader=valid_loader, max_epochs=conf.epoch, logger=logger)

    # 評価
    evaluation(model, logger, test_loader, target_class)

    '''+++'''


if __name__ == "__main__":
    main()