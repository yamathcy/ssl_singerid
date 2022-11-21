import argparse
import os,sys
import datetime
import torch
import requests
import hydra
from hydra.core.config_store import ConfigStore
import mlflow
import omegaconf
from pytorch_lightning.loggers import MLFlowLogger
from src.train import *
from src.model import *
from src.eval import *
from src.dataio import *
from src.preprocess import *
from src.utils import *
from src.config_schema import Config, log_params_from_omegaconf_dict


"""
main.py
メインの根幹部分．ここにロジックをまとめる．
"""
cs = ConfigStore.instance()
cs.store(name="config", node=Config)
#torch.set_default_tensor_type('torch.cuda.FloatTensor')

@hydra.main(config_name="config")
def main(conf: Config):
    print(hydra.utils.get_original_cwd())
    dir = hydra.utils.get_original_cwd() + "/mlruns"
    if not os.path.exists(dir):
        os.makedirs(dir)
    mlflow.set_tracking_uri(dir)
    tracking_uri = mlflow.get_tracking_uri()

    mlflow.set_experiment(conf.experiment_name)

    # コマンドライン引数による実験条件の指定を可能にする (argparser用)
    # parser = argparse.ArgumentParser(description="experiment")
    # parser.add_argument(
    #     "--batch_size", type=int, default=32,
    #     help="Batch size during training Default: 32"
    # )
    # parser.add_argument(
    #     "--model_path", type=str, default="models",
    #     help="the model pt file path Default: models"
    # )
    # parser.add_argument(
    #     "--savename", type=str, default="run",
    #     help="the model pt file name Default: run"
    # )
    # args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # out_model_fn = './model/%s/' % (args.savename)
    out_model_fn = './model/%s' % (conf.savename)
    if not os.path.exists(out_model_fn):
        os.makedirs(out_model_fn)

    # データセットの読み込み
    data_path = 'data/ESC-50-master'
    meta_path = os.path.join(hydra.utils.get_original_cwd(), data_path,'meta/esc50.csv')
    df = pd.read_csv(meta_path)

    # ラベルをsplit
    train_label= df[df['fold']<=3]
    valid_label = df[df['fold']==4]
    test_label = df[df['fold']==5]

    # 音ファイルの読み込み
    audio_path = os.path.join(hydra.utils.get_original_cwd(), data_path, "audio")
    train_data = ESC50(label_df=train_label, base=audio_path)
    valid_data = ESC50(label_df=valid_label, base=audio_path)
    test_data = ESC50(label_df=test_label, base=audio_path)

    # 各データローダーの用意
    train_loader = DataLoader(train_data, batch_size=conf.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    # モデル
    # パラメータによって条件を分岐
    if conf.model == "cnn":
        model = SimpleCNNModel()
    elif conf.model == "resnet":
        model = ResNet()
    else:
        raise NotImplementedError

    SEED = 42
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    pl.seed_everything(SEED)
    np.random.seed(SEED)

    # 学習
    mlf_logger = MLFlowLogger(experiment_name=conf.experiment_name, tracking_uri=tracking_uri)
    model, trainer = train(model, train_loader, valid_loader, max_epochs=conf.epoch, logger=mlf_logger)

    # 評価
    evaluation(model, trainer, test_loader)

    # パラメータのロギング
    mlflow.log_params(conf)
    # モデルの保存
    mlf_logger.experiment.log_artifact(mlf_logger.run_id, out_model_fn)

if __name__ == "__main__":
    main()