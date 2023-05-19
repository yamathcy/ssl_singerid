import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchaudio
# import torchmetrics
from torchmetrics import Accuracy, F1Score, ConfusionMatrix
from transformers import AutoModel
import numpy as np

"""
model.py 
学習に用いるモデルについてを書く
"""

class CRNN(pl.LightningModule):
    """
    Baseline model 
    borrowed from 
    https://github.com/bill317996/Singer-identification-in-artist20/blob/master
    """
    def __init__(self, conf, classes_num):
        super().__init__()
        self.lr = conf.lr
        self.num_classes = classes_num
        self.elu = nn.ELU()
        self.softmax = nn.Softmax(dim=1)

        self.audio = torchaudio.transforms.MelSpectrogram(sample_rate=conf.sr,
                                                          n_mels=128,
                                                        n_fft=2048,
                                                        hop_length=512)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

        self.Conv1 = nn.Conv2d(1, 64, (3,3))
        self.Bn1 = nn.BatchNorm2d(64)
        self.mp1 = nn.MaxPool2d((2,2), stride=(2,2))
        self.drop1 = nn.Dropout2d(p=0.1)

        self.Conv2 = nn.Conv2d(64, 128, (3,3))
        self.Bn2 = nn.BatchNorm2d(128)
        self.mp2 = nn.MaxPool2d((4,2), stride=(2,4))
        self.drop2 = nn.Dropout2d(p=0.1)

        self.Conv3 = nn.Conv2d(128, 128, (3,3))
        self.Bn3 = nn.BatchNorm2d(128)
        self.mp3 = nn.MaxPool2d((4,2), stride=(4,4))
        self.drop3 = nn.Dropout2d(p=0.1)

        self.Conv4 = nn.Conv2d(128, 128, (3,3))
        self.Bn4 = nn.BatchNorm2d(128)
        self.mp4 = nn.MaxPool2d((4,2), stride=(4,4))
        self.drop4 = nn.Dropout2d(p=0.1)

        self.gru1 = nn.GRU(128, 32, num_layers=1, batch_first=True)
        self.gru2 = nn.GRU(32, 32, num_layers=1, batch_first=True)
        self.drop5 = nn.Dropout(p=0.3)

        self.linear1 = nn.Linear(32, classes_num)

        self.train_acc = Accuracy(num_classes=self.num_classes, average='macro', task='multiclass')
        self.val_acc = Accuracy(num_classes=self.num_classes, average='macro', task='multiclass')
        self.test_acc = Accuracy(num_classes=self.num_classes, average='macro', task='multiclass')
        self.test_top2 = Accuracy(num_classes=self.num_classes, average='macro', top_k=2, task='multiclass')
        self.test_top3 = Accuracy(num_classes=self.num_classes, average='macro', top_k=3, task='multiclass')
        self.test_f1 = F1Score(num_classes=self.num_classes, average='macro', task='multiclass')
        self.confusion = ConfusionMatrix(num_classes=self.num_classes, task='multiclass')

    def forward(self, x):
        x = self.audio(x)
        x = self.amplitude_to_db(x)
        # x = torch.permute(x, (0,1,3,2))

        x = self.drop1(self.mp1(self.Bn1(self.elu(self.Conv1(x)))))

        x = self.drop2(self.mp2(self.Bn2(self.elu(self.Conv2(x)))))

        x = self.drop3(self.mp3(self.Bn3(self.elu(self.Conv3(x)))))

        x = self.drop4(self.mp4(self.Bn4(self.elu(self.Conv4(x)))))

        x = x.transpose(1, 3)
        x = torch.reshape(x, (x.size(0),x.size(1),-1))

        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x = self.drop5(x)

        x = torch.reshape(x, (x.size(0), -1))
        emb = x
        x = self.linear1(x)
        return x, emb
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        out,_ = self(x)
        loss = F.cross_entropy(out, y)
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.log('train_acc', self.train_acc(out, y), on_step=False, on_epoch=True)
        return loss
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        out,_ = self(x)
        loss = F.cross_entropy(out, y)
        self.log('val_loss', loss, on_epoch=True, on_step=False)
        self.log('val_acc', self.val_acc(out, y), on_step=False, on_epoch=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        out,_ = self(x)
        self.log('test_accuracy', self.test_acc(out,y), on_epoch=True, on_step=False)
        self.log('test_f1', self.test_f1(out, y), on_epoch=True, on_step=False)
        self.log('test_top2_accuracy', self.test_top2(out, y), on_epoch=True, on_step=False)
        self.log('test_top3_accuracy', self.test_top3(out, y), on_epoch=True, on_step=False)
        self.log('test_confusion', self.confusion(out, y), on_epoch=False, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer



# 畳みこみブロックの定義
class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, shape=(3, 3), pooling=(2, 2), drop_out_rate=0, padding='same'):
        super().__init__()

        self.conv = nn.Conv2d(input_channels, output_channels, shape,
                              padding=padding)

        self.normalize = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.pool_size = pooling
        if pooling is None:
            self.maxpool = nn.Identity()
        else:
            self.maxpool = nn.MaxPool2d(pooling)
        self.dropout = nn.Dropout(drop_out_rate)
        self._init_params()

    def _init_params(self):
        # 重みの初期化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 畳み込み
        x = self.conv(x)
        # バッチ正規化
        x = self.normalize(x)
        # 活性化関数
        x = self.relu(x)
        # プーリング
        x = self.maxpool(x)
        # ドロップアウト
        out = self.dropout(x)
        return out

# ResNetブロック
class Res_2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=2):
        super().__init__()
        # 畳み込み
        self.conv_1 = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
        self.bn_1 = nn.BatchNorm2d(output_channels)
        self.conv_2 = nn.Conv2d(output_channels, output_channels, shape, padding=shape//2)
        self.bn_2 = nn.BatchNorm2d(output_channels)

        # Residual connection 
        self.diff = False
        if (stride != 1) or (input_channels != output_channels):
            self.conv_3 = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
            self.bn_3 = nn.BatchNorm2d(output_channels)
            self.diff = True
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))
        if self.diff:
            x = self.bn_3(self.conv_3(x))
        out = x + out
        out = self.relu(out)
        return out

# 8層のCNNモデル
class SimpleCNNModel(pl.LightningModule):
    def __init__(self, num_channels=32,
                 sample_rate=44100,
                 n_fft=2048,
                 hop_length=512,
                 f_min=20.0,
                 f_max=8000.0,
                 num_mels=128,
                 num_classes=50,
                 lr=1e-4):
        super().__init__()
        self.lr=lr
        self.num_classes = num_classes

        # メルスペクトログラム
        self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                            n_fft=n_fft,
                                                            hop_length=hop_length,
                                                            f_min=f_min,
                                                            f_max=f_max,
                                                            n_mels=num_mels)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

        # 畳み込み層
        self.layer1 = ConvBlock(1, num_channels)
        self.layer2 = ConvBlock(num_channels, num_channels)
        self.layer3 = ConvBlock(num_channels, num_channels * 2)
        self.layer4 = ConvBlock(num_channels * 2, num_channels * 2)
        self.layer5 = ConvBlock(num_channels * 2, num_channels * 4)
        self.layer6 = ConvBlock(num_channels * 4, num_channels * 4)
        self.layer7 = ConvBlock(num_channels * 4, num_channels * 8)
        self.layer8 = ConvBlock(num_channels * 8, num_channels * 8, pooling=None)

        # 全結合層
        self.dense1 = nn.Linear(num_channels * 8, num_channels * 2)
        self.dense2 = nn.Linear(num_channels * 2, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

        # 評価メトリクス
        self.train_acc = Accuracy(num_classes=self.num_classes, average='macro', task='multiclass')
        self.val_acc = Accuracy(num_classes=self.num_classes, average='macro' ,task='multiclass')
        self.test_acc = Accuracy(num_classes=self.num_classes, average='macro' ,task='multiclass')
        self.test_top3 = Accuracy(num_classes=self.num_classes, average='macro', top_k=3 ,task='multiclass')
        self.test_f1 = F1Score(num_classes=self.num_classes, average='macro' ,task='multiclass')
        self.confusion = ConfusionMatrix(num_classes=self.num_classes, task='multiclass')

    def forward(self, x):
        # 入力のスペクトログラム
        out = self.melspec(x)
        out = self.amplitude_to_db(out)

        # 畳み込み層
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        # 全結合層をreshapeするため，Global Average Poolingというテクニックを使う (batch_size, num_channels, 1, 1) -> (batch_size, num_channels)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.squeeze(out)

        # dense layers
        out = self.dense1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.dense2(out)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        out = self(x)
        loss = F.cross_entropy(out, y)
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.log('train_acc', self.train_acc(out, y), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        out = self(x)
        loss = F.cross_entropy(out, y)
        self.log('val_loss', loss, on_epoch=True, on_step=False)
        self.log('val_acc', self.val_acc(out, y), on_step=False, on_epoch=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        out = self(x)
        self.log('test_accuracy', self.test_acc(out,y), on_epoch=True, on_step=False)
        self.log('test_f1', self.test_f1(out, y), on_epoch=True, on_step=False)
        self.log('test_top3_accuracy', self.test_top3(out, y), on_epoch=True, on_step=False)
        # self.log('test_accuracy', self.confusion(out, y), on_epoch=False, on_step=False)


# 7層のResnetモデル
class ResNet(pl.LightningModule):
    def __init__(self, num_channels=32,
                 sample_rate=44100,
                 n_fft=2048,
                 hop_length=512,
                 f_min=20.0,
                 f_max=8000.0,
                 num_mels=128,
                 num_classes=50,
                 lr=1e-4):
        super().__init__()
        self.lr=lr
        self.num_classes = num_classes

        # メルスペクトログラム
        self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                            n_fft=n_fft,
                                                            hop_length=hop_length,
                                                            f_min=f_min,
                                                            f_max=f_max,
                                                            n_mels=num_mels)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        self.spec_bn = nn.BatchNorm2d(1)

        # 畳み込み層
        
        self.layer1 = Res_2d(1, num_channels, stride=2)
        self.layer2 = Res_2d(num_channels, num_channels)
        self.layer3 = Res_2d(num_channels, num_channels*2)
        self.layer4 = Res_2d(num_channels*2, num_channels*2)
        self.layer5 = Res_2d(num_channels*2, num_channels*2)
        self.layer6 = Res_2d(num_channels*2, num_channels*2)
        self.layer7 = Res_2d(num_channels*2, num_channels*4)

        # 全結合層
        self.dense1 = nn.Linear(num_channels * 4, num_channels * 2)
        self.dense2 = nn.Linear(num_channels * 2, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

        # 評価メトリクス
        self.train_acc = Accuracy(num_classes=self.num_classes, average='macro', task='multiclass')
        self.val_acc = Accuracy(num_classes=self.num_classes, average='macro', task='multiclass')
        self.test_acc = Accuracy(num_classes=self.num_classes, average='macro', task='multiclass')
        self.test_top2 = Accuracy(num_classes=self.num_classes, average='macro', top_k=2, task='multiclass')
        self.test_top3 = Accuracy(num_classes=self.num_classes, average='macro', top_k=3, task='multiclass')
        self.test_f1 = F1Score(num_classes=self.num_classes, average='macro', task='multiclass')
        self.confusion = ConfusionMatrix(num_classes=self.num_classes, task='multiclass')
        # self.confusion = torchmetrics.ConfusionMatrix(num_classes=self.num_classes)

    def forward(self, x):
        # 入力のスペクトログラム
        out = self.melspec(x)
        out = self.amplitude_to_db(out)
        out = self.spec_bn(out)

        # 畳み込み層
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)

        # 全結合層をreshapeするため，Global Average Poolingというテクニックを使う (batch_size, num_channels, 1, 1) -> (batch_size, num_channels)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.squeeze(out)

        # dense layers
        out = self.dense1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.dense2(out)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        out = self(x)
        loss = F.cross_entropy(out, y)
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.log('train_acc', self.train_acc(out, y), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        out = self(x)
        loss = F.cross_entropy(out, y)
        self.log('val_loss', loss, on_epoch=True, on_step=False)
        self.log('val_acc', self.val_acc(out, y), on_step=False, on_epoch=True)
        return loss

    # 評価をここに書く
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        out = self(x)
        self.log('test_accuracy', self.test_acc(out,y), on_epoch=True, on_step=False)
        self.log('test_f1', self.test_f1(out, y), on_epoch=True, on_step=False)
        self.log('test_top3_accuracy', self.test_top3(out, y), on_epoch=True, on_step=False)
        # self.log('test_confusion', self.confusion(out, y), on_epoch=False, on_step=False)


# 7層のResnetモデル
class AudioDNN(pl.LightningModule):
    def __init__(self, conf, num_classes):
        super().__init__()
        self.lr=conf.lr
        self.num_classes = num_classes
        self.model = CRNN(conf.sr, num_classes)

        # 評価メトリクス
        self.train_acc = Accuracy(num_classes=self.num_classes, average='macro', task='multiclass')
        self.val_acc = Accuracy(num_classes=self.num_classes, average='macro', task='multiclass')
        self.test_acc = Accuracy(num_classes=self.num_classes, average='macro', task='multiclass')
        self.test_top2 = Accuracy(num_classes=self.num_classes, average='macro', top_k=2, task='multiclass')
        self.test_top3 = Accuracy(num_classes=self.num_classes, average='macro', top_k=3, task='multiclass')
        self.test_f1 = F1Score(num_classes=self.num_classes, average='macro', task='multiclass')
        self.confusion = ConfusionMatrix(num_classes=self.num_classes, task='multiclass')

    def forward(self, x):
        out, feature = self.model(x)
        return out, feature

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        out,_ = self(x)
        loss = F.cross_entropy(out, y)
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.log('train_acc', self.train_acc(out, y), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        out,_ = self(x)
        loss = F.cross_entropy(out, y)
        self.log('val_loss', loss, on_epoch=True, on_step=False)
        self.log('val_acc', self.val_acc(out, y), on_step=False, on_epoch=True)
        return loss

    # 評価をここに書く
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        out,_ = self(x)
        self.log('test_accuracy', self.test_acc(out,y), on_epoch=True, on_step=False)
        self.log('test_f1', self.test_f1(out, y), on_epoch=True, on_step=False)
        self.log('test_top2_accuracy', self.test_top2(out, y), on_epoch=True, on_step=False)
        self.log('test_top3_accuracy', self.test_top3(out, y), on_epoch=True, on_step=False)
        self.log('test_confusion', self.confusion(out, y), on_epoch=False, on_step=False)


class Backend(nn.Module):
    def __init__(self, class_size, encoder_size=12) -> None:
        super().__init__()
        assert encoder_size == 12 or encoder_size == 24
        if encoder_size == 12:
            self.layer_weights = torch.nn.parameter.Parameter(data=torch.ones(13), requires_grad=True)
            feature_dim=768
        elif encoder_size == 24:
            self.layer_weights = torch.nn.parameter.Parameter(data=torch.ones(25), requires_grad=True)
            feature_dim=1024
        self.proj  = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(0.5)
        self.classifier=nn.Linear(feature_dim, class_size)
        
    def forward(self,x):
        weights = torch.sigmoid(self.layer_weights)
        # embeddings = embeddings.transpose(1,3)  # (B, Emb, Time, Ch) * (Ch, 1)
        x = torch.matmul(x, weights)
        x = self.proj(x)
        x = x.mean(1, False)
        feature = x
        x = self.dropout(x)
        x = self.classifier(x)
        return x, feature
    

class SSLNet(pl.LightningModule):
    def __init__(self,
                 conf,
                 weights:dict or list=None,
                 url="microsoft/wavlm-base-plus",
                 class_num=10,
                 freeze_all=False
                 ):
        super().__init__()
        self.num_classes = class_num
        self.lr = conf.lr
        encode_size = 24 if "large" in url else 12
        # if param.sr != 16000:
        #     self.resampler = torchaudio.transforms.Resample(orig_freq=param.sr, new_freq=16000)
        # else:
        #     self.resampler = nn.Identity()
        self.frontend = AutoModel.from_pretrained(url, trust_remote_code=True)
        
        if freeze_all:
            for p in self.frontend.parameters():
                p.requires_grad = False
        else:
            self.frontend.feature_extractor._freeze_parameters()
        self.backend = Backend(class_num, encoder_size=encode_size)

        self.train_acc = Accuracy(num_classes=self.num_classes, average='macro', task='multiclass')
        self.val_acc = Accuracy(num_classes=self.num_classes, average='macro', task='multiclass')
        self.test_acc = Accuracy(num_classes=self.num_classes, average='macro', task='multiclass')
        self.test_top2 = Accuracy(num_classes=self.num_classes, average='macro', top_k=2, task='multiclass')
        self.test_top3 = Accuracy(num_classes=self.num_classes, average='macro', top_k=3, task='multiclass')
        self.test_f1 = F1Score(num_classes=self.num_classes, average='macro', task='multiclass')
        self.confusion = ConfusionMatrix(num_classes=self.num_classes, task='multiclass')
        class_weights = [float(x) for x in weights.values()]
        self.class_weights = torch.from_numpy(np.array(class_weights)).float()


    def forward(self, x):
        x = x.squeeze(dim=1)
        # x = x.to(DEVICE) # FIXME: Unknown behaviour on return to cpu by feature extractor
        x = self.frontend(x, output_hidden_states=True, return_dict=None, output_attentions=None)
        h = x["hidden_states"]
        h = torch.stack(h, dim=3)
        pad_width = (0, 0, 0, 0, 0, 1)
        h = F.pad(h, pad_width, mode='reflect')
        # print(h.shape)
        out, feature = self.backend(h)
        return out, feature

    def configure_optimizers(self, lr=1e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def get_layer_weight(self):
        lw = torch.sigmoid(self.backend.layer_weights)
        lw.detach().cpu()
        return lw
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        out,_ = self(x)
        loss = F.cross_entropy(out, y)
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.log('train_acc', self.train_acc(out, y), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        out,_ = self(x)
        loss = F.cross_entropy(out, y)
        self.log('val_loss', loss, on_epoch=True, on_step=False)
        self.log('val_acc', self.val_acc(out, y), on_step=False, on_epoch=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        out,_ = self(x)
        self.log('test_accuracy', self.test_acc(out,y), on_epoch=True, on_step=False)
        self.log('test_f1', self.test_f1(out, y), on_epoch=True, on_step=False)
        self.log('test_top2_accuracy', self.test_top2(out, y), on_epoch=True, on_step=False)
        self.log('test_top3_accuracy', self.test_top3(out, y), on_epoch=True, on_step=False)
        self.log('test_confusion', self.confusion(out, y), on_epoch=False, on_step=False)


