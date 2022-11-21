import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchaudio
import torchmetrics

"""
model.py 
学習に用いるモデルについてを書く
"""

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
        self.train_acc = torchmetrics.Accuracy(num_classes=self.num_classes, average='macro')
        self.val_acc = torchmetrics.Accuracy(num_classes=self.num_classes, average='macro')
        self.test_acc = torchmetrics.Accuracy(num_classes=self.num_classes, average='macro')
        self.test_top3 = torchmetrics.Accuracy(num_classes=self.num_classes, average='macro', top_k=3)
        self.test_f1 = torchmetrics.F1(num_classes=self.num_classes, average='macro')
        # self.confusion = torchmetrics.ConfusionMatrix(num_classes=self.num_classes)

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
        self.log('test_accuracy', self.test_acc(out,y), on_epoch=False, on_step=False)
        self.log('test_f1', self.test_f1(out, y), on_epoch=False, on_step=False)
        self.log('test_top3_accuracy', self.test_top3(out, y), on_epoch=False, on_step=False)
        # self.log('test_accuracy', self.confusion(out, y), on_epoch=False, on_step=False)



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
        self.train_acc = torchmetrics.Accuracy(num_classes=self.num_classes, average='macro')
        self.val_acc = torchmetrics.Accuracy(num_classes=self.num_classes, average='macro')
        self.test_acc = torchmetrics.Accuracy(num_classes=self.num_classes, average='macro')
        self.test_top3 = torchmetrics.Accuracy(num_classes=self.num_classes, average='macro', top_k=3)
        self.test_f1 = torchmetrics.F1(num_classes=self.num_classes, average='macro')
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