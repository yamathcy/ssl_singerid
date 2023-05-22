import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchaudio
# import torchmetrics
from torchmetrics import Accuracy, F1Score, ConfusionMatrix
from transformers import AutoModel
import numpy as np
from abc import abstractmethod
# from torchsummary import summary

"""
model.py 
学習に用いるモデルについてを書く
"""

class BaseModel(pl.LightningModule):
    """
    Base class for all models
    """

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model # prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])

        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def get_feature(self, z):
        _, feature = self.forward(z)
        if type(feature) == tuple:
            feature = feature[0]
        return feature

    def predict(self, x):
        """

        :param x: input tensor (torch.Tensor)
        :return: single output of model (numpy.array)
        """

        self.eval()
        out, _ = self.forward(x)
        out = torch.argmax(out, dim=1)
        out = out.cpu().detach().numpy().copy()
        # out = np.squeeze(out)
        return out

    def predict_proba(self, x):
        """

        :param x: input tensor (torch.Tensor)
        :return: single output of model (numpy.array)
        """
        self.eval()
        out, _ = self.forward(x)
        out = F.softmax(out, dim=1)  # assuming logits has the shape [batch_size, nb_classes]
        out = out.cpu().detach().numpy().copy()
        out = np.squeeze(out)
        return out

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
        # print(x.shape)
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

    def predict(self, x):
        self.eval()
        out, _ = self.forward(x)
        out = torch.argmax(out, dim=1)
        out = out.cpu().detach().numpy().copy()
        # out = np.squeeze(out)
        return out

    def predict_proba(self, x):
        """

        :param x: input tensor (torch.Tensor)
        :return: single output of model (numpy.array)
        """
        self.eval()
        out, _ = self.forward(x)
        out = torch.softmax(out, dim=1)  # assuming logits has the shape [batch_size, nb_classes]
        out = out.cpu().detach().numpy().copy()
        out = np.squeeze(out)
        return out

class Backend(nn.Module):
    def __init__(self, class_size, encoder_size=12) -> None:
        super().__init__()
        assert encoder_size == 12 or encoder_size == 24
        if encoder_size == 12:
            self.layer_weights = torch.nn.parameter.Parameter(data=torch.ones(13), requires_grad=True)
            feature_dim = 768
        elif encoder_size == 24:
            self.layer_weights = torch.nn.parameter.Parameter(data=torch.ones(25), requires_grad=True)
            feature_dim = 1024
        else:
            raise NotImplementedError
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

class SSLNet(BaseModel):
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
        self.frontend = AutoModel.from_pretrained(url, trust_remote_code=True,cache_dir='./hfmodels')
        
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
        # class_weights = [float(x) for x in weights.values()]
        # self.class_weights = torch.from_numpy(np.array(class_weights)).float()

    def forward(self, x):
        # print(x.shape)
        x = x.squeeze(dim=1)
        # print(x.shape, type(x))
        # x = x.to(DEVICE) # FIXME: Unknown behaviour on return to cpu by feature extractor
        x = self.frontend(x, output_hidden_states=True, return_dict=None, output_attentions=None)
        h = x["hidden_states"]
        h = torch.stack(h, dim=3)
        # pad_width = (0, 0, 0, 0, 0, 1)
        # h = F.pad(h, pad_width, mode='reflect')
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
        # print(x.shape)
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


class SSLNet_RAW(nn.Module):
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
        self.frontend = AutoModel.from_pretrained(url, trust_remote_code=True, cache_dir='./hfmodels')
        
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
        # class_weights = [float(x) for x in weights.values()]
        # self.class_weights = torch.from_numpy(np.array(class_weights)).float()

    def forward(self, x):
        # print(x.shape)
        x = x.squeeze(dim=1)
        # print(x.shape, type(x))
        # x = x.to(DEVICE) # FIXME: Unknown behaviour on return to cpu by feature extractor
        x = self.frontend(x, output_hidden_states=True, return_dict=None, output_attentions=None)
        h = x["hidden_states"]
        h = torch.stack(h, dim=3)
        # pad_width = (0, 0, 0, 0, 0, 1)
        # h = F.pad(h, pad_width, mode='reflect')
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