import torch
from torchvision.models.feature_extraction import create_feature_extractor
from pytorch_lightning import Callback,Trainer,LightningModule
from pytorch_lightning.loggers import WandbLogger
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, top_k_accuracy_score, confusion_matrix, classification_report, f1_score
from umap import UMAP
import seaborn as sns
import matplotlib.pylab as plt
import pandas as pd
from tqdm import tqdm
# import mlflow
"""
eval.py
学習に用いるモデルについてを書く
"""

def evaluation(model, test_loader, target_class, logger:WandbLogger):
    """
    モデルとtrainer, test用のdataloaderを用いてテストデータでの評価を行う
    :param model:
    :param trainer:
    :param test_loader:
    :return:
    """
    feature_vecs = []
    pred = []
    pre_prob = []
    label = []
    print("visualize feature dump...")
    for sig, la in tqdm(test_loader):
        sig = sig.to("cuda")

        la = int(la)
        model = model.to("cuda")
        _ , feature = model(sig)
        feature = feature.detach().cpu().numpy().copy()
        out = model.predict(sig)
        prob= model.predict_proba(sig)
        pred.append(out)
        pre_prob.append(prob)
        label.append(la)
        feature_vecs.append(feature)
    pred = np.array(pred)
    pre_prob = np.array(pre_prob)
    label = np.array(label)
    target_class_inv = {v: k for k, v in target_class.items()}
    embed_visualize((feature_vecs, label),target_class_inv, logger)
    accuracy = accuracy_score(y_true=label, y_pred=pred)
    balanced = balanced_accuracy_score(y_true=label, y_pred=pred)
    top_2 = top_k_accuracy_score(k=2,y_score=pre_prob, y_true=label)
    top_3 = top_k_accuracy_score(k=3,y_score=pre_prob, y_true=label)
    macrof1 = f1_score(y_true=label,y_pred=pred, average='macro')
    report = classification_report(y_true=label, y_pred=pred, target_names=list(target_class.keys()))
    report_dict = classification_report(y_true=label, y_pred=pred, target_names=list(target_class.keys()), output_dict=True)
    print(report)

    cf_data_d=confusion_matrix(label, pred)
    print(cf_data_d)
    df_cmx = pd.DataFrame(cf_data_d, index=target_class.keys(), columns=target_class.keys())
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cmx, annot=True, fmt="g", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matrix")
    print("---Accuracy Report---")
    print("Overall accracy:{:.3f}".format(accuracy))
    print("Overall balanced accracy:{:.3f}".format(balanced))
    print("Top-2:{:.3f}".format(top_2))
    print("Top-3:{:.3f}".format(top_3))
    print("f1-score: {:.3f}".format(macrof1))
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    # mlflow.log_artifact(key='confusion_matrix', images=['confusion_matrix.png'])

    with open("result.txt", 'a') as f:
        print("---Accuracy Report---", file=f)
        print("Overall accracy:{:.3f}".format(accuracy) ,file=f)
        print("Overall balanced accracy:{:.3f}".format(balanced), file=f)
        print("Top-2:{:.3f}".format(top_2), file=f)
        print("Top-3:{:.3f}".format(top_3),file=f)
        print("f1-score: {:.3f}".format(macrof1))
        print(report, file=f)
    # logger.use_artifact("result.txt", artifact_type='text')
    # mlflow.log_metrics({"acc":accuracy, "bacc":balanced, "top2":top_2, "top3":top_3, "f1":macrof1})
    logger.log({"acc":accuracy, "bacc":balanced, "top2":top_2, "top3":top_3, "f1":macrof1})
    logger.use_artifact("confusion_matrix.png", artifact_type='image')
    logger.use_artifact("embedding.png", artifact_type='image')
    for label_name, val in zip(report_dict.keys(), report_dict.values()):
        try:
            logger.log({"class_f1_" + label_name:val['f1-score']})
            # mlflow.log_metrics({"class_f1_" + label_name:val['f1-score']})
        except:
            pass
    try:
        lw = model.get_layer_weight()
        lw.squeeze()
        print(lw)
        plt.plot(lw)
        plt.tight_layout()
        plt.savefig("layer_weight.png")
        logger.use_artifact("layer_weight.png", artifact_type='image')

    except:
        pass

    return accuracy, balanced, top_2, top_3, df_cmx, report_dict



def single_test(model:torch.nn.Module, data):
    """
    1つのデータをモデルに入力して推定結果を得る
    :param model:
    :param data:
    :return:
    """
    model.eval()
    with torch.no_grad():
        output = model(data)
        # todo: outputをtensorではなくidにする？
    return output


def embed_visualize(dataset, target_class_inv, logger=None):
    # dataset must be set of (extracted_feature, label)
    # visualize tsne deep
    sns.set()
    squeezed = []
    for train_vec in dataset[0]:
        train_vec = np.squeeze(train_vec)
        squeezed.append(train_vec)
    mapper = UMAP(random_state=1)
    emb = mapper.fit_transform(squeezed)
    np_label = np.array([dataset[1]])

    dat = np.hstack((emb, np_label.T))
    df = pd.DataFrame(dat,columns=["x","y","label"])
    df = df.astype({'label': int})
    df["label"] = df["label"].replace(target_class_inv)
    plt.figure(figsize=(7,5))
    sp=sns.scatterplot(data=df,palette="tab20",x="x",y="y",hue="label", alpha=.5, linewidth=0.1)
    plt.title("Embedding")
    plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
    plt.tight_layout()
    fig=sp.get_figure()
    fig.savefig("embedding.png")
    if logger:
        logger.log_image(key='embedding', images=['embedding.png'])
    else:
        # mlflow.log_artifact('embedding.png')
        pass

