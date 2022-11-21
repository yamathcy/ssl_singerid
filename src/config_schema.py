from dataclasses import dataclass
from omegaconf import ListConfig, DictConfig
import mlflow

"""
config_schema.py
コンフィグのスキーマについて書く
"""

# hydraのyamlから操作できるパラメータのスキーマ
# ここにハイパーパラメータを記述することによって，プログラムから参照できるように
@dataclass
class Config:
    experiment_name:str = "ESC-50 audio classification"
    model:str = "cnn"
    epoch:int = 100
    batch_size:int = 16
    savename:str = "exp.pt"


# # MLFlowにパラメータを記録する
# def log_params_from_omegaconf_dict(params):
#     for param_name, element in params.items():
#         _explore_recursive(param_name, element)


# # 読み込みのためのヘルパー関数
# def _explore_recursive(parent_name, element):
#     if isinstance(element, DictConfig):
#         for k, v in element.items():
#             if isinstance(v, DictConfig) or isinstance(v, ListConfig):
#                 _explore_recursive(f'{parent_name}.{k}', v)
#             else:
#                 mlflow.log_param(f'{parent_name}.{k}', v)
#     elif isinstance(element, ListConfig):
#         for i, v in enumerate(element):
#             mlflow.log_param(f'{parent_name}.{i}', v)
