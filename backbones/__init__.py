from .pyramidnet import pyramidnet272, pyramidnet164
from .wrn import wrn_40_1, wrn_40_2, wrn_16_2, wrn_40_4, wrn_16_4, wrn_16_1, wrn_40_10

model_dict = {
    "pyramidnet272": pyramidnet272,
    "pyramidnet164": pyramidnet164,
    "wrn_40_1": wrn_40_1,
    "wrn_40_2": wrn_40_2,
    "wrn_40_4": wrn_40_4,
    "wrn_16_1": wrn_16_1,
    "wrn_16_2": wrn_16_2,
    "wrn_16_4": wrn_16_4,
    "wrn_40_10": wrn_40_10
}