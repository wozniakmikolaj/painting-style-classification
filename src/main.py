""" main.py """

from src.configs.config import CFG
from src.models.skip_con_model import CNNSkipConnectionModel


def run():
    """Builds model, loads data, trains and evaluates"""
    model = CNNSkipConnectionModel(CFG)
    model.load_data()
    model.build()
    model.train()
    model.evaluate()


if __name__ == '__main__':
    run()
