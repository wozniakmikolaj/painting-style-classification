""" main.py """

from src.configs.config import CFG
from src.model.style_model import CNNSkipConnection


def run():
    """Builds model, loads data, trains and evaluates"""
    model = CNNSkipConnection(CFG)
    model.load_data()
    model.build()
    model.train()
    model.evaluate()


if __name__ == '__main__':
    run()
