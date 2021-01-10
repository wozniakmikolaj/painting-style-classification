""" main.py """
from PIL import Image
import numpy as np
from src.configs.config import CFG
from src.executors.skip_con_predictor import CNNSkipConnectionPredictor
from src.models.skip_con_model import CNNSkipConnectionModel


def run():
    """Builds model, loads data, trains and evaluates"""
    # model = CNNSkipConnectionModel(CFG)
    # model.load_data()
    # model.build()
    # model.train()
    # model.evaluate()

    # image = np.asarray(Image.open('default_img.jpg')).astype(np.float32)
    # predictor = CNNSkipConnectionPredictor()
    # predicted_class = predictor.infer(image)
    # print(f"Your predicted class: {predicted_class}")


if __name__ == '__main__':
    run()
