"""Model config in json format"""

import os
import pathlib

# print(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data', 'processed', 'data_to_tf_data.npy'))
# print(pathlib.Path(__file__).parent.parent)

CFG = {
    "paths": {
        "path_raw": os.path.join(os.path.dirname(os.getcwd()), 'data', 'raw'),
        "path_raw_csv_file": os.path.join(os.path.dirname(os.getcwd()), 'data', 'raw', 'train_info.csv'),
        "path_processed": os.path.join(os.path.dirname(os.getcwd()), 'data', 'processed'),
        "path_processed_data": os.path.join(os.path.dirname(os.getcwd()), 'data', 'processed', 'data_to_tf_data.npy'),
        "path_processed_labels": os.path.join(os.path.dirname(os.getcwd()), 'data', 'processed', 'labels_to_tf_data.npy'),
        "path_model_saved": os.path.join(os.path.dirname(os.getcwd()), 'models'),
        "path_model_plot": os.path.join(os.path.dirname(os.getcwd()), 'figures'),
        "path_s3": ""
    },
    "data": {
        "images_per_class": 1100,
        "image_size": 200,
        "num_channels": 1,
        "train_split": 0.7,
        "test_split": 0.15,
        "validation_split": 0.15
    },
    "train": {
        "learning_rate": 1e-4,
        "batch_size": 64,
        "buffer_size": 1000,
        "epochs": 40,
        "optimizer": {
            "type": "adam"
        },
        "metrics": ["accuracy"]
    },
    "model": {
        "model_name": "cnn_skip_connection_model",
        "input": [200, 200, 1],
        "output": 10
    }
}
