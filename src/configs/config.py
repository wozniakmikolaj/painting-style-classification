"""Model config in json format"""

import os

CFG = {
    "paths": {
        "path_raw": os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                                 'data', 'raw'),
        "path_raw_csv_file": os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                                          'data', 'raw', 'train_info.csv'),
        "path_processed": os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                                       'data', 'processed'),
        "path_processed_data": os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                                            'data', 'processed', 'data_to_tf_data.npy'),
        "path_processed_labels": os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                                              'data', 'processed', 'labels_to_tf_data.npy'),
        "path_model_plot": os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                                        'figures'),
        "path_s3": ""
    },
    "data": {
        "image_size": 200,
        "num_channels": 1,
    },
    "train": {
        "learning_rate": 1e-4,
        "batch_size": 50,
        "buffer_size": 1000,
        "epochs": 30,
        "optimizer": {
            "type": "adam"
        },
        "metrics": ["accuracy"]
    },
    "model": {
        "model_name": "cnn_skip_connection_model",
        "input": [200, 200, 1],
        "output": 11
    }
}
