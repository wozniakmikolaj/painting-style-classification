"""Model config in json format"""

import os


CFG = {
   "data": {
       "path_raw": os.path.join(os.path.dirname(os.getcwd()), 'data', 'raw'),
       "path_processed": os.path.join(os.path.dirname(os.getcwd()), 'data', 'processed'),
       "path_s3": "",
       "image_size": 200,
       "load_with_info": True
   },
   "train": {
       "batch_size": 50,
       "buffer_size": 1000,
       "epochs": 30,
       "optimizer": {
           "type": "adam"
       },
       "metrics": ["accuracy"]
   },
   "models": {
       "input": [200, 200, 1],
       "up_stack": {
           "layer_1": 512,
           "layer_2": 256,
           "layer_3": 128,
           "layer_4": 64,
           "kernels": 3
       },
       "output": 11
   }
}

