import os


CFG = {
   "data": {
       "path_raw": "oxford_iiit_pet:3.*.*",
       "path_processed":"",
       "path_s3":"",
       "image_size": 128,
       "load_with_info": True
   },
   "train": {
       "batch_size": 64,
       "buffer_size": 1000,
       "epochs": 20,
       "val_subsplits": 5,
       "optimizer": {
           "type": "adam"
       },
       "metrics": ["accuracy"]
   },
   "model": {
       "input": [128, 128, 3],
       "up_stack": {
           "layer_1": 512,
           "layer_2": 256,
           "layer_3": 128,
           "layer_4": 64,
           "kernels": 3
       },
       "output": 3
   }
}

