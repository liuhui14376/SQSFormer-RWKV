import os, sys
import json, time
import numpy as np
import torch
import platform
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# device = torch.device('cuda:0')
# device = torch.device("cpu")
# 检查CUDA是否可用
cuda_available = torch.cuda.is_available()

if cuda_available:
    # 获取GPU设备数量
    num_gpu = torch.cuda.device_count()

    # 获取当前使用的GPU索引
    current_gpu_index = torch.cuda.current_device()

    # 获取当前GPU的名称
    current_gpu_name = torch.cuda.get_device_name(current_gpu_index)

    # 获取GPU显存的总量和已使用量
    total_memory = torch.cuda.get_device_properties(current_gpu_index).total_memory / (1024 ** 3)  # 显存总量(GB)
    used_memory = torch.cuda.memory_allocated(current_gpu_index) / (1024 ** 3)  # 已使用显存(GB)
    free_memory = total_memory - used_memory  # 剩余显存(GB)

    print(f"CUDA可用，共有 {num_gpu} 个GPU设备可用。")
    print(f"当前使用的GPU设备索引：{current_gpu_index}")
    print(f"当前使用的GPU设备名称：{current_gpu_name}")
    print(f"GPU显存总量：{total_memory:.2f} GB")
    print(f"已使用的GPU显存：{used_memory:.2f} GB")
    print(f"剩余GPU显存：{free_memory:.2f} GB")
else:
    print("CUDA不可用。")
print("python.version:", platform.python_version())
print("torch.version:", torch.__version__)
print("CUDA.version:", torch.version.cuda)
print("cuDNN.version:", torch.backends.cudnn.version())

# 检查PyTorch版本
print(f"PyTorch版本：{torch.__version__}")

import torch
print(f"CUDA版本：{torch.version.cuda}")




# config_path_prefix = './params_use' #相对路径
config_path_prefix = "D:\python\SQSFormer-RWKV\src\params_use" #绝对路径
def check_convention(name):
    for a in ['knn', 'random_forest', 'svm']:
        if a in name:
            return True
    return False


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

    def get_avg(self):
        return self.avg

    def get_num(self):
        return self.cnt

class HSIRecoder(object):
    def __init__(self) -> None:
        self.record_data = {}
        self.pred = None

    def append_index_value(self, name, index, value):
        """
        index : int,
        value: Any
        save to dict
        {index: list, value: list}
        """
        if name not in self.record_data:
            self.record_data[name] = {
                "type": "index_value",
                "index":[],
                "value":[]
            }
        self.record_data[name]['index'].append(index)
        self.record_data[name]['value'].append(value)

    def record_time(self, time):
        self.record_data['eval_time'] = time

    def record_param(self, param):
        self.record_data['param'] = param

    def record_eval(self, eval_obj):
        self.record_data['eval'] = eval_obj

    def record_pred(self, pred_matrix):
        self.pred = pred_matrix

    def to_file(self, path):
        # 获取当前的时间戳（秒数）
        time_stamp = int(time.time())
        time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
        save_path_json = "%s_%s.json" % (path, time_str)
        save_path_json = r"D:/python/SQSFormer-RWKV/src/res/Indian_test_%s.json" % (
            time_str)
        save_path_pred = "%s_%s.pred.npy" % (path, str(time_stamp))

        ss = json.dumps(self.record_data, indent=4)
        with open(save_path_json, 'w') as fout:
            fout.write(ss)
            fout.flush()
        # np.save(save_path_pred, self.pred)
        #
        print("save record of %s done!" % path)

    def reset(self):
        self.record_data = {}


# global recorder
recorder = HSIRecoder()
