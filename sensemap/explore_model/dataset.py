import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class MapReconstructionDataset(Dataset):
    def __init__(self, dataset_dir):
        """
        Args:
            dataset_dir (str): 数据所在目录，例如 'dataset'。
        """
        self.dataset_dir = dataset_dir
        self.data_ids = [f.name for f in os.scandir(dataset_dir) if f.is_dir()]
        self.local_map_path = 'local_map_0.png'
        self.obs_map_path = 'obs_0.png'

    def __len__(self):
        return len(self.data_ids)

    def _get_image(self, data_id, filename, color_conversion=None):
        """读取并返回图像数据，避免重复代码"""
        image_path = os.path.join(self.dataset_dir, data_id, filename)
        image = Image.open(image_path)
        if color_conversion:
            image = image.convert(color_conversion)
        return image

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]

        # 读取标签图像
        label_map = self._get_image(data_id, self.local_map_path, color_conversion='L')
        # 读取特征图像并转换为 RGB
        feature_map = self._get_image(data_id, self.obs_map_path, color_conversion='RGB')

        # 将图像大小调整为 256x256
        label_map = label_map.resize((256, 256), Image.NEAREST)
        feature_map = feature_map.resize((256, 256), Image.NEAREST)

        # 将标签图像处理成二值图
        label_data = np.array(label_map) == 255
        label_data = label_data.astype(np.float32)

        # 将特征图中的所有 255 的像素值设置为 1
        feature_map = np.array(feature_map)
        feature_map[feature_map == 255] = 1.0

        # 创建 mask 数据
        mask_data = feature_map[:, :, 1].astype(np.float32)

        # 转换为 PyTorch 张量
        feature_map = torch.tensor(feature_map, dtype=torch.float32).permute(2, 0, 1)
        label_data = torch.tensor(label_data, dtype=torch.float32).unsqueeze(0)
        mask_data = torch.tensor(mask_data, dtype=torch.float32).unsqueeze(0)

        return feature_map, label_data, mask_data, data_id
