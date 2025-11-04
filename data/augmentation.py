import numpy as np
import torch
from typing import List, Callable
from typing import Tuple

class TimeSeriesAugmentation:
    """时间序列数据增强"""
    
    @staticmethod
    def time_jitter(x: np.ndarray, sigma: float = 0.05) -> np.ndarray:
        """时间抖动"""
        jitter = np.random.normal(0, sigma, x.shape)
        return x + jitter
    
    @staticmethod
    def amplitude_scaling(x: np.ndarray, sigma: float = 0.1) -> np.ndarray:
        """幅度缩放"""
        scaling_factor = np.random.normal(1, sigma)
        return x * scaling_factor
    
    @staticmethod
    def add_gaussian_noise(x: np.ndarray, sigma: float = 0.1) -> np.ndarray:
        """添加高斯噪声"""
        noise = np.random.normal(0, sigma, x.shape)
        return x + noise
    
    @staticmethod
    def random_crop(x: np.ndarray, crop_ratio: Tuple[float, float] = (0.8, 1.0)) -> np.ndarray:
        """随机裁剪"""
        min_ratio, max_ratio = crop_ratio
        crop_ratio = np.random.uniform(min_ratio, max_ratio)
        crop_length = int(len(x) * crop_ratio)
        
        start_idx = np.random.randint(0, len(x) - crop_length + 1)
        cropped = x[start_idx:start_idx + crop_length]
        
        # 插值回原始长度
        from scipy.interpolate import interp1d
        original_indices = np.linspace(0, crop_length - 1, len(x))
        f = interp1d(np.arange(crop_length), cropped, axis=0, 
                    kind='linear', fill_value='extrapolate')
        return f(original_indices)
    
    @staticmethod
    def time_warping(x: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        """时间扭曲"""
        from scipy.interpolate import CubicSpline
        
        n_time = len(x)
        original_indices = np.arange(n_time)
        
        # 生成扭曲点
        warp_steps = np.random.normal(1.0, sigma, 5)
        warp_steps = np.cumsum(warp_steps)
        warp_steps = (warp_steps - warp_steps[0]) * (n_time - 1) / (warp_steps[-1] - warp_steps[0])
        
        # 创建扭曲函数
        warp_indices = np.linspace(0, n_time - 1, 5)
        cs = CubicSpline(warp_indices, warp_steps)
        warped_indices = cs(original_indices)
        warped_indices = np.clip(warped_indices, 0, n_time - 1)
        
        # 应用扭曲
        from scipy.interpolate import interp1d
        f = interp1d(original_indices, x, axis=0, kind='linear', 
                    fill_value='extrapolate')
        return f(warped_indices)
    
    @staticmethod
    def frequency_masking(x: np.ndarray, mask_ratio: float = 0.1) -> np.ndarray:
        """频率掩码"""
        # FFT变换
        x_fft = np.fft.fft(x, axis=0)
        n_freq = len(x_fft)
        
        # 随机选择要掩码的频率
        n_mask = int(n_freq * mask_ratio)
        mask_indices = np.random.choice(n_freq, n_mask, replace=False)
        
        # 应用掩码
        x_fft[mask_indices] = 0
        
        # 逆FFT变换
        return np.real(np.fft.ifft(x_fft, axis=0))
    
    @staticmethod
    def window_shuffle(x: np.ndarray, n_windows: int = 5) -> np.ndarray:
        """窗口洗牌"""
        window_size = len(x) // n_windows
        windows = [x[i*window_size:(i+1)*window_size] for i in range(n_windows)]
        np.random.shuffle(windows)
        return np.concatenate(windows, axis=0)

class ContrastiveAugmentor:
    """对比学习数据增强器"""
    
    def __init__(self, weak_augmentations: List[Callable] = None,
                 strong_augmentations: List[Callable] = None):
        
        self.weak_augmentations = weak_augmentations or [
            lambda x: TimeSeriesAugmentation.time_jitter(x, 0.05),
            lambda x: TimeSeriesAugmentation.amplitude_scaling(x, 0.1),
            lambda x: TimeSeriesAugmentation.add_gaussian_noise(x, 0.1),
            lambda x: TimeSeriesAugmentation.random_crop(x, (0.9, 1.0))
        ]
        
        self.strong_augmentations = strong_augmentations or [
            lambda x: TimeSeriesAugmentation.time_warping(x, 0.2),
            lambda x: TimeSeriesAugmentation.frequency_masking(x, 0.2),
            lambda x: TimeSeriesAugmentation.window_shuffle(x, 8)
        ]
    
    def weak_augment(self, x: np.ndarray) -> np.ndarray:
        """弱增强"""
        aug_fn = np.random.choice(self.weak_augmentations)
        return aug_fn(x)
    
    def strong_augment(self, x: np.ndarray) -> np.ndarray:
        """强增强"""
        aug_fn = np.random.choice(self.strong_augmentations)
        return aug_fn(x)
    
    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """生成增强样本对"""
        view1 = self.weak_augment(x)
        view2 = self.strong_augment(x)
        return view1, view2