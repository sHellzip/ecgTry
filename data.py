import wfdb  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
from glob import glob
from numpy.typing import NDArray
from typing import Any
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler  # 确保你已经安装了 scikit-learn 包

# 初始化结果字典，每种标签类型都对应一个形状为(0, 256)的空数组
result: dict[str, NDArray[Any]] = {}
for label in ["N", "A", "V", "L", "R"]:
    result[label] = np.empty([0, 256], dtype=np.float32)  # 修改为滑动窗口大小


# 判断标注符号是否是目标类型
def is_target(symbol: str) -> bool:
    return symbol in ["N", "A", "V", "L", "R"]


# 创建滑动窗口的函数
def create_windows(data, window_size, step_size):
    windows = []
    for i in range(0, len(data) - window_size + 1, step_size):
        windows.append(data[i:i + window_size])
    return np.array(windows)


# 获取每个窗口的标签（距离最近的标注）
def get_closest_label(window_start, window_end, annotations):
    closest_label = None
    min_distance = float('inf')

    for label, pos in zip(annotations.symbol, annotations.sample):
        if window_start <= pos <= window_end:
            distance = min(abs(window_start - pos), abs(window_end - pos))
            if distance < min_distance:
                min_distance = distance
                closest_label = label
    return closest_label


# 滑动窗口大小和步长
window_size = 256
step_size = 128

# 处理所有的 .dat 文件
for path in tqdm(glob("./assets/*.dat", recursive=False)):
    record_name: str = path[:-4]  # 去掉文件扩展名，得到记录名
    # 读取 ECG 信号
    record = np.array(
        wfdb.rdrecord(record_name, channel_names=["MLII"]).p_signal,
        dtype=np.float32,
    )

    # 对信号进行归一化，使用 scikit-learn 中的 MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized_signal = scaler.fit_transform(record.flatten().reshape(-1, 1)).flatten()

    # 创建滑动窗口
    windows = create_windows(normalized_signal, window_size, step_size)

    # 读取标注数据
    annotation = wfdb.rdann(record_name, "atr")
    if annotation.symbol is None:
        continue

    # 为每个窗口找到最近的标注，并保存到对应的标签字典中
    for i in range(len(windows)):
        window_start = i * step_size
        window_end = window_start + window_size
        label = get_closest_label(window_start, window_end, annotation)

        # 只保留需要的五种类型
        if label in result.keys() and is_target(label):
            # 限制每种标签类型最多存储10000个窗口
            if result[label].shape[0] >= 10000:
                continue
            result[label] = np.vstack((result[label], windows[i]))

# 分别保存每种标签的数据为 .npy 文件
for label, window_data in result.items():
    if window_data.size > 0:  # 如果该标签有数据
        np.save(f'./assets/{label}.npy', window_data)

print("处理完成，每个标签的窗口数据已分别保存为 .npy 文件！")
