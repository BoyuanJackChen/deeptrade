import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 读取数据
data = pd.read_csv('trading_data.csv', parse_dates=['date'])
data = data.sort_values('date').reset_index(drop=True)

# 选择特征
features = ['open', 'high', 'low', 'close', 'volume']
data_features = data[features].values

# 归一化
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_features)

# 划分训练和测试集
split = int(0.8 * len(scaled_data))
train_data = scaled_data[:split]
test_data = scaled_data[split:]

# 保存预处理后的数据（可选）
np.save('scaled_train_data.npy', train_data)
np.save('scaled_test_data.npy', test_data)