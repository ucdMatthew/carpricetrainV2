import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

# 读取数据
data = pd.read_csv("carprice.csv")

# 数据清理
data['mileage'] = data['mileage'].str.replace(',', '').str.extract('(\d+)').astype(float)
data['price'] = data['price'].str.replace('$', '').str.replace(',', '').astype(float)

# 对其他分类变量进行标签编码
label_encoders = {}
cat_cols = ['brand', 'model', 'fuel_type', 'engine', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title']
for col in cat_cols:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

# 分割特征和目标变量
X = data.drop('price', axis=1)
y = data['price']

# 划分训练集、验证集和测试集
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# 标准化处理
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# GradientBoostingRegressor 模型
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train_scaled, y_train)

# 在测试集上进行预测
y_pred = gb_model.predict(X_test_scaled)

# 计算均方误差、平均绝对误差和 R^2
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Squared Error on Test Set: {mse}")
print(f"R^2 Score on Test Set: {r2}")
print(f"Mean Absolute Error on Test Set: {mae}")

# 绘制训练损失曲线
plt.figure(figsize=(10, 6))
plt.plot(gb_model.train_score_, label='Training Loss')
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid(True)
plt.show()

# 绘制真实价格与预测价格的对比图
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("True Values []")
plt.ylabel("Predictions&nbsp;[]")
plt.title("True vs Predicted Prices")
plt.show()