import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# CSV 文件路径
current_dir = Path(__file__).resolve().parent
csv_dir = current_dir.parent / "data" / "csv"
csv_file = f"{csv_dir}/SAC_stable_new_obs_20250525_172044.csv"

# 读取数据
df = pd.read_csv(csv_file)

# 可视化：每个指标随 n_updates 变化
plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.plot(df["train/n_updates"], df["train/actor_loss"], label="actor_loss")
plt.xlabel("n_updates")
plt.ylabel("actor_loss")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(df["train/n_updates"], df["train/critic_loss"], label="critic_loss")
plt.xlabel("n_updates")
plt.ylabel("critic_loss")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(df["train/n_updates"], df["train/ent_coef"], label="ent_coef")
plt.xlabel("n_updates")
plt.ylabel("ent_coef")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(df["train/n_updates"], df["train/ent_coef_loss"], label="ent_coef_loss")
plt.xlabel("n_updates")
plt.ylabel("ent_coef_loss")
plt.legend()

plt.tight_layout()
plt.suptitle("SAC Training Log Analysis", y=1.02)
plt.show()
