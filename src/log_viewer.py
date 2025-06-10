import matplotlib.pyplot as plt
import pandas as pd

import utils

CSV_DIR = utils.CSV_DIR  # CSV目录
csv_file = f"{CSV_DIR}/SAC_test_new_obs_20250526_100822.csv"


df = pd.read_csv(csv_file)

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
