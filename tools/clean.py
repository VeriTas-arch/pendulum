import shutil
from pathlib import Path

# 进入脚本所在目录的上一级
ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT_DIR / "log"
CSV_DIR = LOG_DIR / "csv"

# 删除 log 目录下 SAC_* 文件夹
for item in LOG_DIR.glob("SAC_*"):
    if item.is_dir():
        shutil.rmtree(item)

# 删除 log 目录下 TD3_* 文件夹
for item in LOG_DIR.glob("TD3_*"):
    if item.is_dir():
        shutil.rmtree(item)

# 删除 log/csv 目录下所有内容
if CSV_DIR.exists():
    for item in CSV_DIR.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
else:
    CSV_DIR.mkdir(parents=True, exist_ok=True)

# 重新创建 log/csv 目录
CSV_DIR.mkdir(parents=True, exist_ok=True)
