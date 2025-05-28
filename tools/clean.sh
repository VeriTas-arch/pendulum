#!/bin/bash

# 进入脚本所在目录
cd "$(dirname "$0")"

# 自动移除Log目录下名称为SAC_*的文件夹
rm -rf ./log/SAC_*
