#!/bin/sh

set -e

STORAGE_DIR="${STORAGE_DIR:-/app/storage}"

# 确保存储目录存在，处理首次启动或宿主机空目录的情�?
if [ ! -d "$STORAGE_DIR" ]; then
    mkdir -p "$STORAGE_DIR"
fi

# 检查目录所有权是否为应用用户，若不是则修正以避免挂载权限问�?
if [ "$(stat -c %u "$STORAGE_DIR" 2>/dev/null || echo)" != "1000" ] || \
   [ "$(stat -c %g "$STORAGE_DIR" 2>/dev/null || echo)" != "1000" ]; then
    chown -R appuser:appuser "$STORAGE_DIR" || echo "Warning: unable to adjust ownership of $STORAGE_DIR"
fi

exec "$@"
