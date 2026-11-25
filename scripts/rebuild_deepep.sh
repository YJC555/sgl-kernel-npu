#!/bin/bash
# 重新编译 DeepEP 并把新生成的 .so / 自定义算子包更新到当前 Python 环境。
# 用法:
#   ./scripts/rebuild_deepep.sh          # 增量编译(自动识别 A2/A3/A5) + 安装
#   ./scripts/rebuild_deepep.sh -c       # 先清 build/output/build_out 再全量编译
#   ./scripts/rebuild_deepep.sh -d       # DEBUG 编译
#   ./scripts/rebuild_deepep.sh -s       # 只编译不安装
# 可用环境变量覆盖: SOC_VERSION(默认交给 build.sh 自动识别) PYTHON(默认 python3)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SOC_VERSION="${SOC_VERSION:-}"
PYTHON="${PYTHON:-python3}"

DO_CLEAN=0
DEBUG_MODE=0
DO_INSTALL=1

usage() {
    sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
}

while getopts ":cdsh" opt; do
    case "$opt" in
        c) DO_CLEAN=1 ;;
        d) DEBUG_MODE=1 ;;
        s) DO_INSTALL=0 ;;
        h) usage ;;
        *) echo "未知参数 -$OPTARG, 用 -h 查看用法" >&2; exit 1 ;;
    esac
done

echo "==> 工程目录: $ROOT"
echo "==> SOC_VERSION: ${SOC_VERSION:-自动识别}"

if [[ "$DO_CLEAN" == "1" ]]; then
    echo "==> 清理 build/ output/ build_out/"
    rm -rf "$ROOT/build" "$ROOT/output" "$ROOT/csrc/deepep/ops/build_out"
fi

# AscendC 的 kernel 缓存不会因为 .cpp 改动自动失效，必须清掉 kernel_meta，
# 否则改完 op_kernel/*.cpp 重编仍会链接旧算子。
echo "==> 清理 AscendC kernel_meta 缓存"
find "$ROOT/csrc/deepep" -type d -name kernel_meta -prune -exec rm -rf {} + 2>/dev/null || true

BUILD_ARGS=(-a deepep)
[[ -n "$SOC_VERSION" ]] && BUILD_ARGS+=("$SOC_VERSION")
[[ "$DEBUG_MODE" == "1" ]] && BUILD_ARGS+=(-d)

echo "==> bash build.sh ${BUILD_ARGS[*]}"
bash "$ROOT/build.sh" "${BUILD_ARGS[@]}"

WHEEL="$(ls -t "$ROOT"/output/deep_ep-*.whl 2>/dev/null | head -1 || true)"
if [[ -z "$WHEEL" ]]; then
    echo "Error: output/ 下没有找到 deep_ep wheel，编译可能失败" >&2
    exit 1
fi
echo "==> 新 wheel: $WHEEL"

if [[ "$DO_INSTALL" != "1" ]]; then
    echo "==> 按 -s 跳过安装，仅完成编译"
    exit 0
fi

echo "==> 安装 wheel 到当前 Python 环境"
"$PYTHON" -m pip install --force-reinstall --no-deps "$WHEEL"

SITE_DIR="$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')"
NEW_SO="$(ls "$SITE_DIR"/deep_ep/deep_ep_cpp*.so 2>/dev/null | head -1 || true)"
if [[ -z "$NEW_SO" ]]; then
    echo "Error: $SITE_DIR/deep_ep 下没有 deep_ep_cpp*.so" >&2
    exit 1
fi

# deep_ep_cpp 需要在 site-packages 根目录也有一份软链接，否则 import 报找不到
ln -sfn "deep_ep/$(basename "$NEW_SO")" "$SITE_DIR/$(basename "$NEW_SO")"
echo "==> 已更新: $NEW_SO"

echo "==> 校验 import 与 .so 时间戳"
"$PYTHON" - "$SITE_DIR" <<'EOF'
import glob
import os
import sys
import time

site_dir = sys.argv[1]
import deep_ep  # noqa: E402

print("deep_ep:", deep_ep.__file__)
paths = sorted(glob.glob(os.path.join(site_dir, "deep_ep", "deep_ep_cpp*.so")))
paths += sorted(glob.glob(os.path.join(site_dir, "deep_ep", "vendors", "*", "op_api", "lib", "*cust_opapi*.so")))
for path in paths:
    mtime = time.strftime("%F %T", time.localtime(os.path.getmtime(path)))
    print("  so:", path, mtime)
EOF

echo "==> 完成。从节点要跑的话，同步仓库后在从节点执行同一脚本。"
