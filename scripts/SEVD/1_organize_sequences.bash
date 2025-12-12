#!/usr/bin/env bash
set -e

# ============================================================
# organize_sequences.sh
#   - CARLAシミュレーションデータ構造整理スクリプト
#   - base_dir/001_Town.../Town.../ego0/* を base_dir/001_Town.../ へ移動
#   - metadata*.json も上に移動
#   - Town... ディレクトリは削除 (ego0 + metadata 以外も全削除)
#   - 衝突時は (2), (3), ... とリネーム
#   - _ERROR, .DS_Store などを削除
#   - --dry-run で確認モード
# ============================================================

# -------- 色 -------
NC="\033[0m"
CYAN="\033[0;36m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"

# -------- Dry-run オプション処理 -------
DRY_RUN=0
TARGET_DIR=""

for arg in "$@"; do
    case "$arg" in
        --dry-run|-n)
            DRY_RUN=1
            ;;
        *)
            TARGET_DIR="$arg"
            ;;
    esac
done

if [[ -z "$TARGET_DIR" ]]; then
    echo -e "${RED}Usage:${NC} $0 <base_dir> [--dry-run]"
    exit 1
fi

if [[ ! -d "$TARGET_DIR" ]]; then
    echo -e "${RED}Error:${NC} $TARGET_DIR is not a directory."
    exit 1
fi

cd "$TARGET_DIR"

if [[ $DRY_RUN -eq 1 ]]; then
    echo -e "${YELLOW}[ DRY RUN MODE ]${NC}"
fi

log()  { echo -e "${CYAN}[*]${NC} $1"; }
ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERR]${NC} $1"; }

# ============================================================
# 1. ゴミファイル削除
# ============================================================
log "Removing _ERROR, .DS_Store, Thumbs.db ..."
if [[ $DRY_RUN -eq 0 ]]; then
    find . -name "_ERROR*" -exec rm -rf {} +
    find . -type f -name ".DS_Store" -delete
    find . -type f -name "Thumbs.db" -delete
fi
ok "Garbage cleaned."

# ============================================================
# 2. シーン探索 (例: 001_Town01_Opt_ClearNoon)
# ============================================================
SCENES=( $(find . -maxdepth 1 -type d -name "*Town*" | sort) )
if [[ ${#SCENES[@]} -eq 0 ]]; then
    err "No Town scene directories found under $TARGET_DIR"
    exit 1
fi

log "Found scenes:"
for s in "${SCENES[@]}"; do
    echo "   - $s"
done

# ============================================================
# 衝突時リネーム関数
# ============================================================
rename_if_exists() {
    local target="$1"
    local base="${target%.*}"
    local ext="${target##*.}"
    if [[ "$ext" == "$target" ]]; then
        ext=""
    else
        ext=".$ext"
    fi
    local n=2
    while [[ -e "$target" ]]; do
        target="${base}(${n})${ext}"
        ((n++))
    done
    echo "$target"
}

# ============================================================
# 3. 各シーンを処理
# ============================================================
for scene in "${SCENES[@]}"; do
    # scene 内の Town... を探す（必ず1つ）
    SUBDIR=$(find "$scene" -maxdepth 1 -type d -name "Town*" | sort | head -n 1)
    if [[ -z "$SUBDIR" ]]; then
        warn "No Town subdirectory found inside $scene"
        continue
    fi

    log "Processing scene: $scene"
    EGO_PATH="$SUBDIR/ego0"

    if [[ ! -d "$EGO_PATH" ]]; then
        warn "ego0 not found in $SUBDIR, skipping."
        continue
    fi

    # 3-1: metadata*.json を scene 直下へ移動
    for meta in "$SUBDIR"/metadata*.json; do
        [[ -f "$meta" ]] || continue
        dest="$scene/$(basename "$meta")"
        if [[ -e "$dest" ]]; then
            newname=$(rename_if_exists "$dest")
            warn "metadata collision: renaming to $newname"
            dest="$newname"
        fi
        log "Moving metadata: $(basename "$meta") -> $dest"
        [[ $DRY_RUN -eq 0 ]] && mv "$meta" "$dest"
    done

    # 3-2: ego0 内部を scene 直下へ移動
    shopt -s dotglob
    for item in "$EGO_PATH"/*; do
        [[ -e "$item" ]] || continue
        basename_item=$(basename "$item")
        target="$scene/$basename_item"

        if [[ -e "$target" ]]; then
            newname=$(rename_if_exists "$target")
            warn "Collision: $basename_item -> $(basename "$newname")"
            target="$newname"
        fi

        log "Moving $basename_item -> $scene"
        [[ $DRY_RUN -eq 0 ]] && mv "$item" "$target"
    done
    shopt -u dotglob

    # 3-3: Town... ディレクトリごと削除
    log "Removing old Town directory: $SUBDIR"
    [[ $DRY_RUN -eq 0 ]] && rm -rf "$SUBDIR"

    ok "Scene cleaned: $scene"
done

ok "All sequences processed successfully."
