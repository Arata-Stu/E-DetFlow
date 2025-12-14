from typing import List
from pathlib import Path

def find_sevd_sequences(root_path: Path) -> List[Path]:
    sequence_paths = []
    
    # 1階層目: シーンディレクトリ (例: 001_Town01_Opt_ClearNoon)
    for scene_dir in root_path.iterdir():
        if not scene_dir.is_dir():
            continue
            
        # 2階層目: シーケンス番号 (例: 01, 02)
        for seq_dir in scene_dir.iterdir():
            if seq_dir.is_dir():
                sequence_paths.append(seq_dir)
                
    # 順序を保証するためにソートして返す
    return sorted(sequence_paths)