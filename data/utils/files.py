from typing import List
from pathlib import Path

def find_sevd_sequences(root_path: Path) -> List[Path]:

    sequence_paths = []
    for p in root_path.rglob('*'):
        if p.is_dir():
            has_subdirs = any(child.is_dir() for child in p.iterdir())
            if not has_subdirs:
                sequence_paths.append(p)
    
    return sorted(sequence_paths)