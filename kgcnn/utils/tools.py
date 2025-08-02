import subprocess
import os
from typing import Optional

def get_git_commit_hash() -> Optional[str]:
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=script_dir).strip().decode('utf-8')
        return commit_hash
    except subprocess.CalledProcessError as e:
        print(f"Error obtaining git commit hash: {e}")
        return None