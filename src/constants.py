from pathlib import Path

current_file_dir = Path(__file__).resolve().parent
PROJECT_PATH = current_file_dir.parent
EXPERIMENTS_PATH = PROJECT_PATH / 'experiments'
