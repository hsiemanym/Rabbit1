# utils/config_utils.py
import yaml

def load_config():
    """config.yml 파일을 로드합니다."""
    with open("config.yml", "r") as f:
        return yaml.safe_load(f)