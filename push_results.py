from huggingface_hub import HfApi
from pathlib import Path
import config, os


def push_daily_result(local_path: Path):
    token = config.HF_TOKEN or os.environ.get("HF_TOKEN")
    if not token:
        print("❌ No HF_TOKEN. Skipping upload."); return
    api = HfApi(token=token)
    try:
        api.upload_file(path_or_fileobj=str(local_path), path_in_repo=local_path.name,
                         repo_id=config.OUTPUT_REPO, repo_type="dataset",
                         commit_message=f"Add {local_path.name}")
        print(f"✅ Uploaded {local_path.name} to {config.OUTPUT_REPO}")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
