from huggingface_hub import HfApi
import os
import sys

# Configuration - Token should be set via environment variable or HF CLI login
# Run: huggingface-cli login
# Or set: export HF_TOKEN=your_token_here
TOKEN = os.environ.get("HF_TOKEN", None)
REPO_ID = "Danielchris145/IronGuard"
FOLDER_PATH = "."

def deploy():
    print(f"🚀 Starting deployment to {REPO_ID}...")
    try:
        # Use token from env or HF CLI cached login
        api = HfApi(token=TOKEN) if TOKEN else HfApi()
        
        # Verify login/token works
        user = api.whoami()
        print(f"✅ Authenticated as: {user['name']}")
        
        # Upload folder
        print("📤 Uploading files...")
        api.upload_folder(
            folder_path=FOLDER_PATH,
            repo_id=REPO_ID,
            repo_type="space",
            ignore_patterns=[".git", ".git/*", "deploy.py", "__pycache__", "*.pyc"],
            commit_message="Automated deployment via API"
        )
        print("✅ Upload complete!")
        print(f"🔗 View Space: https://huggingface.co/spaces/{REPO_ID}")
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    deploy()
