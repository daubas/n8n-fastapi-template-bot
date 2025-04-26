# main.py
import importlib.util
import sys
import os
import logging
from pathlib import Path
import json # <--- 新增導入 json 模組
import tempfile # <--- 新增導入 tempfile 模組
from typing import Optional

try:
    from dotenv import load_dotenv
    import google.auth
except ImportError as e:
     print(f"FATAL: Core check dependencies missing: {e}. Please install 'python-dotenv' and 'google-auth'.", file=sys.stderr)
     sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(asctime)s - %(message)s')

# --- 常數和套件映射 ---
REQUIREMENTS_FILE = "requirements.txt"
PACKAGE_TO_MODULE_MAP = {
    "fastapi": "fastapi", "uvicorn": "uvicorn", "requests": "requests",
    "google-cloud-aiplatform": "google.cloud.aiplatform",
    "google-cloud-speech": "google.cloud.speech", "python-dotenv": "dotenv",
    "line-bot-sdk": "linebot", "python-multipart": "multipart",
    "setuptools": "setuptools", "pytest": "pytest", "pytest-mock": "pytest_mock",
    "httpx": "httpx", "gunicorn": "gunicorn",
}

# --- 環境檢查函式 (check_packages 保持不變) ---
def check_packages(requirements_path: str):
    # ... (之前的 check_packages 函式代碼保持不變) ...
    installed_count = 0
    missing_packages = []
    packages_checked = set()

    logging.info(f"--- Checking Packages listed in {requirements_path} ---")
    req_file = Path(requirements_path)
    if not req_file.is_file():
        logging.error(f"'{requirements_path}' not found!")
        return False

    try:
        with open(req_file, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        logging.error(f"Error reading '{requirements_path}': {e}")
        return False

    for line_num, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('-e'):
            continue
        package_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('<')[0].split('>')[0].split('~=')[0].split('!=')[0].split('[')[0].strip()
        if not package_name or package_name in packages_checked:
            continue
        packages_checked.add(package_name)

        module_name = None
        base_package = package_name.split('[')[0]
        if base_package in PACKAGE_TO_MODULE_MAP:
            module_name = PACKAGE_TO_MODULE_MAP[base_package]
        elif package_name in PACKAGE_TO_MODULE_MAP:
             module_name = PACKAGE_TO_MODULE_MAP[package_name]
        else:
            logging.warning(f" No import name mapped for '{package_name}' (from line {line_num+1} in {requirements_path}). Skipping check.")
            continue

        logging.info(f" Checking: '{package_name}' (via import '{module_name}')")
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            logging.error(f"  -> MISSING! Package '{package_name}' (module '{module_name}') not found.")
            missing_packages.append(package_name)
        else:
            logging.info(f"  -> Found.")
            installed_count += 1

    if missing_packages:
        logging.error(f" Package Summary: Found {installed_count} / Checked {len(packages_checked)}. Missing: {', '.join(missing_packages)}")
        logging.error(f" Please ensure dependencies are installed (e.g., check build logs for 'pip install -r {requirements_path}').")
        return False
    else:
        logging.info(f" Package Summary: OK - All {installed_count} checked packages appear to be installed.")
        return True


def check_and_setup_credentials():
    """檢查並設定 Google Cloud 憑證，將 JSON 內容寫入臨時文件。"""
    logging.info("--- Checking and Setting Up Google Cloud Credentials ---")
    gac_env_var_content = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if not gac_env_var_content:
        logging.error(" Environment variable 'GOOGLE_APPLICATION_CREDENTIALS' is NOT SET.")
        logging.error("  -> Cannot proceed without credentials.")
        return False # 沒有設定內容，直接失敗

    logging.info(" Environment variable 'GOOGLE_APPLICATION_CREDENTIALS' is SET.")

    # 驗證內容是否是有效的 JSON
    try:
        json.loads(gac_env_var_content) # 嘗試解析 JSON
        logging.info("  -> Value appears to be valid JSON content.")
    except json.JSONDecodeError as e:
        logging.error(f"  -> Value is NOT valid JSON content: {e}")
        logging.error("     Please ensure the environment variable contains the exact, complete JSON key content.")
        return False # JSON 格式錯誤，失敗

    # --- 關鍵步驟：將 JSON 內容寫入臨時文件 ---
    try:
        # 創建一個臨時文件 (會在關閉時自動刪除)
        # 使用 delete=False 確保文件在 with 區塊外仍然存在，直到程式退出
        # 在 /tmp 目錄創建 (多數 Linux 系統都有)
        creds_dir = Path(tempfile.gettempdir())
        creds_dir.mkdir(parents=True, exist_ok=True) # 確保目錄存在
        # 使用 NamedTemporaryFile 創建一個有名字的臨時文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, dir=creds_dir) as temp_creds_file:
             temp_creds_file.write(gac_env_var_content)
             temp_file_path = temp_creds_file.name # 獲取臨時文件的完整路徑
             logging.info(f"  -> Successfully wrote JSON content to temporary file: {temp_file_path}")

        # --- 將環境變數重新指向這個臨時文件 ---
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_file_path
        logging.info(f"  -> Environment variable 'GOOGLE_APPLICATION_CREDENTIALS' now points to: {temp_file_path}")

        # 現在再次嘗試使用 google.auth.default()，它應該會讀取這個文件
        logging.info(" Re-attempting to find credentials using google.auth.default() with file path...")
        credentials, project_id = google.auth.default()
        if credentials:
             logging.info("  -> SUCCESS: google.auth.default() found credentials using the temporary file.")
             if project_id:
                 logging.info(f"     -> Associated Project ID found: {project_id}")
             return True # 成功找到憑證
        else:
             logging.error("  -> UNEXPECTED: google.auth.default() returned None even after writing to temp file.")
             return False

    except google.auth.exceptions.DefaultCredentialsError as e:
         logging.error(f"  -> FAILED: google.auth.default() could not find credentials even with temp file: {e}")
         return False
    except OSError as e:
         logging.error(f"  -> FAILED: Could not write temporary credentials file (check permissions for /tmp?): {e}")
         return False
    except Exception as e:
        logging.error(f"  -> UNEXPECTED ERROR during credential setup: {e}")
        return False


# --- 執行環境檢查 ---
logging.info("====== Running Pre-Startup Environment Checks ======")
env_path = Path('.env')
if env_path.is_file():
    logging.info("Found .env file, loading variables.")
    load_dotenv()
else:
     logging.info("No .env file found.")

packages_ok = check_packages(REQUIREMENTS_FILE)
# *** 修改這裡：調用新的檢查和設定函式 ***
credentials_ok = check_and_setup_credentials()

if not packages_ok or not credentials_ok:
    logging.critical("====== Environment checks FAILED. Application will NOT start. ======")
    logging.critical("Please review the logs above to fix the issues.")
    sys.exit(1)

logging.info("====== Environment Checks Passed. Proceeding with FastAPI Initialization ======")

# --- FastAPI 應用程式定義 ---
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {
        "message": "Hello World - FastAPI is running!",
        "environment_checks": "Passed",
        "gac_path": os.getenv("GOOGLE_APPLICATION_CREDENTIALS") # 顯示目前 GAC 指向的路徑
        }

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}

# --- 本地開發運行器 ---
if __name__ == "__main__":
    logging.info("Running FastAPI app locally using Uvicorn...")
    try:
        import uvicorn
        port = int(os.getenv("PORT", 8000))
        uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
    except ImportError:
         logging.error("Uvicorn not found. Please install it: pip install 'uvicorn[standard]'")
    except Exception as e:
         logging.error(f"Failed to start Uvicorn: {e}")
