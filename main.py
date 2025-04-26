# main.py

# --- 0. 標準函式庫和日誌設定 ---
import importlib.util
import sys
import os
import logging
from pathlib import Path
from typing import Optional # FastAPI 可能需要

# --- 1. 檢查所需的第三方函式庫 (先導入檢查本身需要的) ---
try:
    from dotenv import load_dotenv
    import google.auth
except ImportError as e:
     # 如果連 dotenv 或 google.auth 都沒有，後續檢查意義不大
     print(f"FATAL: Core check dependencies missing: {e}. Please install 'python-dotenv' and 'google-auth'.", file=sys.stderr)
     sys.exit(1) # 嚴重錯誤，直接退出

# 設定基礎日誌 (盡早設定)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(asctime)s - %(message)s')

# --- 2. 環境檢查邏輯 (來自 check_env.py) ---

REQUIREMENTS_FILE = "requirements.txt"
# 較為完整的 pip 名稱到 import 名稱的映射
PACKAGE_TO_MODULE_MAP = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "requests": "requests",
    "google-cloud-aiplatform": "google.cloud.aiplatform",
    "google-cloud-speech": "google.cloud.speech",
    "python-dotenv": "dotenv",
    "line-bot-sdk": "linebot",
    "python-multipart": "multipart",
    "setuptools": "setuptools",
    "pytest": "pytest",
    "pytest-mock": "pytest_mock",
    "httpx": "httpx",
    "gunicorn": "gunicorn",
}

def check_packages(requirements_path: str):
    """讀取 requirements.txt 並檢查套件是否可導入。"""
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

        # 簡化解析，提取基礎包名
        package_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('<')[0].split('>')[0].split('~=')[0].split('!=')[0].split('[')[0].strip()

        if not package_name or package_name in packages_checked:
            continue
        packages_checked.add(package_name)

        module_name = None
        base_package = package_name.split('[')[0] # 處理 extras like [all], [standard]
        if base_package in PACKAGE_TO_MODULE_MAP:
            module_name = PACKAGE_TO_MODULE_MAP[base_package]
        elif package_name in PACKAGE_TO_MODULE_MAP: # 直接匹配
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

def check_credentials():
    """檢查 Google Cloud 憑證狀態。"""
    logging.info("--- Checking Google Cloud Credentials ---")
    gac_env_var = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if gac_env_var:
        logging.info(f" Environment variable 'GOOGLE_APPLICATION_CREDENTIALS' is SET.")
        # 簡單檢查是否像 JSON 內容
        if '{' in gac_env_var and '}' in gac_env_var and '"type":' in gac_env_var and ('private_key' in gac_env_var or 'client_email' in gac_env_var):
             logging.info(f"   -> Value looks like inline JSON content.")
        elif Path(gac_env_var).is_file(): # 檢查是否為本地存在的檔案路徑 (主要用於本地測試)
            logging.info(f"   -> Value points to an EXISTING local file: '{gac_env_var}'")
        else: # 可能是設定了但內容錯誤或指向不存在的路徑
            logging.warning(f"   -> Value is set but doesn't look like valid inline JSON or an existing local file path: '{gac_env_var[:50]}...'")
    else:
        logging.warning(" Environment variable 'GOOGLE_APPLICATION_CREDENTIALS' is NOT SET.")
        logging.warning("  -> Relying on Application Default Credentials (ADC) search order (e.g., attached service account in Cloud Run/GCE).") # Zeabur 上這個通常意味著失敗

    logging.info(" Attempting to find credentials using google.auth.default()...")
    try:
        credentials, project_id = google.auth.default() # 這會實際嘗試加載 GAC_ENV_VAR 或尋找其他預設憑證
        if credentials:
            logging.info("  -> SUCCESS: google.auth.default() found credentials.")
            if project_id:
                logging.info(f"     -> Associated Project ID found: {project_id}")
            return True
        else:
             logging.error("  -> UNEXPECTED: google.auth.default() returned None without error.")
             return False
    except google.auth.exceptions.DefaultCredentialsError as e:
        logging.error(f"  -> FAILED: google.auth.default() could not find credentials: {e}")
        logging.error("     -> Ensure GOOGLE_APPLICATION_CREDENTIALS env var has the correct JSON content OR running with implicit credentials.")
        return False
    except Exception as e:
        logging.error(f"  -> UNEXPECTED ERROR during google.auth.default(): {e}")
        return False


# --- 3. 執行環境檢查 (在腳本加載時立即執行) ---

logging.info("====== Running Pre-Startup Environment Checks ======")
# 嘗試加載 .env 文件 (對本地開發有用, 在 Zeabur 上會被環境變數覆蓋)
env_path = Path('.env')
if env_path.is_file():
    logging.info("Found .env file, loading variables (will be overridden by platform env vars).")
    load_dotenv()
else:
     logging.info("No .env file found.")

packages_ok = check_packages(REQUIREMENTS_FILE)
credentials_ok = check_credentials()

if not packages_ok or not credentials_ok:
    logging.critical("====== Environment checks FAILED. Application will NOT start. ======")
    logging.critical("Please review the logs above to fix the issues.")
    # 在 Zeabur 等生產環境中，讓啟動失敗很重要
    sys.exit(1) # 關鍵：如果檢查失敗，退出程式，阻止 FastAPI 啟動

logging.info("====== Environment Checks Passed. Proceeding with FastAPI Initialization ======")

# --- 4. FastAPI 應用程式定義 (只有在檢查通過後才會執行到這裡) ---
# 在這裡導入 FastAPI 和其他應用程式需要的模組比較安全
from fastapi import FastAPI

# 在這裡初始化 Google Cloud 客戶端也比較安全 (雖然檢查階段已確認能找到憑證)
# import google.cloud.aiplatform as vertexai
# try:
#     project_id = os.getenv("VERTEX_AI_PROJECT_ID")
#     location = os.getenv("VERTEX_AI_LOCATION")
#     if project_id and location:
#         vertexai.init(project=project_id, location=location)
#         logging.info(f"Vertex AI SDK initialized for {project_id} in {location}")
#     else:
#         logging.warning("VERTEX_AI_PROJECT_ID or VERTEX_AI_LOCATION not set, skipping Vertex AI init.")
# except Exception as e:
#     logging.error(f"Failed to initialize Vertex AI SDK: {e}")
#     # 根據需要決定是否退出 sys.exit(1)

app = FastAPI()

@app.get("/")
async def root():
    # 返回簡單訊息，或包含一些檢查狀態
    return {
        "message": "Hello World - FastAPI is running!",
        "environment_checks": "Passed" # 因為如果沒過，程式已經退出了
        }

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}

# --- 5. 本地開發運行器 (通常由 Gunicorn 在伺服器上處理) ---
if __name__ == "__main__":
    # 只有直接運行 python main.py 時才執行
    logging.info("Running FastAPI app locally using Uvicorn...")
    try:
        import uvicorn
        # 注意：PORT 環境變數在本地可能未設定，但在 Zeabur 上會由平台提供
        port = int(os.getenv("PORT", 8000)) # 預設本地用 8000
        uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
    except ImportError:
         logging.error("Uvicorn not found. Please install it: pip install 'uvicorn[standard]'")
    except Exception as e:
         logging.error(f"Failed to start Uvicorn: {e}")
