# main.py

# --- 0. 標準函式庫和日誌設定 ---
import importlib.util
import sys
import os
import logging
from fastapi import FastAPI  # <--- 添加或確保這一行存在
from pathlib import Path
import json
import base64
import asyncio
from typing import Optional, Literal

# --- 1. 檢查所需的第三方函式庫 ---
try:
    from dotenv import load_dotenv
    # 核心依賴
    import fastapi
    from pydantic import BaseModel, Field
    import requests # 雖然主要用 httpx，保留以防萬一
    import httpx # 用於圖像生成的直接 API 調用
    import openai # <-- 使用 OpenAI SDK

except ImportError as e:
     print(f"FATAL: Core application dependencies missing: {e}. Please ensure all packages in requirements.txt (including openai>=1.0.0) are installed.", file=sys.stderr)
     sys.exit(1)

# 設定基礎日誌
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 2. 環境檢查邏輯 (檢查套件和 Grok Key) ---
REQUIREMENTS_FILE = "requirements.txt"
# 更新映射以包含 openai
PACKAGE_TO_MODULE_MAP = {
    "fastapi": "fastapi", "uvicorn": "uvicorn", "requests": "requests",
    "python-dotenv": "dotenv", "line-bot-sdk": "linebot",
    "python-multipart": "multipart", "setuptools": "setuptools",
    "pytest": "pytest", "pytest-mock": "pytest_mock",
    "httpx": "httpx", "gunicorn": "gunicorn",
    "openai": "openai", # <-- 添加 openai 映射
}

def check_packages(requirements_path: str):
    # ... (之前的 check_packages 函式代碼，使用更新的 PACKAGE_TO_MODULE_MAP) ...
    installed_count = 0; missing_packages = []; packages_checked = set()
    logger.info(f"--- Checking Packages listed in {requirements_path} ---")
    req_file = Path(requirements_path)
    if not req_file.is_file(): logger.error(f"'{requirements_path}' not found!"); return False
    try:
        with open(req_file, 'r') as f: lines = f.readlines()
    except Exception as e: logger.error(f"Error reading '{requirements_path}': {e}"); return False
    for line_num, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('-e'): continue
        package_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('<')[0].split('>')[0].split('~=')[0].split('!=')[0].split('[')[0].strip()
        if not package_name or package_name in packages_checked: continue
        packages_checked.add(package_name)
        module_name = None; base_package = package_name.split('[')[0]
        if base_package in PACKAGE_TO_MODULE_MAP: module_name = PACKAGE_TO_MODULE_MAP[base_package]
        elif package_name in PACKAGE_TO_MODULE_MAP: module_name = PACKAGE_TO_MODULE_MAP[package_name]
        else: logger.warning(f" No import name mapped for '{package_name}' (from line {line_num+1} in {requirements_path}). Skipping check."); continue
        logger.info(f" Checking: '{package_name}' (via import '{module_name}')")
        spec = importlib.util.find_spec(module_name)
        if spec is None: logger.error(f"  -> MISSING! Package '{package_name}' (module '{module_name}') not found."); missing_packages.append(package_name)
        else: logger.info(f"  -> Found."); installed_count += 1
    if missing_packages: logger.error(f" Package Summary: Found {installed_count} / Checked {len(packages_checked)}. Missing: {', '.join(missing_packages)}"); logger.error(f" Please ensure dependencies are installed."); return False
    else: logger.info(f" Package Summary: OK - All {installed_count} checked packages appear to be installed."); return True


def check_grok_api_key():
    """檢查 GROK_API_KEY 環境變數是否存在。"""
    logger.info("--- Checking Grok API Key ---")
    grok_api_key = os.getenv("GROK_API_KEY")
    if not grok_api_key:
        logger.error("CRITICAL: Environment variable 'GROK_API_KEY' is NOT SET.")
        return False
    else:
        logger.info(f" Environment variable 'GROK_API_KEY' is SET (Key starts with: {grok_api_key[:4]}...).")
        return True

# --- 3. 執行環境檢查 ---
logging.info("====== Running Pre-Startup Environment Checks ======")
env_path = Path('.env')
if env_path.is_file(): logging.info("Found .env file, loading variables."); load_dotenv()
else: logging.info("No .env file found.")

packages_ok = check_packages(REQUIREMENTS_FILE)
key_ok = check_grok_api_key()

if not packages_ok or not key_ok:
    logging.critical("====== Environment checks FAILED. Application will NOT start. ======")
    logging.critical("Please review the logs above to fix the issues.")
    sys.exit(1)

logging.info("====== Environment Checks Passed. Proceeding with Initialization ======")

# --- 4. 初始化 OpenAI 客戶端以指向 Grok API ---
try:
    GROK_API_KEY = os.getenv("GROK_API_KEY")
    # *** 使用 OpenAI SDK 初始化，配置 Grok 端點 ***
    openai_client = openai.OpenAI(
        api_key=GROK_API_KEY,
        base_url="https://api.x.ai/v1", # 指向 xAI 的 API 端點
    )
    logging.info("OpenAI client configured for Grok API initialized successfully.")
    # 可選：嘗試調用一個簡單的方法來驗證連接，例如列出模型
    # try:
    #     models = openai_client.models.list()
    #     logger.info(f"Successfully listed models via Grok API: {[(m.id, m.owned_by) for m in models.data]}") # 可能需要調整打印格式
    # except Exception as list_err:
    #     logger.warning(f"Could not list models using configured client: {list_err}")

except Exception as e:
    logging.critical(f"Failed to initialize OpenAI client for Grok: {e}", exc_info=True)
    sys.exit(1)

# --- 5. 定義 Pydantic 模型 (與之前相同，無音頻) ---
class WebhookRequest(BaseModel):
    platform: str = Field(..., description="來源平台，例如 'line', 'discord'")
    user_id: str = Field(..., description="使用者 ID")
    message_type: Literal["text", "image", "sticker", "unknown"] = Field(..., description="訊息類型")
    message_content: Optional[str] = Field(None, description="文字訊息內容")
    is_mention: bool = Field(False, description="在群組中是否提及機器人")

class WebhookResponse(BaseModel):
    reply_type: Literal["text", "image", "error", "ignore"] = Field(..., description="回應類型")
    reply_content: Optional[str] = Field(None, description="回應內容 (文字或圖片 URL/Data URI)")

# --- 6. AI 輔助函式 (使用配置好的 OpenAI SDK 調用 Grok LLM) ---
async def generate_text_grok_via_openai(prompt: str) -> str:
    """使用配置好的 OpenAI SDK 調用 Grok LLM 生成文字"""
    try:
        logger.info(f"Calling Grok LLM via OpenAI SDK with prompt: {prompt[:100]}...")
        # 使用 OpenAI SDK 的 chat completions 方法
        # 注意：OpenAI SDK v1+ 是同步的，在異步函數中需要用 asyncio.to_thread
        response = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model="grok-3-beta",  # 指定要使用的 Grok 模型
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.7,
            # stream=False # 非流式輸出
        )
        logger.info("Grok LLM call via OpenAI SDK successful.")
        # 解析 OpenAI SDK 標準的回應結構
        if response.choices:
            return response.choices[0].message.content.strip()
        else:
            logger.warning("Grok LLM response (via OpenAI SDK) did not contain expected choices.")
            return "抱歉，Grok 無法生成回應。"
    # 捕獲 OpenAI SDK 可能拋出的特定錯誤
    except openai.APIConnectionError as e:
        logger.error(f"Grok API connection error: {e}", exc_info=True)
        raise fastapi.HTTPException(status_code=503, detail="AI Service connection error (Grok).")
    except openai.RateLimitError as e:
        logger.error(f"Grok API rate limit exceeded: {e}", exc_info=True)
        raise fastapi.HTTPException(status_code=429, detail="AI Service rate limit exceeded (Grok).")
    except openai.AuthenticationError as e:
        logger.error(f"Grok API authentication error (check API Key?): {e}", exc_info=True)
        raise fastapi.HTTPException(status_code=401, detail="AI Service authentication failed (Grok).")
    except openai.APIError as e: # 其他 API 層級錯誤
        logger.error(f"Grok API returned an error: {e}", exc_info=True)
        raise fastapi.HTTPException(status_code=e.status_code or 500, detail=f"AI Service error (Grok): {str(e)}")
    except Exception as e: # 捕獲其他未預料的錯誤
        logger.error(f"Unexpected error calling Grok LLM via OpenAI SDK: {e}", exc_info=True)
        raise fastapi.HTTPException(status_code=500, detail="Internal server error during AI processing.")


async def generate_image_grok(prompt: str) -> str:
    """使用配置好的 OpenAI SDK 調用 Grok 圖像生成 API"""
    try:
        logger.info(f"Calling Grok Image Generation via OpenAI SDK with prompt: {prompt[:100]}...")

        # 使用 OpenAI SDK 的 images.generate 方法
        # 注意：OpenAI SDK v1+ 是同步的，在異步函數中需要用 asyncio.to_thread
        response = await asyncio.to_thread(
            openai_client.images.generate,
            model="grok-2-image-1212",  # <-- 使用文件中提到的目前唯一可用模型，或 "grok-2-image"
            prompt=prompt,
            n=1,                      # 生成圖片數量
            size="1024x1024",         # OpenAI SDK images.generate 常用的尺寸格式，確認 Grok 是否支持
                                      # Grok 文件之前提到 "1024x768"，你需要確認 Grok 透過此 SDK 接口支持哪些尺寸
            response_format="url"     # 請求返回圖片 URL，這對 LINE 來說更理想
                                      # 也可以是 "b64_json" 如果你需要 Base64
        )
        logger.info("Grok Image Generation via OpenAI SDK successful.")

        # 解析 OpenAI SDK 標準的圖像回應結構
        if response.data and len(response.data) > 0:
            image_data = response.data[0]
            if image_data.url:
                logger.info(f"Grok Image URL: {image_data.url}")
                return image_data.url
            elif image_data.b64_json: # 如果請求的是 b64_json
                logger.info("Grok Image returned Base64 data.")
                return f"data:image/png;base64,{image_data.b64_json}" # 假設是 PNG
            else:
                logger.error("Grok Image response (via OpenAI SDK) missing URL or B64_JSON.")
                raise fastapi.HTTPException(status_code=500, detail="AI Service (Grok Image) response format error.")
        else:
            logger.warning("Grok Image response (via OpenAI SDK) did not contain expected data.")
            raise fastapi.HTTPException(status_code=500, detail="AI Service (Grok Image) returned no data.")

    # 捕獲 OpenAI SDK 可能拋出的特定錯誤
    except openai.APIConnectionError as e:
        logger.error(f"Grok Image API connection error: {e}", exc_info=True)
        raise fastapi.HTTPException(status_code=503, detail="AI Service connection error (Grok Image).")
    except openai.RateLimitError as e:
        logger.error(f"Grok Image API rate limit exceeded: {e}", exc_info=True)
        raise fastapi.HTTPException(status_code=429, detail="AI Service rate limit exceeded (Grok Image).")
    except openai.AuthenticationError as e: # API Key 問題
        logger.error(f"Grok Image API authentication error: {e}", exc_info=True)
        raise fastapi.HTTPException(status_code=401, detail="AI Service authentication failed (Grok Image).")
    except openai.BadRequestError as e: # 例如模型不支持或參數錯誤
        logger.error(f"Grok Image API bad request (check model/params?): {e}", exc_info=True)
        detail_msg = "AI Service (Grok Image) bad request."
        if e.body and "message" in e.body: # OpenAI SDK 錯誤通常在 e.body 中
            detail_msg += f" Message: {e.body['message']}"
        raise fastapi.HTTPException(status_code=400, detail=detail_msg)
    except openai.APIError as e: # 其他 API 層級錯誤 (例如 404 如果模型或端點仍然不對)
        logger.error(f"Grok Image API returned an error: {e}", exc_info=True)
        detail_msg = f"AI Service error (Grok Image): {str(e)}"
        if e.body and "message" in e.body:
            detail_msg += f" Message: {e.body['message']}"
        raise fastapi.HTTPException(status_code=e.status_code or 500, detail=detail_msg)
    except Exception as e: # 捕獲其他未預料的錯誤
        logger.error(f"Unexpected error calling Grok Image via OpenAI SDK: {e}", exc_info=True)
        raise fastapi.HTTPException(status_code=500, detail="Internal server error during image generation.")
# --- 7. FastAPI 應用程式實例與路由 (調用新的文字生成函式) ---
app = FastAPI(
    title="N8N Dialog Bot Backend (Grok via OpenAI SDK)",
    description="Handles AI logic (Grok LLM via OpenAI SDK, Grok Image Gen via REST) from n8n.",
    version="1.1.0-grok"
)

@app.get("/", include_in_schema=False)
async def root():
    return {"status": "ok", "message": "N8N Bot Backend (Grok via OpenAI SDK) is running!"}

@app.post("/webhook/chat", response_model=WebhookResponse, tags=["Chatbot"])
async def handle_chat_webhook(request: WebhookRequest):
    logger.info(f"Received request - Platform: {request.platform}, User: {request.user_id}, Type: {request.message_type}, Mention: {request.is_mention}")
    logger.debug(f"Full request payload: {request.dict()}")

    reply_type: Literal["text", "image", "error", "ignore"] = "ignore"
    reply_content: Optional[str] = None

    try:
        if request.platform != 'line':
             raise fastapi.HTTPException(status_code=400, detail="Unsupported platform")

        if request.message_type == "text":
            if not request.message_content:
                 logger.warning("Received text message type but content is empty.")
                 reply_type = "ignore"
            else:
                text_content = request.message_content.strip()
                draw_prefixes = ("畫圖：", "畫圖:", "draw:")
                if text_content.lower().startswith(draw_prefixes):
                     prompt = text_content.split(":", 1)[1].strip()
                     if not prompt:
                          reply_type = "text"
                          reply_content = "請告訴我要畫什麼圖，例如：「畫圖：一隻太空貓」"
                     else:
                          logger.info(f"Detected 'draw' command. Generating image with Grok REST API for: {prompt}")
                          image_output = await generate_image_grok(prompt) # 調用圖像生成函式 (REST)
                          reply_type = "image"
                          reply_content = image_output
                else:
                    logger.info(f"Processing general text message with Grok LLM via OpenAI SDK...")
                    # *** 調用使用 OpenAI SDK 的文字生成函式 ***
                    response_text = await generate_text_grok_via_openai(text_content)
                    reply_type = "text"
                    reply_content = response_text

        elif request.message_type in ["image", "sticker", "unknown"]:
             logger.info(f"Ignoring message type: {request.message_type}")
             reply_type = "ignore"

    except fastapi.HTTPException as http_exc:
         logger.error(f"HTTP Exception occurred: {http_exc.status_code} - {http_exc.detail}", exc_info=True)
         reply_type = "error"
         reply_content = f"處理請求時發生錯誤：{http_exc.detail} (Code: {http_exc.status_code})"
    except Exception as e:
         logger.critical(f"An unexpected error occurred while processing webhook: {e}", exc_info=True)
         reply_type = "error"
         reply_content = "抱歉，伺服器內部發生了未預期的錯誤，請稍後再試。"

    logger.info(f"Responding to n8n - Type: '{reply_type}', Content (first 100 chars): '{reply_content[:100] if reply_content else None}'")
    return WebhookResponse(reply_type=reply_type, reply_content=reply_content)

# --- 8. 本地開發運行器 ---
if __name__ == "__main__":
    logger.info("Running FastAPI app locally using Uvicorn...")
    try:
        import uvicorn
        port = int(os.getenv("PORT", 8000))
        uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
    except ImportError:
         logger.error("Uvicorn not found. Please install it: pip install 'uvicorn[standard]'")
    except Exception as e:
         logging.error(f"Failed to start Uvicorn: {e}")
