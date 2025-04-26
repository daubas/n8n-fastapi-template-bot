# main.py

# --- 0. 標準函式庫和日誌設定 ---
import importlib.util
import sys
import os
import logging
from pathlib import Path
import json
import tempfile
import base64 # <--- 新增：用於處理音檔 Base64
import asyncio # <--- 新增：用於異步處理
from typing import Optional, Literal # FastAPI 可能需要

# --- 1. 檢查所需的第三方函式庫 ---
try:
    from dotenv import load_dotenv
    import google.auth
    # 導入其他核心依賴，確保它們存在
    import fastapi
    from pydantic import BaseModel, Field # Pydantic 用於數據驗證
    import google.cloud.aiplatform as vertexai # Vertex AI SDK
    from google.cloud import speech # Speech-to-Text SDK
    import requests # 可能需要下載音檔 (如果 n8n 不傳 base64)
except ImportError as e:
     # 如果核心依賴缺失，無法繼續
     print(f"FATAL: Core application dependencies missing: {e}. Please ensure all packages in requirements.txt are installed.", file=sys.stderr)
     sys.exit(1)

# 設定基礎日誌 (盡早設定)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(asctime)s - %(message)s')
logger = logging.getLogger(__name__) # 使用具名 logger

# --- 2. 環境檢查邏輯 (來自 check_env.py) ---
REQUIREMENTS_FILE = "requirements.txt"
PACKAGE_TO_MODULE_MAP = {
    "fastapi": "fastapi", "uvicorn": "uvicorn", "requests": "requests",
    "google-cloud-aiplatform": "google.cloud.aiplatform",
    "google-cloud-speech": "google.cloud.speech", "python-dotenv": "dotenv",
    "line-bot-sdk": "linebot", "python-multipart": "multipart",
    "setuptools": "setuptools", "pytest": "pytest", "pytest-mock": "pytest_mock",
    "httpx": "httpx", "gunicorn": "gunicorn",
}

def check_packages(requirements_path: str):
    # ... (之前的 check_packages 函式代碼保持不變) ...
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


def check_and_setup_credentials():
    # ... (之前的 check_and_setup_credentials 函式代碼保持不變) ...
    logger.info("--- Checking and Setting Up Google Cloud Credentials ---")
    gac_env_var_content = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not gac_env_var_content: logger.error(" Environment variable 'GOOGLE_APPLICATION_CREDENTIALS' is NOT SET."); logger.error("  -> Cannot proceed without credentials."); return False
    logger.info(" Environment variable 'GOOGLE_APPLICATION_CREDENTIALS' is SET.")
    try: json.loads(gac_env_var_content); logger.info("  -> Value appears to be valid JSON content.")
    except json.JSONDecodeError as e: logger.error(f"  -> Value is NOT valid JSON content: {e}"); logger.error("     Please ensure the environment variable contains the exact, complete JSON key content."); return False
    try:
        creds_dir = Path(tempfile.gettempdir()); creds_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, dir=creds_dir) as temp_creds_file:
             temp_creds_file.write(gac_env_var_content); temp_file_path = temp_creds_file.name
             logger.info(f"  -> Successfully wrote JSON content to temporary file: {temp_file_path}")
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_file_path
        logger.info(f"  -> Environment variable 'GOOGLE_APPLICATION_CREDENTIALS' now points to: {temp_file_path}")
        logging.info(" Re-attempting to find credentials using google.auth.default() with file path...")
        credentials, project_id = google.auth.default()
        if credentials:
             logger.info("  -> SUCCESS: google.auth.default() found credentials using the temporary file.")
             if project_id: logger.info(f"     -> Associated Project ID found: {project_id}")
             return True
        else: logger.error("  -> UNEXPECTED: google.auth.default() returned None even after writing to temp file."); return False
    except google.auth.exceptions.DefaultCredentialsError as e: logger.error(f"  -> FAILED: google.auth.default() could not find credentials even with temp file: {e}"); return False
    except OSError as e: logger.error(f"  -> FAILED: Could not write temporary credentials file (check permissions for /tmp?): {e}"); return False
    except Exception as e: logger.error(f"  -> UNEXPECTED ERROR during credential setup: {e}"); return False


# --- 3. 執行環境檢查 (在腳本加載時立即執行) ---
logging.info("====== Running Pre-Startup Environment Checks ======")
env_path = Path('.env')
if env_path.is_file(): logging.info("Found .env file, loading variables."); load_dotenv()
else: logging.info("No .env file found.")

packages_ok = check_packages(REQUIREMENTS_FILE)
credentials_ok = check_and_setup_credentials()

# 在這裡獲取 Project ID 和 Location 以便後續使用
GCP_PROJECT_ID = os.getenv("VERTEX_AI_PROJECT_ID")
GCP_LOCATION = os.getenv("VERTEX_AI_LOCATION")
if not GCP_PROJECT_ID or not GCP_LOCATION:
    logging.error("CRITICAL: Environment variables VERTEX_AI_PROJECT_ID or VERTEX_AI_LOCATION are not set.")
    credentials_ok = False # 標記為檢查失敗

if not packages_ok or not credentials_ok:
    logging.critical("====== Environment checks FAILED. Application will NOT start. ======")
    logging.critical("Please review the logs above to fix the issues.")
    sys.exit(1)

logging.info("====== Environment Checks Passed. Proceeding with Initialization ======")

# --- 4. 初始化 Google Cloud 客戶端 (檢查通過後) ---
try:
    logging.info(f"Initializing Vertex AI for Project: {GCP_PROJECT_ID}, Location: {GCP_LOCATION}...")
    vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
    speech_client = speech.SpeechClient()
    # 初始化具體的模型 (可以按需調整模型名稱)
    gemini_model = vertexai.GenerativeModel("gemini-1.5-flash-001")
    # 注意：Imagen 模型名稱可能需要更新，請參考 Google Cloud 文檔
    # imagegeneration@006 是較新的選項
    imagen_model = vertexai.ImageGenerationModel.from_pretrained("imagegeneration@006")
    logging.info("Vertex AI and Speech clients initialized successfully.")
except Exception as e:
    logging.critical(f"Failed to initialize Google Cloud clients: {e}", exc_info=True)
    sys.exit(1) # 初始化失敗也應阻止啟動

# --- 5. 定義 Pydantic 模型 (請求與回應結構) ---
class WebhookRequest(BaseModel):
    platform: str = Field(..., description="來源平台，例如 'line', 'discord'")
    user_id: str = Field(..., description="使用者 ID")
    message_type: Literal["text", "audio", "image", "sticker", "unknown"] = Field(..., description="訊息類型")
    message_content: Optional[str] = Field(None, description="文字訊息內容")
    audio_content_base64: Optional[str] = Field(None, description="Base64 編碼的音檔內容 (若為音檔)")
    audio_format: Optional[str] = Field("webm", description="音檔格式 (n8n 需傳遞，例如 'webm', 'm4a', 'wav')")
    is_mention: bool = Field(False, description="在群組中是否提及機器人")

class WebhookResponse(BaseModel):
    reply_type: Literal["text", "image", "error", "ignore"] = Field(..., description="回應類型")
    reply_content: Optional[str] = Field(None, description="回應內容 (文字或圖片 URL/Data URI)")

# --- 6. AI 輔助函式 ---
async def generate_text_gemini(prompt: str) -> str:
    """使用 Gemini 生成文字"""
    try:
        logger.info(f"Calling Gemini with prompt (first 100 chars): {prompt[:100]}...")
        # 注意：更複雜的應用可能需要處理聊天歷史 (contents=[])
        response = await gemini_model.generate_content_async(prompt)
        logger.info("Gemini call successful.")
        # 檢查是否有文字回應
        if response.candidates and response.candidates[0].content.parts:
             return response.candidates[0].content.parts[0].text
        else:
             logger.warning("Gemini response did not contain expected text content.")
             return "抱歉，我無法生成回應。" # 或者拋出錯誤
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}", exc_info=True)
        # 在實際應用中可能需要更細緻的錯誤處理
        raise fastapi.HTTPException(status_code=503, detail="AI Service (Gemini) unavailable or failed.")

async def generate_image_imagen(prompt: str) -> str:
    """使用 Imagen 生成圖片，返回 Base64 Data URI 或錯誤"""
    try:
        logger.info(f"Calling Imagen with prompt: {prompt[:100]}...")
        # 可以添加更多參數，如圖片數量、尺寸、風格等
        response = await imagen_model.generate_images_async(
            prompt=prompt,
            number_of_images=1
        )
        logger.info("Imagen call successful.")
        if response.images:
            # Imagen SDK 可能返回 _image_bytes
            image_bytes = response.images[0]._image_bytes
            if image_bytes:
                 # 將 bytes 轉換為 Base64 Data URI (假設 PNG 格式)
                 # 注意：對於大型圖片，這會使回應非常大，上傳到 GCS 返回 URL 是更好的方案
                 logger.info(f"Imagen returned image bytes ({len(image_bytes)} bytes). Converting to Base64 Data URI.")
                 base64_image = base64.b64encode(image_bytes).decode('utf-8')
                 # 你可能需要根據實際返回的圖片類型調整 'image/png'
                 return f"data:image/png;base64,{base64_image}"
            else:
                 logger.error("Imagen API response did not contain image bytes.")
                 raise fastapi.HTTPException(status_code=500, detail="Imagen generation failed (no image data).")
        else:
            logger.error("Imagen API response did not contain any images.")
            raise fastapi.HTTPException(status_code=500, detail="Imagen generation failed (no images returned).")
    except Exception as e:
        logger.error(f"Error calling Imagen API: {e}", exc_info=True)
        raise fastapi.HTTPException(status_code=503, detail="AI Service (Imagen) unavailable or failed.")


async def transcribe_speech(audio_content_base64: str, language_code: str = "zh-TW", audio_format: str = "webm") -> str:
    """使用 Speech-to-Text 將 Base64 音檔轉文字"""
    if not audio_content_base64:
        raise ValueError("Audio content (Base64) is required for transcription.")

    try:
        logger.info(f"Decoding Base64 audio data (format: {audio_format})...")
        audio_bytes = base64.b64decode(audio_content_base64)
        logger.info(f"Decoded audio size: {len(audio_bytes)} bytes")
        audio = speech.RecognitionAudio(content=audio_bytes)

        # 根據 audio_format 選擇 encoding
        encoding_map = {
            "wav": speech.RecognitionConfig.AudioEncoding.LINEAR16,
            "flac": speech.RecognitionConfig.AudioEncoding.FLAC,
            "mp3": speech.RecognitionConfig.AudioEncoding.MP3,
            "ogg": speech.RecognitionConfig.AudioEncoding.OGG_OPUS,
            "webm": speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            "amr": speech.RecognitionConfig.AudioEncoding.AMR, # 某些手機格式
            "awb": speech.RecognitionConfig.AudioEncoding.AMR_WB,
            # M4A/AAC 通常需要轉換或使用 specific recognition features
        }
        encoding = encoding_map.get(audio_format.lower())
        if not encoding:
            logger.warning(f"Unsupported audio format '{audio_format}' for STT direct encoding. Trying unspecified...")
            encoding = speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED
            # 或者報錯：raise ValueError(f"Unsupported audio format: {audio_format}")

        config = speech.RecognitionConfig(
            encoding=encoding,
            language_code=language_code,
            enable_automatic_punctuation=True,
            # sample_rate_hertz=16000, # 對 LINEAR16 等可能需要
            # model='telephony', # 或 'medical_dictation' 等，根據場景選擇
        )

        logger.info(f"Calling Speech-to-Text API (Lang: {language_code}, Encoding: {encoding.name if encoding else 'Unspecified'})...")
        # 使用 asyncio.to_thread 在異步環境中運行同步的 SDK 調用
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, speech_client.recognize, config, audio)
        logger.info("Speech-to-Text call successful.")

        transcript = ""
        if response.results:
            transcript = response.results[0].alternatives[0].transcript
            logger.info(f"Transcription result: {transcript}")
        else:
            logger.warning("Speech-to-Text API returned no results.")
        return transcript

    except base64.binascii.Error as e:
         logger.error(f"Error decoding Base64 audio data: {e}", exc_info=True)
         raise fastapi.HTTPException(status_code=400, detail="Invalid Base64 audio data provided.")
    except Exception as e:
        logger.error(f"Error calling Speech-to-Text API: {e}", exc_info=True)
        raise fastapi.HTTPException(status_code=503, detail="AI Service (Speech-to-Text) unavailable or failed.")

# --- 7. FastAPI 應用程式實例與路由 ---
app = FastAPI(
    title="N8N Dialog Bot Backend",
    description="Handles AI logic (Gemini, Imagen, STT) via API calls from n8n.",
    version="1.0.0" # 更新版本號
)

@app.get("/", include_in_schema=False) # 根目錄健康檢查
async def root():
    return {
        "status": "ok",
        "message": "N8N Bot Backend is running!",
        "gac_path": os.getenv("GOOGLE_APPLICATION_CREDENTIALS") # 顯示目前 GAC 指向
    }

# 主要的 Webhook 端點
@app.post("/webhook/chat", response_model=WebhookResponse, tags=["Chatbot"])
async def handle_chat_webhook(request: WebhookRequest):
    """
    接收來自 n8n 的 Webhook 請求，處理訊息並返回 AI 生成的回應。
    n8n 需要發送包含 platform, user_id, message_type, message_content/audio_content_base64, audio_format, is_mention 的 JSON body。
    """
    logger.info(f"Received request - Platform: {request.platform}, User: {request.user_id}, Type: {request.message_type}, Mention: {request.is_mention}")
    logger.debug(f"Full request payload: {request.dict()}")

    reply_type: Literal["text", "image", "error", "ignore"] = "ignore"
    reply_content: Optional[str] = None

    try:
        # --- 平台過濾 (可選) ---
        if request.platform != 'line':
             logger.warning(f"Received request from unsupported platform: {request.platform}")
             # return WebhookResponse(reply_type="ignore", reply_content="Unsupported platform") # 直接忽略
             raise fastapi.HTTPException(status_code=400, detail="Unsupported platform")

        # --- 訊息類型處理 ---
        if request.message_type == "text":
            if not request.message_content:
                 logger.warning("Received text message type but content is empty.")
                 reply_type = "ignore"
            # 在群組中，如果沒被提及，可以選擇忽略 (取決於 n8n 是否已過濾)
            # elif not request.is_mention:
            #     logger.info("Ignoring text message (not a mention in group).")
            #     reply_type = "ignore"
            else:
                text_content = request.message_content.strip()
                # 檢查畫圖指令
                draw_prefixes = ("畫圖：", "畫圖:", "draw:")
                if text_content.lower().startswith(draw_prefixes):
                     prompt = text_content.split(":", 1)[1].strip()
                     if not prompt:
                          reply_type = "text"
                          reply_content = "請告訴我要畫什麼圖，例如：「畫圖：一隻太空貓」"
                     else:
                          logger.info(f"Detected 'draw' command. Generating image for: {prompt}")
                          image_output = await generate_image_imagen(prompt)
                          reply_type = "image"
                          reply_content = image_output
                else:
                    # 一般文字訊息，調用 Gemini
                    logger.info(f"Processing general text message...")
                    response_text = await generate_text_gemini(text_content)
                    reply_type = "text"
                    reply_content = response_text

        elif request.message_type == "audio":
            if not request.audio_content_base64:
                logger.warning("Received audio message type but audio content is missing.")
                reply_type = "ignore"
            else:
                 logger.info(f"Processing audio message (Format: {request.audio_format or 'default webm'})...")
                 transcribed_text = await transcribe_speech(
                     request.audio_content_base64,
                     audio_format=request.audio_format or "webm" # 提供預設值
                 )
                 if transcribed_text:
                     logger.info(f"Audio transcribed. Passing text to Gemini: '{transcribed_text}'")
                     response_text = await generate_text_gemini(transcribed_text)
                     reply_type = "text"
                     reply_content = response_text
                 else:
                     logger.warning("Transcription resulted in empty text.")
                     reply_type = "text"
                     reply_content = "抱歉，我聽不清楚您說什麼，或者無法處理這段語音。"

        # 其他訊息類型 (image, sticker, unknown) 暫時忽略
        elif request.message_type in ["image", "sticker", "unknown"]:
             logger.info(f"Ignoring message type: {request.message_type}")
             reply_type = "ignore"

    except fastapi.HTTPException as http_exc:
         # 捕獲由輔助函式或邏輯中拋出的 HTTP 錯誤
         logger.error(f"HTTP Exception occurred: {http_exc.status_code} - {http_exc.detail}", exc_info=True)
         reply_type = "error"
         reply_content = f"處理請求時發生錯誤：{http_exc.detail} (Code: {http_exc.status_code})"
         # FastAPI 會自動處理狀態碼，但我們在這裡包裝成標準回應給 n8n

    except Exception as e:
         # 捕獲未預料的錯誤
         logger.critical(f"An unexpected error occurred while processing webhook: {e}", exc_info=True)
         reply_type = "error"
         reply_content = "抱歉，伺服器內部發生了未預期的錯誤，請稍後再試。"
         # 這裡不直接拋出 500 給客戶端，而是返回一個錯誤訊息給 n8n

    logger.info(f"Responding to n8n - Type: '{reply_type}', Content (first 100 chars): '{reply_content[:100] if reply_content else None}'")
    return WebhookResponse(reply_type=reply_type, reply_content=reply_content)


# --- 8. 本地開發運行器 ---
if __name__ == "__main__":
    # 只有直接運行 python main.py 時才執行
    logger.info("Running FastAPI app locally using Uvicorn...")
    try:
        import uvicorn
        port = int(os.getenv("PORT", 8000))
        # reload=True 對於包含頂層檢查和初始化的代碼可能有副作用，生產不用
        # 如果本地開發需要 reload，可能需要調整檢查邏輯的位置或方式
        uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False) # 改為 False 避免檢查邏輯重複執行問題
    except ImportError:
         logger.error("Uvicorn not found. Please install it: pip install 'uvicorn[standard]'")
    except Exception as e:
         logging.error(f"Failed to start Uvicorn: {e}")
