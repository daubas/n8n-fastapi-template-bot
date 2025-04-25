# main.py
import importlib.util
import sys
import os
import base64
import logging
from io import BytesIO # 用於處理音檔字節
from typing import Optional, Literal # 用於類型提示

from fastapi import FastAPI, HTTPException, Request # FastAPI 框架及錯誤處理
from pydantic import BaseModel, Field # 用於定義請求和回應的資料結構
from dotenv import load_dotenv # 載入環境變數
import google.auth # 檢查預設憑證
import google.cloud.aiplatform as vertexai # Vertex AI (Gemini, Imagen)
from google.cloud import speech # Speech-to-Text
import requests # 若需從 URL 下載音檔

# --- 0. 基本設定與環境變數載入 ---
load_dotenv()
print("--- Loading environment variables from .env ---")

# 配置基礎日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. 環境檢查 (精簡版，專注關鍵部分) ---
logger.info("--- Starting Environment Checks ---")
critical_missing = False

# 檢查關鍵套件
required_packages = ["fastapi", "pydantic", "google-cloud-aiplatform", "google-cloud-speech", "python-dotenv"]
for pkg in required_packages:
    spec = importlib.util.find_spec(pkg.replace('-', '_')) # pip 名稱轉 import 名稱
    if spec is None:
        logger.error(f"CRITICAL: Required package '{pkg}' not found. Please install it.")
        critical_missing = True

# 檢查關鍵環境變數
required_env_vars = [
    "GOOGLE_APPLICATION_CREDENTIALS", # 或者確認是在有預設服務帳號的環境 (如 Cloud Run)
    "VERTEX_AI_PROJECT_ID",
    "VERTEX_AI_LOCATION",
    "LINE_CHANNEL_ACCESS_TOKEN", # 需要用於 n8n 回覆，但 FastAPI 本身可能不直接用
    "LINE_CHANNEL_SECRET",       # 需要用於 n8n Webhook 驗證，FastAPI 本身不用
]
gac_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
vertex_project = os.getenv("VERTEX_AI_PROJECT_ID")
vertex_location = os.getenv("VERTEX_AI_LOCATION")

if not vertex_project or not vertex_location:
    logger.error("CRITICAL: Environment variables VERTEX_AI_PROJECT_ID or VERTEX_AI_LOCATION are not set.")
    critical_missing = True

# 檢查憑證有效性 (如果設置了 GOOGLE_APPLICATION_CREDENTIALS)
if gac_path:
    if not os.path.exists(gac_path):
        logger.error(f"CRITICAL: GOOGLE_APPLICATION_CREDENTIALS path does not exist: '{gac_path}'")
        critical_missing = True
    else:
         logger.info(f"GOOGLE_APPLICATION_CREDENTIALS path found: '{gac_path}'")
else:
    # 如果沒設置 GAC 路徑，檢查是否能找到預設憑證 (適用於 Cloud Run 等環境)
    try:
        credentials, project_id_from_auth = google.auth.default()
        if credentials:
            logger.info("Google Cloud default credentials found successfully.")
            # 如果環境變數的 project_id 沒設，可以嘗試用預設憑證的
            if not vertex_project and project_id_from_auth:
                 logger.info(f"Using project ID from default credentials: {project_id_from_auth}")
                 vertex_project = project_id_from_auth
                 # 注意：需要確保這個 project_id 正確且已啟用 API
        else:
             logger.warning("Could not find Google Cloud default credentials. Ensure GOOGLE_APPLICATION_CREDENTIALS is set or running in a GCP environment with a service account.")
             # 在本地測試若沒設定 GAC 可能會觸發此警告
    except google.auth.exceptions.DefaultCredentialsError:
        logger.error("CRITICAL: Could not find Google Cloud default credentials. Set GOOGLE_APPLICATION_CREDENTIALS or run in a GCP environment.")
        critical_missing = True

if critical_missing:
    logger.critical("Exiting due to critical environment errors.")
    sys.exit(1) # 環境有嚴重問題，直接退出

# --- 2. 初始化 Google Cloud 客戶端 ---
logger.info("--- Initializing Google Cloud Clients ---")
try:
    vertexai.init(project=vertex_project, location=vertex_location)
    speech_client = speech.SpeechClient()
    # 初始化 Gemini 和 Imagen 模型 (可以延遲到需要時才初始化，或在這裡先載入)
    gemini_model = vertexai.GenerativeModel("gemini-1.5-flash-001")
    imagen_model = vertexai.ImageGenerationModel.from_pretrained("imagegeneration@006") # 使用較新的模型
    logger.info(f"Vertex AI SDK initialized for Project: {vertex_project}, Location: {vertex_location}")
    logger.info("Google Cloud Speech client initialized.")
except Exception as e:
    logger.critical(f"Failed to initialize Google Cloud clients: {e}", exc_info=True)
    sys.exit(1)

# --- 3. 定義 Pydantic 模型 (資料結構驗證) ---
class WebhookRequest(BaseModel):
    platform: str = Field(..., description="來源平台，例如 'line', 'discord'")
    user_id: str = Field(..., description="使用者 ID")
    message_type: Literal["text", "audio", "image", "sticker", "unknown"] = Field(..., description="訊息類型")
    message_content: Optional[str] = Field(None, description="文字訊息內容")
    audio_content_base64: Optional[str] = Field(None, description="Base64 編碼的音檔內容 (若為音檔)")
    audio_format: Optional[str] = Field("webm", description="音檔格式 (例如 'webm', 'm4a', 'wav') - 需要 n8n 傳遞") # 告知STT API格式
    is_mention: bool = Field(False, description="在群組中是否提及機器人")

class WebhookResponse(BaseModel):
    reply_type: Literal["text", "image", "error", "ignore"] = Field(..., description="回應類型")
    reply_content: Optional[str] = Field(None, description="回應內容 (文字或圖片 URL)")

# --- 4. 輔助函式 (呼叫 AI API) ---

async def generate_text_gemini(prompt: str) -> str:
    """使用 Gemini 生成文字"""
    try:
        logger.info(f"Calling Gemini with prompt: {prompt[:100]}...") # 日誌記錄部分提示
        # 注意：Gemini 可能需要更複雜的聊天歷史或設定，這裡簡化處理
        response = await gemini_model.generate_content_async(prompt)
        logger.info("Gemini call successful.")
        return response.text
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Gemini API call failed")

async def generate_image_imagen(prompt: str) -> str:
    """使用 Imagen 生成圖片"""
    try:
        logger.info(f"Calling Imagen with prompt: {prompt[:100]}...")
        # number_of_images=1, aspect_ratio="1:1", safety_filter_level=... 可選參數
        response = await imagen_model.generate_images_async(
            prompt=prompt,
            number_of_images=1
        )
        logger.info("Imagen call successful.")
        # 假設總是成功且至少有一張圖片
        if response.images:
             # 返回第一張圖片的 URL，如果 URL 為 None，則返回錯誤或占位符
             image_url = response.images[0]._image_bytes # 或者 .url 如果模型返回的是 URL
             if image_url:
                 # 如果返回的是 bytes, 需要上傳到 GCS 或轉為 base64 data URI
                 # 這裡假設返回了 bytes，我們將其轉為 base64 data URI (可能非常長)
                 # 注意：LINE 可能不支援過長的 data URI，上傳 GCS 是更好的方案
                 logger.warning("Imagen returned bytes, converting to base64 data URI. Consider uploading to GCS instead.")
                 base64_image = base64.b64encode(image_url).decode('utf-8')
                 # 需要知道圖片格式才能正確組裝 data URI，假設是 PNG
                 # 這部分可能需要根據 Imagen 實際返回調整
                 return f"data:image/png;base64,{base64_image}"
             else:
                logger.error("Imagen API response did not contain image bytes/URL.")
                raise HTTPException(status_code=500, detail="Imagen API did not return image.")

        else:
            logger.error("Imagen API response did not contain any images.")
            raise HTTPException(status_code=500, detail="Imagen API did not return image.")
    except Exception as e:
        logger.error(f"Error calling Imagen API: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Imagen API call failed")

async def transcribe_speech(audio_content_base64: str, language_code: str = "zh-TW", audio_format: str = "webm") -> str:
    """使用 Speech-to-Text 將 Base64 音檔轉文字"""
    if not audio_content_base64:
        raise ValueError("Audio content (Base64) is required.")

    try:
        logger.info(f"Decoding Base64 audio data (format: {audio_format})...")
        audio_bytes = base64.b64decode(audio_content_base64)
        logger.info(f"Decoded audio size: {len(audio_bytes)} bytes")

        # 準備 STT API 請求
        audio = speech.RecognitionAudio(content=audio_bytes)

        # --- 設定辨識配置 ---
        # 需要根據 n8n 傳來的 audio_format 選擇正確的 encoding
        # 這裡做一些常見格式的映射，需要根據實際情況調整
        encoding_map = {
            "wav": speech.RecognitionConfig.AudioEncoding.LINEAR16,
            "flac": speech.RecognitionConfig.AudioEncoding.FLAC,
            "mp3": speech.RecognitionConfig.AudioEncoding.MP3,
            "ogg": speech.RecognitionConfig.AudioEncoding.OGG_OPUS, # Ogg Opus
            "webm": speech.RecognitionConfig.AudioEncoding.WEBM_OPUS, # WebM Opus
            # M4A 通常需要轉換，Speech-to-Text 可能不直接支持
            # "m4a": ??? 需要轉換
        }
        encoding = encoding_map.get(audio_format.lower())
        if not encoding:
             logger.error(f"Unsupported audio format for STT: {audio_format}. Falling back to ENCODING_UNSPECIFIED.")
             # 可以嘗試讓 API 自動檢測，但不一定成功
             encoding = speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED
             # 或者直接報錯
             # raise ValueError(f"Unsupported audio format for STT: {audio_format}")


        config = speech.RecognitionConfig(
            encoding=encoding,
            # sample_rate_hertz=16000, # 對於某些格式如 LINEAR16 需要指定，Opus/MP3 通常不需要
            language_code=language_code,
            enable_automatic_punctuation=True, # 自動加標點
        )

        logger.info(f"Calling Speech-to-Text API with language: {language_code}, encoding: {encoding.name}")
        response = await speech_client.recognize(config=config, audio=audio) # 使用異步客戶端需調整
        # 注意：原生 google-cloud-speech 沒有內建的 async client，
        # 如果需要完全異步，可能要用 google.api_core.gapic_v1.client_async 或者 asyncio.to_thread
        # 這裡為了簡單，我們先用同步的方式，但在 async def 中調用
        # 改為使用 asyncio.to_thread 模擬異步調用同步庫
        import asyncio
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, speech_client.recognize, config, audio)


        logger.info("Speech-to-Text call successful.")

        # 處理回應結果
        transcript = ""
        if response.results:
            transcript = response.results[0].alternatives[0].transcript
            logger.info(f"Transcription result: {transcript}")
        else:
            logger.warning("Speech-to-Text API returned no results.")

        return transcript

    except base64.binascii.Error as e:
         logger.error(f"Error decoding Base64 audio data: {e}", exc_info=True)
         raise HTTPException(status_code=400, detail="Invalid Base64 audio data")
    except Exception as e:
        logger.error(f"Error calling Speech-to-Text API: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Speech-to-Text API call failed")


# --- 5. FastAPI 應用程式實例 ---
app = FastAPI(
    title="N8N Dialog Bot Backend",
    description="Handles AI logic (Gemini, Imagen, STT) via API calls from n8n.",
    version="0.1.0",
    # 添加根路徑，用於健康檢查或基本信息
    root_path="/api/v1" # 假設 API 部署在 /api/v1 路徑下 (可選)
)
logger.info("FastAPI instance created.")


# --- 6. API 端點 (Endpoint) ---

@app.get("/", include_in_schema=False) # 根目錄健康檢查，不顯示在 OpenAPI 文件中
async def root():
    return {"status": "ok", "message": "Welcome to the N8N Bot Backend!"}

@app.post("/webhook/chat", response_model=WebhookResponse, tags=["Chat Webhook"])
async def handle_chat_webhook(request: WebhookRequest):
    """
    接收來自 n8n 的 Webhook 請求，處理訊息並返回 AI 生成的回應。
    """
    logger.info(f"Received request from platform '{request.platform}' for user '{request.user_id}'. Message type: {request.message_type}")
    logger.debug(f"Full request data: {request.dict()}") # Debug 級別記錄完整請求

    reply_type: Literal["text", "image", "error", "ignore"] = "ignore"
    reply_content: Optional[str] = None

    try:
        # --- 平台適配 (目前只有 LINE) ---
        if request.platform != 'line':
             logger.warning(f"Received request from unsupported platform: {request.platform}")
             # 可以選擇忽略或返回錯誤
             # return WebhookResponse(reply_type="ignore", reply_content=None)
             raise HTTPException(status_code=400, detail="Unsupported platform")

        # --- 訊息類型處理 ---
        if request.message_type == "text":
            if not request.message_content:
                 logger.warning("Received text message type but content is empty.")
                 reply_type = "ignore"
            # 檢查是否需要提及 (如果 is_mention 是 False，可以選擇忽略，這裡假設 n8n 已過濾)
            # elif not request.is_mention:
            #     logger.info("Ignoring text message because it's not a mention.")
            #     reply_type = "ignore"
            else:
                text_content = request.message_content.strip()
                # 檢查是否為畫圖指令
                if text_content.startswith("畫圖：") or text_content.startswith("畫圖:") or text_content.lower().startswith("draw:"):
                     prompt = text_content.split(":", 1)[1].strip()
                     if not prompt:
                          reply_type = "text"
                          reply_content = "請告訴我要畫什麼圖，例如：畫圖：一隻貓"
                     else:
                          image_url_or_data = await generate_image_imagen(prompt)
                          reply_type = "image"
                          reply_content = image_url_or_data
                else:
                    # 一般文字訊息，呼叫 Gemini
                    response_text = await generate_text_gemini(text_content)
                    reply_type = "text"
                    reply_content = response_text

        elif request.message_type == "audio":
            if not request.audio_content_base64:
                logger.warning("Received audio message type but audio content is missing.")
                reply_type = "ignore"
            else:
                 # 呼叫 STT
                 transcribed_text = await transcribe_speech(
                     request.audio_content_base64,
                     audio_format=request.audio_format or "webm" # 提供預設值
                 )

                 if transcribed_text:
                     # 將辨識出的文字交給 Gemini
                     logger.info(f"Passing transcribed text to Gemini: {transcribed_text}")
                     response_text = await generate_text_gemini(transcribed_text)
                     reply_type = "text"
                     reply_content = response_text
                 else:
                     # STT 沒有結果
                     reply_type = "text"
                     reply_content = "抱歉，我聽不清楚您說什麼。"

        # 其他訊息類型 (圖片、貼圖等) 暫時忽略
        elif request.message_type in ["image", "sticker", "unknown"]:
             logger.info(f"Ignoring message type: {request.message_type}")
             reply_type = "ignore"

    except HTTPException as http_exc:
         # 捕獲由輔助函式或邏輯中拋出的 HTTPException
         logger.error(f"HTTP Exception occurred: {http_exc.status_code} - {http_exc.detail}")
         reply_type = "error"
         reply_content = f"處理請求時發生錯誤：{http_exc.detail}"
         # FastAPI 會自動處理 HTTPException 並返回對應的狀態碼，但我們這裡包裝成標準回應給 n8n

    except Exception as e:
         # 捕獲未預料的錯誤
         logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
         reply_type = "error"
         reply_content = "抱歉，發生了未預期的內部錯誤。"
         # 這裡不直接拋出 500 給客戶端，而是返回一個錯誤訊息給 n8n

    logger.info(f"Prepared response: Type='{reply_type}', Content='{reply_content[:100] if reply_content else None}'")
    return WebhookResponse(reply_type=reply_type, reply_content=reply_content)


# --- 7. 本地測試運行器 ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000)) # Cloud Run 會設置 PORT 環境變數
    logger.info(f"--- Starting Uvicorn Server locally on http://0.0.0.0:{port} ---")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
    # 注意：生產環境部署時， reload 應設為 False 或移除
    # 生產環境通常使用 Gunicorn + Uvicorn workers
