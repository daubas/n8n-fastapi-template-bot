# check_env.py
import importlib.util
import sys
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import google.auth

# --- Configuration ---
REQUIREMENTS_FILE = "requirements.txt"
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Mapping from pip install name (or part of it) to Python import name ---
# This might need adjustment based on specific packages
PACKAGE_TO_MODULE_MAP = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn", # Matches uvicorn[standard]
    "requests": "requests",
    "google-cloud-aiplatform": "google.cloud.aiplatform",
    "google-cloud-speech": "google.cloud.speech",
    "python-dotenv": "dotenv", # Checks the core import name
    "line-bot-sdk": "linebot",
    "python-multipart": "multipart",
    "setuptools": "setuptools",
    "pytest": "pytest",
    "pytest-mock": "pytest_mock", # Note the underscore
    "httpx": "httpx",
    "gunicorn": "gunicorn",
}

# --- Functions ---

def check_packages(requirements_path: str):
    """Reads requirements.txt and checks if packages are importable."""
    installed_count = 0
    missing_packages = []
    packages_checked = set() # Avoid checking duplicates in requirements.txt

    logging.info(f"--- Checking Packages listed in {requirements_path} ---")
    req_file = Path(requirements_path)
    if not req_file.is_file():
        logging.error(f"'{requirements_path}' not found!")
        return False, 0

    try:
        with open(req_file, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        logging.error(f"Error reading '{requirements_path}': {e}")
        return False, 0

    if not lines:
        logging.warning(f"'{requirements_path}' is empty.")
        return True, 0 # Technically no missing packages if file is empty

    for line_num, line in enumerate(lines):
        line = line.strip()
        # Skip empty lines, comments, and editable installs for simplicity
        if not line or line.startswith('#') or line.startswith('-e'):
            continue

        # Basic parsing: get the package name before version specifiers like ==, >=, [
        package_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('<')[0].split('>')[0].split('~=')[0].split('!=')[0].split('[')[0].strip()

        if not package_name or package_name in packages_checked:
            continue # Skip if empty name after parsing or already checked
        packages_checked.add(package_name)

        # Find the corresponding module name from our map
        module_name = None
        # Try direct match first, then partial match (e.g., "uvicorn" in "uvicorn[standard]")
        if package_name in PACKAGE_TO_MODULE_MAP:
             module_name = PACKAGE_TO_MODULE_MAP[package_name]
        else:
             # Handle cases like fastapi[all] -> fastapi
             base_package = package_name.split('[')[0]
             if base_package in PACKAGE_TO_MODULE_MAP:
                 module_name = PACKAGE_TO_MODULE_MAP[base_package]
             else:
                 logging.warning(f" No import name mapped for '{package_name}' in requirements.txt (line {line_num+1}). Skipping check.")
                 continue # Skip if we don't know what module to import

        logging.info(f" Checking: '{package_name}' (via import '{module_name}')")
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            logging.error(f"  -> MISSING! Package '{package_name}' (module '{module_name}') not found.")
            missing_packages.append(package_name)
        else:
            logging.info(f"  -> Found.")
            installed_count += 1

    logging.info("--- Package Check Summary ---")
    if missing_packages:
        logging.error(f" Found {installed_count} out of {len(packages_checked)} required packages.")
        logging.error(f" Missing packages: {', '.join(missing_packages)}")
        logging.error(f" Please run 'pip install -r {requirements_path}' in your environment.")
        return False, installed_count
    else:
        logging.info(f" OK: All {installed_count} checked packages appear to be installed.")
        return True, installed_count

def check_credentials():
    """Checks the Google Cloud credentials status."""
    logging.info("--- Checking Google Cloud Credentials ---")

    # 1. Check GOOGLE_APPLICATION_CREDENTIALS environment variable
    gac_env_var = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if gac_env_var:
        logging.info(f" Environment variable 'GOOGLE_APPLICATION_CREDENTIALS' is SET.")
        gac_path = Path(gac_env_var)
        # Check if it looks like a path or JSON content
        if '{' in gac_env_var and '}' in gac_env_var and '"type":' in gac_env_var:
             logging.info(f"   -> Value looks like inline JSON content (common for Zeabur/Cloud Run env vars).")
             # We won't validate the JSON here, assume it's correct if set this way
        elif gac_path.is_file():
            logging.info(f"   -> Value points to an EXISTING file: '{gac_path}'")
        else:
            logging.warning(f"   -> Value points to a NON-EXISTENT file/path: '{gac_env_var}'")
            # This is usually an error unless the value is meant to be JSON content
            # but doesn't look like it based on the basic check above.
    else:
        logging.warning(" Environment variable 'GOOGLE_APPLICATION_CREDENTIALS' is NOT SET.")
        logging.warning("  -> Relying on Application Default Credentials (ADC) search order.")

    # 2. Try using Application Default Credentials (ADC)
    logging.info(" Attempting to find credentials using google.auth.default()...")
    try:
        credentials, project_id = google.auth.default()
        if credentials:
            logging.info("  -> SUCCESS: google.auth.default() found credentials.")
            if project_id:
                logging.info(f"     -> Associated Project ID found: {project_id}")
            else:
                logging.warning("     -> Credentials found, but no specific Project ID associated.")
        else:
            # This case should technically not happen if google.auth.default() doesn't raise error
             logging.error("  -> UNEXPECTED: google.auth.default() returned None for credentials without error.")
             return False

    except google.auth.exceptions.DefaultCredentialsError as e:
        logging.error(f"  -> FAILED: google.auth.default() could not find credentials.")
        logging.error(f"     -> Error details: {e}")
        logging.error(f"     -> Ensure GOOGLE_APPLICATION_CREDENTIALS is set correctly OR you are running in an environment with implicit credentials (e.g., Cloud Run with service account).")
        return False
    except Exception as e:
        logging.error(f"  -> UNEXPECTED ERROR during google.auth.default(): {e}")
        return False

    return True


# --- Main Execution ---
if __name__ == "__main__":
    print(f"Running Environment Checks...")
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")

    # Attempt to load .env file if it exists (useful for local checks)
    if Path(".env").is_file():
        logging.info("Found .env file, attempting to load environment variables.")
        load_dotenv()
    else:
        logging.info("No .env file found in the current directory.")

    packages_ok, _ = check_packages(REQUIREMENTS_FILE)
    credentials_ok = check_credentials()

    print("\n--- Final Summary ---")
    if packages_ok and credentials_ok:
        print("✅ Environment check passed successfully!")
    else:
        print("❌ Environment check failed. Please review the logs above.")
        if not packages_ok:
             print(f"   - Issue: Missing Python packages. Run 'pip install -r {REQUIREMENTS_FILE}'.")
        if not credentials_ok:
             print("   - Issue: Google Cloud credentials could not be found or verified.")
