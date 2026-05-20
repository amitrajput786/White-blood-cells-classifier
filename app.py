import os
import sys
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main FastAPI app
from main import app

# (Optional) Configure logging for Hugging Face Spaces
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# For Hugging Face Spaces, we just need to expose the app.
# The uvicorn server will be started by the platform or Docker CMD.