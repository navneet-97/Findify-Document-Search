import uvicorn
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Import the app after environment variables are set
    from server import app
    
    # Get port from environment variable, default to 10000
    port = int(os.environ.get("PORT", 10000))
    
    # Log startup information
    logger.info(f"Starting server on port {port}")
    logger.info(f"Environment variables: PORT={os.environ.get('PORT', 'Not set')}")
    
    # Run the application
    uvicorn.run(app, host="0.0.0.0", port=port)