from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import boto3
import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")  # Default to 'us-east-1' if not set
SAGEMAKER_ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT_NAME", "mySageMakerEndpoint")  # Replace with your endpoint name

# Initialize the SageMaker Runtime client
sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)

# Create FastAPI instance
app = FastAPI(title="SageMaker Integration API", description="API to interact with SageMaker endpoint", version="1.0.0")


@app.get("/", summary="Health Check")
def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    logger.info("Health check called.")
    return {"message": "Service is running"}


@app.post("/predict", summary="Make Prediction")
async def predict(request: Request):
    """
    Handles prediction requests by forwarding them to the SageMaker endpoint.
    """
    try:
        # Parse JSON request body
        request_data = await request.json()

        # Validate input
        if "data" not in request_data:
            logger.error("Missing 'data' in request payload.")
            raise HTTPException(status_code=400, detail="Missing 'data' in request payload.")

        # Log the received request
        logger.info(f"Received data for prediction: {request_data['data']}")

        # Send input data to SageMaker endpoint
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT_NAME,
            Body=json.dumps({"instances": request_data["data"]}),
            ContentType="application/json"
        )

        # Parse SageMaker response
        result = json.loads(response["Body"].read())
        logger.info(f"Prediction result: {result}")

        return {"prediction": result}

    except boto3.exceptions.Boto3Error as boto3_error:
        logger.error(f"SageMaker invocation error: {boto3_error}")
        raise HTTPException(status_code=500, detail="Error invoking SageMaker endpoint.")

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Custom handler for HTTP exceptions.
    """
    logger.error(f"HTTP error occurred: {exc.detail}")
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
