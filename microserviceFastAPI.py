import json
import logging
import os
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
SAGEMAKER_ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT_NAME", "mySageMakerEndpoint")


class PredictionRequest(BaseModel):
    """Incoming payload for prediction requests."""

    data: list[Any] = Field(..., min_length=1, description="Input records for model inference")


class PredictionResponse(BaseModel):
    """Outgoing payload that wraps model predictions."""

    prediction: Any


# Initialize the SageMaker Runtime client once at startup
sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)

# Create FastAPI instance
app = FastAPI(
    title="SageMaker Integration API",
    description="API to interact with SageMaker endpoint",
    version="1.1.0",
)


@app.get("/", summary="Health Check")
def health_check() -> dict[str, str]:
    """Health check endpoint to verify the service is running."""
    logger.info("Health check called")
    return {"message": "Service is running"}


@app.post("/predict", summary="Make Prediction", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Forward inference payload to SageMaker and return predictions."""
    logger.info("Received prediction request with %d records", len(request.data))

    try:
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT_NAME,
            Body=json.dumps({"instances": request.data}),
            ContentType="application/json",
            Accept="application/json",
        )

        result = json.loads(response["Body"].read())
        logger.info("Prediction completed successfully")
        return PredictionResponse(prediction=result)

    except (BotoCoreError, ClientError) as aws_error:
        logger.exception("SageMaker invocation failed: %s", aws_error)
        raise HTTPException(status_code=502, detail="Error invoking SageMaker endpoint") from aws_error
    except json.JSONDecodeError as decode_error:
        logger.exception("Invalid JSON returned by SageMaker: %s", decode_error)
        raise HTTPException(status_code=502, detail="Received invalid response from SageMaker endpoint") from decode_error
    except Exception as unexpected_error:  # defensive fallback
        logger.exception("Unexpected server error: %s", unexpected_error)
        raise HTTPException(status_code=500, detail="An unexpected error occurred") from unexpected_error


@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException) -> JSONResponse:
    """Custom handler for HTTP exceptions."""
    logger.error("HTTP error occurred: %s", exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
