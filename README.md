# fastapi-sagemaker-integration

A production-oriented FastAPI microservice for real-time inference against an AWS SageMaker endpoint.

## What this service does

- Exposes a lightweight REST API for model predictions.
- Validates incoming request payloads with Pydantic.
- Sends inference data to SageMaker Runtime (`InvokeEndpoint`).
- Returns normalized JSON responses.
- Includes structured logging and clearer AWS error handling.

## API Endpoints

### `GET /`
Health check endpoint.

**Response**
```json
{
  "message": "Service is running"
}
```

### `POST /predict`
Runs model inference through the configured SageMaker endpoint.

**Request body**
```json
{
  "data": [
    {"feature_1": 1.2, "feature_2": 3.4}
  ]
}
```

**Successful response**
```json
{
  "prediction": {
    "predictions": [0.93]
  }
}
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `AWS_REGION` | `us-east-1` | AWS region where your SageMaker endpoint is deployed |
| `SAGEMAKER_ENDPOINT_NAME` | `mySageMakerEndpoint` | Name of the SageMaker endpoint to invoke |
| `LOG_LEVEL` | `INFO` | Python log level (`DEBUG`, `INFO`, `WARNING`, etc.) |

## Local development

### 1. Install dependencies

```bash
pip install fastapi uvicorn boto3 pydantic
```

### 2. Configure environment

```bash
export AWS_REGION=us-east-1
export SAGEMAKER_ENDPOINT_NAME=your-endpoint-name
export LOG_LEVEL=INFO
```

### 3. Run the service

```bash
uvicorn microserviceFastAPI:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Try the API

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"data": [{"feature_1": 1.2, "feature_2": 3.4}]}'
```

## AWS prerequisites

- A deployed and active SageMaker endpoint.
- AWS credentials available to the runtime (environment variables, IAM role, or profile).
- IAM permissions for `sagemaker:InvokeEndpoint`.

## Notes

- The service wraps SageMaker failures in HTTP `502` responses.
- Unexpected server failures return HTTP `500`.
- HTTP errors are returned in a consistent format:

```json
{
  "error": "..."
}
```
