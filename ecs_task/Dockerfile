FROM tiangolo/uwsgi-nginx-flask:python3.8

COPY ./app /app
COPY requirements.txt .

RUN pip install -r requirements.txt

ARG S3_MODEL_BUCKET
ENV S3_MODEL_BUCKET=$S3_MODEL_BUCKET

ARG S3_MODEL_KEY
ENV S3_MODEL_KEY=$S3_MODEL_KEY

# Build like:
# docker build -t ecs-infer . --build-arg S3_MODEL_BUCKET=<bucket name> --build-arg S3_MODEL_KEY=<model name>

# Run like:
# docker run ecs-infer
