ARG BUILD_FROM
FROM $BUILD_FROM

WORKDIR /app

# Install requirements for add-on
RUN \
    apk add --no-cache \
    python3 \
    py3-pip \
    gcc musl-dev python3-dev libffi-dev openssl-dev

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN apk del gcc musl-dev python3-dev libffi-dev openssl-dev


# Copy application and library code
COPY app/ /app/app/
COPY src/ /app/src/

ENV PYTHONPATH="/app/src"

CMD [ "python", "/app/app/main.py" ]