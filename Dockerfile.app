FROM python:3.10-slim

RUN pip install --upgrade pip

COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy the entire src directory
COPY ./src /app/src

# Expose the correct ports (8001 for the API, 4201 for Prefect)
EXPOSE 8001 4200


# Create modelling directory in web_service and copy the model file
RUN mkdir -p /app/src/web_service/modelling
RUN cp /app/src/modelling/LR.pkl /app/src/web_service/modelling/


# Set working directory
WORKDIR /app/src/web_service

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
