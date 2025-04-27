FROM python:3.9-slim   
WORKDIR /app

# Install dependencies first (cached unless requirements.txt changes)
COPY ./check-service/requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the code
COPY ./check-service .

# Run Flask app (adjust `app.py` to your entrypoint)
CMD ["python", "app.py"]