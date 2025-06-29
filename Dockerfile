# Use a specific Python base image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port your app will run on
EXPOSE $PORT

# Command to run the application using Gunicorn
CMD exec gunicorn --bind 0.0.0.0:$PORT app:app