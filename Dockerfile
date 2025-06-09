# Stage 1: Use an official, slim Python runtime as a parent image.
# Using a specific version like 3.12 is a good practice for reproducibility.
FROM python:3.12-slim

# Stage 2: Set up the working environment inside the container.
# Set the working directory to /app
WORKDIR /app

# Set environment variables to prevent Python from writing .pyc files
# and to run in unbuffered mode, which is better for logging.
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Stage 3: Install dependencies.
# Copy the requirements file first to leverage Docker's layer caching.
# This step will only be re-run if requirements.txt changes, speeding up future builds.
COPY requirements.txt .

# Install the Python dependencies using pip.
# --no-cache-dir makes the image smaller.
# --upgrade pip ensures we have the latest version.
RUN pip install --no-cache-dir --upgrade pip -r requirements.txt

# Stage 4: Copy the application code into the container.
# This includes our 'app/' and 'scripts/' directories.
COPY . .

# Stage 5: Expose the port the app runs on.
# This informs Docker that the container will listen on port 8000 at runtime.
EXPOSE 8000

# Stage 6: Define the command to run the application.
# This is the command that will be executed when the container starts.
# We use "--host 0.0.0.0" to make the server accessible from outside the container.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]