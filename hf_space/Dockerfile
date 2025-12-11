# IronGuard - AI Foundry Temperature Monitoring
# Hugging Face Spaces Docker Deployment

FROM python:3.11-slim

# Create non-root user (HF requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirements first for better caching
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy application files
COPY --chown=user . /app

# Create necessary directories
RUN mkdir -p logs models

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PORT=7860

# Run the application
CMD ["python", "app.py"]
