FROM python:3.10-slim

LABEL name="SmartInboxEnv" \
      version="2.0.0" \
      description="AI Email Triage — OpenEnv Environment"

# Create non-root user for security
RUN useradd -m -u 1000 agent

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY env.py \
     models.py \
     tasks.py \
     grader.py \
     inference.py \
     openenv_validator.py \
     openenv.yaml \
     ./

# Run validation at build time — build fails if env is non-compliant
RUN python openenv_validator.py

# Hand off to non-root user
RUN chown -R agent:agent /app
USER agent

# Default: run all tasks with reproducible seed
ENTRYPOINT ["python", "inference.py"]
CMD ["--task", "all", "--seed", "42"]
