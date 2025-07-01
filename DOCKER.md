# Preventing Root-Owned Files in Docker Containers

When running Docker containers with volume mounts, files created within the container often end up owned by the root user on the host system. This can cause permission issues when trying to manage these files from the host. Here are several approaches to solve this issue:

## Solution 1: Use the `user` directive in docker-compose

Modify your docker-compose.yml to specify the user ID and group ID that matches your host system user:

```yaml
services:
  aiams:
    # ... other configuration ...
    user: "${UID}:${GID}"
    # ... volumes and other settings ...
```

When starting the container, pass your user and group IDs:

```bash
export UID=$(id -u)
export GID=$(id -g)
docker-compose up
```

## Solution 2: Create a non-root user in the Dockerfile

Modify your Dockerfile to create a user that matches your host user ID:

```Dockerfile
# ... existing Dockerfile content ...

# Create a non-root user with the same UID/GID as the host user
ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd -g $GROUP_ID appuser && \
    useradd -u $USER_ID -g $GROUP_ID -m -s /bin/bash appuser

# Set the working directory permissions
RUN chown -R appuser:appuser /workspace/aiams

# Switch to the non-root user
USER appuser
```

Build with the correct user IDs:

```bash
docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t aiams-image .
```

## Solution 3: Use an entrypoint script to adjust permissions

1. Create an entrypoint script:

```bash
#!/bin/bash
# entrypoint.sh

# Get user ID from environment or use default
APP_USER_ID=${APP_USER_ID:-1000}
APP_GROUP_ID=${APP_GROUP_ID:-1000}

# Create the user and group with the specified IDs
groupadd -g $APP_GROUP_ID appgroup
useradd -u $APP_USER_ID -g $APP_GROUP_ID -m -s /bin/bash appuser

# Ensure ownership of working directory
chown -R appuser:appgroup /workspace/aiams

# Execute the command as the appuser
exec gosu appuser "$@"
```

2. Update your Dockerfile:

```Dockerfile
# ... existing content ...

# Install gosu for easy step-down from root
RUN apt-get update && apt-get install -y gosu

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
```

## Solution 4: Use Docker's user namespace remapping

This is a more advanced solution that configures Docker daemon to remap user namespaces:

1. Edit `/etc/docker/daemon.json`:

```json
{
  "userns-remap": "default"
}
```

2. Restart Docker:

```bash
sudo systemctl restart docker
```

## Recommended Approach for Your Project

For your AI model project, I recommend either Solution 1 or Solution 2, as they are the simplest to implement and maintain. Solution 1 is particularly good if your needs are simple and you're the only one working on the project.

Example implementation for your docker-compose.yml:

```yaml
services:
  aiams:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: aiams
    restart: unless-stopped
    runtime: nvidia
    user: "${UID}:${GID}"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      # Mount the entire project directory
      - .:/workspace/aiams
      # Mount specific data directories with more explicit bindings
      - ./data:/workspace/aiams/data
      - ./runs:/workspace/aiams/runs
      - ./model:/workspace/aiams/model
    working_dir: /workspace/aiams
    environment:
      - PYTHONPATH=/workspace/aiams
      # Basic NVIDIA environment variables for GPU access
      - NVIDIA_VISIBLE_DEVICES=all
    network_mode: host
    shm_size: 32g
    tty: true
```

Start your container with:

```bash
export UID=$(id -u)
export GID=$(id -g)
docker-compose up
