# Export UID and GID for non-root container usage
export UID=$(shell id -u)
export GID=$(shell id -g)

# Standard run without GPU support
run:
	export UID=$(shell id -u) && export GID=$(shell id -g) && \
	docker compose up -d --build

# Run with GPU support using docker compose

dev-up:
	export UID=$(shell id -u) && export GID=$(shell id -g) && \
	docker compose -f docker-compose.dev.yml up --build -d

# Enter a running container with proper user mapping
dev-run:
	export UID=$(shell id -u) && export GID=$(shell id -g) && \
	docker exec -it -e UID=$(shell id -u) -e GID=$(shell id -g) aiams-dev bash

# Run a command in a new container
exec:
	export UID=$(shell id -u) && export GID=$(shell id -g) && \
	docker run -it --rm \
	--runtime=nvidia \
	-v $(CURDIR):/workspace/aiams \
	-v $(CURDIR)/data:/workspace/aiams/data \
	-v $(CURDIR)/runs:/workspace/aiams/runs \
	-v $(CURDIR)/model:/workspace/aiams/model \
	--network=host \
	--shm-size=32g \
	-e PYTHONPATH=/workspace/aiams \
	-e NVIDIA_VISIBLE_DEVICES=all \
	-e UID=$(shell id -u) \
	-e GID=$(shell id -g) \
	aiams $(filter-out $@,$(MAKECMDGOALS))

# Stop and remove containers
stop:
	docker compose down
	docker rm -f aiams 2>/dev/null || true
	docker rm -f aiams-dev 2>/dev/null || true

# Rebuild the image
build:
	docker build -t aiams .

# Help command to show available targets
help:
	@echo "Available targets:"
	@echo "  run              - Run standard container with non-root permissions"
	@echo "  dev-up           - Run development container with GPU and non-root permissions"
	@echo "  dev-run          - Start shell in development container as non-root user"
	@echo "  exec             - Run a command in a new container as non-root user"
	@echo "                     Example: make exec python -m tools.converters.excel_utils to-pdf ./data/file.xlsx"
	@echo "  stop             - Stop and remove containers"
	@echo "  build            - Build the Docker image"
	@echo "  help             - Show this help message"

# Allow passing arguments to exec target
%:
	@:
