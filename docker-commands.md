# Docker Basic Commands

## Pull an Image from Docker Hub

```bash
# Pull hello-world image
docker pull hello-world

# Pull specific version
docker pull nginx:latest

# Pull from a specific registry
docker pull ubuntu:22.04
```

## Run a Container

```bash
# Run hello-world
docker run hello-world

# Run container with a custom name
docker run --name my-container hello-world

# Run container in detached mode (background)
docker run -d nginx

# Run with port mapping
docker run -d -p 8080:80 nginx

# Run interactively with terminal
docker run -it ubuntu bash
```

## Push an Image to Docker Hub

```bash
# 1. Login to Docker Hub
docker login

# 2. Tag your image with your username
docker tag my-image:latest username/my-image:latest

# 3. Push the image
docker push username/my-image:latest
```

## Build an Image

```bash
# Build from Dockerfile in current directory
docker build -t my-app:v1 .

# Build with custom Dockerfile name
docker build -f Dockerfile.prod -t my-app:prod .

# Build with no cache
docker build --no-cache -t my-app:v1 .
```

## Container Management

```bash
# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# Stop a running container
docker stop <container-id>

# Start a stopped container
docker start <container-id>

# Restart a container
docker restart <container-id>

# Remove a container
docker rm <container-id>

# Remove a running container (force)
docker rm -f <container-id>
```

## Image Management

```bash
# List all images
docker images

# Remove an image
docker rmi <image-id>

# Remove unused images
docker image prune

# Remove all unused images
docker image prune -a
```

## View Logs

```bash
# View container logs
docker logs <container-id>

# Follow logs in real-time
docker logs -f <container-id>

# View last 100 lines
docker logs --tail 100 <container-id>
```

## Execute Commands in Running Container

```bash
# Execute command in running container
docker exec <container-id> ls -la

# Open bash shell in running container
docker exec -it <container-id> bash

# Run as specific user
docker exec -u root <container-id> whoami
```

## Inspect Container/Image

```bash
# Inspect container details
docker inspect <container-id>

# Inspect image details
docker inspect <image-name>

# Get container IP address
docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' <container-id>
```

## Docker System Commands

```bash
# Show docker disk usage
docker system df

# Remove unused data (containers, images, networks)
docker system prune

# Remove everything including volumes
docker system prune -a --volumes

# Show docker info
docker info

# Show docker version
docker version
```

## Complete Example Workflow

```bash
# 1. Pull hello-world image
docker pull hello-world

# 2. Run hello-world
docker run hello-world

# 3. Build your own image
docker build -t myapp:v1 .

# 4. Run your image
docker run -d -p 5000:5000 --name myapp-container myapp:v1

# 5. Check if it's running
docker ps

# 6. View logs
docker logs myapp-container

# 7. Login to Docker Hub
docker login

# 8. Tag your image
docker tag myapp:v1 username/myapp:v1

# 9. Push to Docker Hub
docker push username/myapp:v1

# 10. Stop and remove container
docker stop myapp-container
docker rm myapp-container
```

## Useful Shortcuts

```bash
# Stop all running containers
docker stop $(docker ps -q)

# Remove all stopped containers
docker rm $(docker ps -a -q)

# Remove all images
docker rmi $(docker images -q)

# Remove dangling images
docker rmi $(docker images -f "dangling=true" -q)
```
