For local clusters (minikube):

```bash
# Use minikube's Docker daemon
eval $(minikube docker-env)

# Build again in minikube's context
docker build -t mlops-app:v1 .
```

## Step 3: Deploy to Kubernetes

```bash
# Apply the manifest file
kubectl apply -f k8s/mlops-test.yaml

# Verify the deployment
kubectl get deployments
kubectl get pods
kubectl get services
```

## Step 4: Verify Deployment Status

```bash
# Check pod status
kubectl get pods -l app=mlops-app

# Check deployment details
kubectl describe deployment mlops-app

# Check service details
kubectl describe service mlops-service

# View logs
kubectl logs -l app=mlops-app --tail=50
```

## Step 5: Access the Application

### Using NodePort

```bash
# Get the NodePort
kubectl get service mlops-service

# For minikube, get the URL
minikube service mlops-service --url

# For other clusters, access using:
# http://<node-ip>:30080
```

### Using Port Forwarding (for testing)

```bash
# Forward local port 8080 to the service
kubectl port-forward service/mlops-service 8080:80

# Or forward to a specific pod
kubectl port-forward pod/<pod-name> 8080:8080

# Access the application at:
# http://localhost:8080
```

### Port Forwarding with Different Ports

```bash
# Forward local port 3000 to service port 80
kubectl port-forward service/mlops-service 3000:80

# Forward with background process
kubectl port-forward service/mlops-service 8080:80 &

# Access at http://localhost:3000 or http://localhost:8080
```

## Step 6: Test the Application

```bash
# Using curl
curl http://localhost:8080

# Or open in browser
# http://localhost:8080
```

## Useful Commands

### Scaling the Deployment
```bash
kubectl scale deployment mlops-app --replicas=3
```

### Update the Deployment
```bash
# After building a new image version
kubectl set image deployment/mlops-app mlops-container=mlops-app:v2
```

### Rollback Deployment
```bash
kubectl rollout undo deployment/mlops-app
```

### Delete Resources
```bash
kubectl delete -f k8s/mlops-test.yaml
```

### View Real-time Logs
```bash
kubectl logs -f deployment/mlops-app
```

## Troubleshooting

### Pods not starting
```bash
kubectl describe pod <pod-name>
kubectl logs <pod-name>
```

### Service not accessible
```bash
kubectl get endpoints mlops-service
kubectl describe service mlops-service
```

### Image pull errors
```bash
# For local images, ensure imagePullPolicy is set to Never or IfNotPresent
kubectl edit deployment mlops-app
```
