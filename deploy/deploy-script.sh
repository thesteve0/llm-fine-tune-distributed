#!/bin/bash

# SmolLM3-3B Distributed Fine-tuning Deployment Script for OpenShift AI with Kubeflow
# Project: lyric-professor
# Model: SmolLM3-3B for wilderness survival and practical skills Q&A

set -e  # Exit on any error

# Configuration
REGISTRY_URL=${REGISTRY_URL:-"ghcr.io"}
IMAGE_NAME=${IMAGE_NAME:-"thesteve0/smollm3-distributed-finetune"}

echo "Deploying SmolLM3-3B distributed wilderness survival fine-tuning job to OpenShift AI..."

# Step 1: Generate timestamp-based version
echo "Generating timestamp-based version..."
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
NEW_VERSION="0.1.${TIMESTAMP}"
echo "New version: ${NEW_VERSION}"

# Update .version file
echo "${NEW_VERSION}" > ../.version
echo "Updated .version file"

# Full image name with new version
FULL_IMAGE_NAME="${REGISTRY_URL}/${IMAGE_NAME}:${NEW_VERSION}"
echo "Container image: ${FULL_IMAGE_NAME}"

# Step 2: Build and push container image
echo "Building container image..."
cd .. # Go to project root
docker build -t "${FULL_IMAGE_NAME}" .

echo "Pushing container image to registry..."
docker push "${FULL_IMAGE_NAME}"
echo "Container image built and pushed successfully"

# Step 3: Update PyTorchJob with new version and image
echo "Updating PyTorchJob configuration..."
cd deploy

# Create a temporary copy of the PyTorchJob to modify
cp pytorchjob.yaml pytorchjob-temp.yaml

# Add project-version label to metadata
sed -i "/ml-platform\/workbench:/a\\    project-version: \"${NEW_VERSION}\"" pytorchjob-temp.yaml

# Update the container image for distributed training
sed -i "s|image: your-registry/smollm3-distributed-finetune:.*|image: ${FULL_IMAGE_NAME}|g" pytorchjob-temp.yaml

echo "PyTorchJob updated with version ${NEW_VERSION}"

# Step 4: Ensure we're in the correct namespace IT IS ASSUMED YOU ALREADY DID THIS
#echo "Setting up OpenShift AI namespace..."
#kubectl config set-context --current --namespace=lyric-professor

# Check for existing PyTorchJob and clean up if necessary
echo "Checking for existing PyTorchJob deployments..."
if kubectl get pytorchjob smollm3-distributed-finetuning >/dev/null 2>&1; then
    echo "Found existing PyTorchJob 'smollm3-distributed-finetuning', deleting..."
    kubectl delete pytorchjob smollm3-distributed-finetuning

    # Wait for pods to be cleaned up
    echo "Waiting for pods to be cleaned up..."
    sleep 15

    # Check if any pods are still running and force delete if necessary
    if kubectl get pods -l pytorch-job-name=smollm3-distributed-finetuning --no-headers 2>/dev/null | grep -v "No resources found"; then
        echo "Force deleting remaining pods..."
        kubectl delete pods -l pytorch-job-name=smollm3-distributed-finetuning --force --grace-period=0 2>/dev/null || true
        sleep 10
    fi

    echo "Cleanup completed"
else
    echo "No existing PyTorchJob found"
fi

# Create PVC for master pod to persist trained models
echo "Creating PVC for master model storage..."
kubectl apply -f storage.yaml

# Create service for master pod
echo "Creating service for master pod..."
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Service
metadata:
  name: smollm3-distributed-finetuning-master-0
  namespace: lyric-professor
  labels:
    app: smollm3-distributed-finetuning
    role: master
spec:
  selector:
    training.kubeflow.org/job-name: smollm3-distributed-finetuning
    training.kubeflow.org/replica-type: master
    training.kubeflow.org/replica-index: "0"
  ports:
  - name: master-port
    port: 23456
    targetPort: 23456
    protocol: TCP
  type: ClusterIP
EOF

# Apply the PyTorchJob to start distributed training
echo "Starting distributed PyTorchJob for wilderness survival fine-tuning with version ${NEW_VERSION}..."
kubectl apply -f pytorchjob-temp.yaml

# Wait a moment for the job to initialize
sleep 10

# Check the status of the PyTorchJob
echo "Checking PyTorchJob status..."
kubectl get pytorchjob smollm3-distributed-finetuning -o wide

# Show all pods created by the distributed job
echo "Listing distributed training pods..."
kubectl get pods -l pytorch-job-name=smollm3-distributed-finetuning

# Display resource usage information for distributed training
echo ""
echo "Resource allocation for distributed training:"
echo "- Master Node: 1x NVIDIA L40S (48GB VRAM), 20GB RAM, 6 vCPU (saves final model)"
echo "- Worker Node: 1x NVIDIA L40S (48GB VRAM), 20GB RAM, 6 vCPU (training only)"
echo "- Total GPUs: 2 (1 Master + 1 Worker)"
echo "- Total VRAM: 96GB (48GB × 2 GPUs)"
echo "- Storage: Local pod storage only, no shared PVCs"
echo "- Effective batch size: 8 × 2 = 16 (per device × world size)"

echo ""
echo "To monitor distributed training progress:"
echo "Master node: kubectl logs -f smollm3-distributed-finetuning-master-0"
echo "Worker node: kubectl logs -f smollm3-distributed-finetuning-worker-0"
echo ""
echo "To check OpenShift AI metrics:"
echo "Navigate to: OpenShift AI → Distributed workloads → Project metrics"

echo ""
echo "Following master node training logs..."
kubectl logs -f smollm3-distributed-finetuning-master-0