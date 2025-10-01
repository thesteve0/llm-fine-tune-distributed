#!/bin/bash

# Distributed Training Monitor Script for SmolLM3-3B Fine-tuning
# Monitors all nodes in the distributed training setup

set -e

PYTORCHJOB_NAME="smollm3-distributed-finetuning"

echo "Monitoring Distributed Training: ${PYTORCHJOB_NAME}"
echo "==========================================================="

# Function to check if PyTorchJob exists
check_job_exists() {
    kubectl get pytorchjob $PYTORCHJOB_NAME >/dev/null 2>&1
}

# Check if the job exists
if ! check_job_exists; then
    echo "PyTorchJob '${PYTORCHJOB_NAME}' not found"
    echo "Deploy the job first with: ./deploy-script.sh"
    exit 1
fi

# Show PyTorchJob status
echo "PyTorchJob Status:"
kubectl get pytorchjob $PYTORCHJOB_NAME -o wide

echo ""
echo "Distributed Training Pods:"
kubectl get pods -l pytorch-job-name=$PYTORCHJOB_NAME -o wide

echo ""
echo "Resource Usage per Pod:"
kubectl top pods -l pytorch-job-name=$PYTORCHJOB_NAME

echo ""
echo "GPU Allocation:"
kubectl describe pods -l pytorch-job-name=$PYTORCHJOB_NAME | grep -A 5 "nvidia.com/gpu"

echo ""
echo "Node Distribution:"
kubectl get pods -l pytorch-job-name=$PYTORCHJOB_NAME -o custom-columns="POD:.metadata.name,NODE:.spec.nodeName,STATUS:.status.phase"

echo ""
echo "Pod Storage:"
echo "Each pod uses local storage only - no shared PVCs"

echo ""
echo "Recent Events:"
kubectl get events --field-selector involvedObject.name=$PYTORCHJOB_NAME --sort-by='.lastTimestamp' | tail -10

echo ""
echo "Available Log Commands:"
echo "Master node:  kubectl logs -f ${PYTORCHJOB_NAME}-master-0"
echo "Worker node:  kubectl logs -f ${PYTORCHJOB_NAME}-worker-0"
echo ""
echo "Monitor all logs: kubectl logs -f -l pytorch-job-name=${PYTORCHJOB_NAME}"

# Ask user which logs to follow
echo ""
read -p "Follow logs from which node? [master/worker0/all/none]: " choice

case $choice in
    master)
        echo "Following master node logs..."
        kubectl logs -f ${PYTORCHJOB_NAME}-master-0
        ;;
    worker0)
        echo "Following worker-0 logs..."
        kubectl logs -f ${PYTORCHJOB_NAME}-worker-0
        ;;
    all)
        echo "Following all distributed training logs..."
        kubectl logs -f -l pytorch-job-name=${PYTORCHJOB_NAME}
        ;;
    none|*)
        echo "Monitoring complete. Use the commands above to follow specific logs."
        ;;
esac