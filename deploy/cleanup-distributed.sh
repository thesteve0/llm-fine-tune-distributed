#!/bin/bash

# Distributed Training Cleanup Script for SmolLM3-3B Fine-tuning
# Cleans up all distributed training resources

set -e

PYTORCHJOB_NAME="smollm3-distributed-finetuning"

echo "🧹 Cleaning up Distributed Training Resources"
echo "=============================================="

# Function to check if PyTorchJob exists
check_job_exists() {
    kubectl get pytorchjob $PYTORCHJOB_NAME >/dev/null 2>&1
}

# Check current status
echo "🔍 Checking current distributed training status..."

if check_job_exists; then
    echo "📊 Current PyTorchJob status:"
    kubectl get pytorchjob $PYTORCHJOB_NAME -o wide

    echo ""
    echo "📋 Current pods:"
    kubectl get pods -l pytorch-job-name=$PYTORCHJOB_NAME -o wide

    echo ""
    read -p "⚠️  Do you want to delete the PyTorchJob and all associated pods? [y/N]: " confirm

    if [[ $confirm =~ ^[Yy]$ ]]; then
        echo "🗑️  Deleting PyTorchJob: $PYTORCHJOB_NAME"
        kubectl delete pytorchjob $PYTORCHJOB_NAME

        echo "⏳ Waiting for pods to be cleaned up..."
        sleep 15

        # Check if any pods are still running
        echo "🔍 Checking for remaining pods..."
        remaining_pods=$(kubectl get pods -l pytorch-job-name=$PYTORCHJOB_NAME --no-headers 2>/dev/null | wc -l)

        if [ "$remaining_pods" -gt 0 ]; then
            echo "⚠️  Found $remaining_pods remaining pods. Force deleting..."
            kubectl delete pods -l pytorch-job-name=$PYTORCHJOB_NAME --force --grace-period=0 2>/dev/null || true
            sleep 10
        fi

        echo "✅ PyTorchJob and pods deleted"
    else
        echo "❌ Cleanup cancelled"
        exit 0
    fi
else
    echo "✅ No PyTorchJob '${PYTORCHJOB_NAME}' found"
fi

# Check for any orphaned pods
echo ""
echo "🔍 Checking for orphaned pods..."
orphaned_pods=$(kubectl get pods -l pytorch-job-name=$PYTORCHJOB_NAME --no-headers 2>/dev/null | wc -l)

if [ "$orphaned_pods" -gt 0 ]; then
    echo "⚠️  Found $orphaned_pods orphaned pods:"
    kubectl get pods -l pytorch-job-name=$PYTORCHJOB_NAME

    read -p "🗑️  Delete orphaned pods? [y/N]: " confirm_orphaned

    if [[ $confirm_orphaned =~ ^[Yy]$ ]]; then
        kubectl delete pods -l pytorch-job-name=$PYTORCHJOB_NAME --force --grace-period=0
        echo "✅ Orphaned pods deleted"
    fi
else
    echo "✅ No orphaned pods found"
fi

# Ask about PVC cleanup
echo ""
echo "📦 Storage cleanup options:"
kubectl get pvc

echo ""
read -p "🗑️  Do you want to clean up PVCs (this will delete all training data and models)? [y/N]: " confirm_pvc

if [[ $confirm_pvc =~ ^[Yy]$ ]]; then
    echo "⚠️  Deleting all PVCs - this will permanently delete training data and models!"
    read -p "Are you absolutely sure? Type 'DELETE' to confirm: " final_confirm

    if [[ $final_confirm == "DELETE" ]]; then
        kubectl delete pvc trained-models-pvc workspace-pvc 2>/dev/null || echo "Some PVCs may not exist"
        echo "✅ PVCs deleted"
    else
        echo "❌ PVC cleanup cancelled"
    fi
else
    echo "📦 PVCs preserved (models and data intact)"
fi

# Clean up temporary files
echo ""
echo "🧹 Cleaning up temporary deployment files..."
if [ -f "pytorchjob-temp.yaml" ]; then
    rm pytorchjob-temp.yaml
    echo "✅ Removed pytorchjob-temp.yaml"
fi

echo ""
echo "✅ Distributed training cleanup complete!"
echo ""
echo "📋 Summary:"
echo "- PyTorchJob deleted: ✅"
echo "- Pods cleaned up: ✅"
echo "- Temp files removed: ✅"
echo "- PVCs: $(if [[ $confirm_pvc =~ ^[Yy]$ ]] && [[ $final_confirm == "DELETE" ]]; then echo "Deleted"; else echo "Preserved"; fi)"

echo ""
echo "🚀 To start a new distributed training run:"
echo "./deploy-script.sh"