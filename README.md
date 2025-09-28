# llm-fine-tune-distributed

Wilderness Survival & Practical Skills Q&A Distributed Fine-tuning with OpenShift AI and Kubeflow

This project demonstrates **distributed fine-tuning** of HuggingFaceTB/SmolLM3-3B using TRL framework across **multiple GPUs** on practical skills Q&A data with Red Hat OpenShift AI and PyTorchJob orchestration. The model transforms from a generic language model into a comprehensive wilderness survival expert that provides detailed guidance on essential survival and practical skills, with **4x faster training** through distributed computing.

## Distributed Training Architecture

- **Master Node**: 1 replica (coordinates training, handles model saving)
- **Worker Nodes**: 3 replicas (participate in distributed training)
- **Total GPUs**: 4× NVIDIA L40S (192GB total VRAM)
- **Communication**: NCCL backend for GPU-to-GPU coordination
- **Training Speed**: ~4x faster than single-node training

## Good Questions for Testing

1. How many cups in a gallon?
2. How do I treat a nosebleed?
3. What are the advantages of a mirrorless DSLR camera?
4. What is the easiest loop knot to tie?
5. I have a whistle, what is the right way to signal for help?

## Quick Start (Distributed Training)

1. **Deploy Distributed Training Job**:
   ```bash
   cd deploy
   ./deploy-script.sh
   ```

2. **Monitor Distributed Training**:
   ```bash
   # Master node (coordinates training)
   kubectl logs -f smollm3-distributed-finetuning-master-0

   # Worker nodes (parallel training)
   kubectl logs -f smollm3-distributed-finetuning-worker-0
   kubectl logs -f smollm3-distributed-finetuning-worker-1
   kubectl logs -f smollm3-distributed-finetuning-worker-2
   ```

3. **Check All Training Pods**:
   ```bash
   kubectl get pods -l pytorch-job-name=smollm3-distributed-finetuning
   ```

4. **Download Trained Model** (after distributed training completes):
   ```bash
   oc apply -f outputs/temp-pod.yaml
   ```

   Download the model files:
   ```bash
   # Wait for pod to be ready
   oc wait --for=condition=Ready pod/model-extractor

   oc rsync --progress=true model-extractor:/models/ .

   # Clean up
   oc delete pod model-extractor
   ```

## Distributed Training Performance

**Resource Allocation:**
- **Master Node**: 1× NVIDIA L40S (48GB VRAM), 20GB RAM, 6 vCPU
- **Worker Nodes**: 3× NVIDIA L40S (48GB VRAM each), 20GB RAM, 6 vCPU each
- **Total Resources**: 4 GPUs, 192GB VRAM, 80GB RAM, 24 vCPU
- **Effective Batch Size**: 64 (8 per device × 4 GPUs × 2 gradient accumulation)

**Training Optimizations:**
- **Distributed Data Parallel (DDP)**: Gradient synchronization across GPUs
- **NCCL Backend**: Optimized GPU-to-GPU communication
- **Learning Rate Scaling**: 2e-4 (scaled by world size for larger batch)
- **BF16 Precision**: Optimal for L40S Ada Lovelace architecture

## What You'll Get

After distributed training completes, your downloaded model directory will contain:

- `best_model/` - Your fine-tuned SmolLM3-3B wilderness survival expert model
- `training_summary.json` - Training configuration and distributed results
- `training_history.json` - Epoch-by-epoch metrics from distributed training
- `demo_outputs/sample_responses.json` - Sample Q&A model outputs
- `checkpoints/` - Training checkpoints from distributed run

## Converting safetensors to gguf: Convert the Model Yourself (The Technical Way)

If you can't find a pre-converted version or want to use your specific safetensors file, you'll need to convert it to GGUF yourself using a tool like llama.cpp. This process is more involved but is a great skill to learn.

### Clone llama.cpp:

First, you need to get the conversion tools from the llama.cpp project on GitHub.

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
```

### Install Dependencies:

Install the required Python packages. It's best to do this in a virtual environment.

```bash
pip install -r requirements.txt
```

**Expected Difference**: The distributed fine-tuned model provides comprehensive, step-by-step wilderness survival guidance with safety warnings and multiple approaches, while the original model gives more general responses. Training is significantly faster with 4 GPUs working in parallel.

The convert.py script in the llama.cpp directory can handle many model types. You'll point it to the directory containing your original SmolLM3 safetensors files.

```bash
python convert_hf_to_gguf.py /path/to/your/smollm3/model --outfile wilderness-smollm3.gguf --outtype f16
```

Replace `/path/to/your/original/smollm3/model/` with the actual path.

`--outtype f16` creates a 16-bit float model, which is a good balance of size and quality.

## Dataset Information

**Source**: [cahlen/offline-practical-skills-qa-synthetic](https://huggingface.co/datasets/cahlen/offline-practical-skills-qa-synthetic)

**Dataset Details**:
- **Size**: 2,845 Q&A pairs
- **Format**: Converted from JSONL to Parquet for efficient loading
- **Topics**: Car maintenance, home repairs, computer troubleshooting, practical life skills
- **Integration**: Dataset embedded directly in Docker container (no runtime download)
- **Runtime Processing**: Converted to TRL chat template format during distributed training
- **Distribution**: Same dataset loaded on all nodes (optimal for this dataset size)

**Sample Q&A**:
```
Question: "For Simple Car Maintenance Checks, What is the recommended tire pressure for my car?"
Answer: "Check your car's owner's manual or the tire information placard on the driver's side doorjamb for the recommended tire pressure."
```

**Data Processing Pipeline**:
1. Downloaded original JSONL format from HuggingFace
2. Transformed to combine topic and question: "For [topic], [question]"
3. Converted to Parquet format for efficient loading
4. Embedded in container for offline distributed training

## Distributed Training Technical Details

**PyTorch Distributed Setup**:
- **Backend**: NCCL for GPU communication
- **Master Address**: `smollm3-distributed-finetuning-master-0`
- **Master Port**: 23456
- **World Size**: 4 (1 Master + 3 Workers)
- **Rank Assignment**: Automatic via PyTorchJob controller

**Communication and Synchronization**:
- **Gradient Synchronization**: Automatic via PyTorch DDP
- **Model Parameter Sharing**: Efficient parameter broadcasting
- **Checkpoint Coordination**: Only rank 0 saves model artifacts
- **Error Handling**: Restart policy for failed workers

**Performance Benefits**:
- **Linear Speedup**: ~4x faster training with 4 GPUs
- **Memory Efficiency**: 192GB total VRAM for larger models/batches
- **Throughput**: Higher samples per second through parallel processing
- **Convergence**: Better convergence with larger effective batch size

## OpenShift AI Integration

**Monitoring**:
- Navigate to: **OpenShift AI → Distributed workloads → Project metrics**
- View GPU utilization across all 4 nodes
- Monitor memory usage and training progress
- Track distributed communication efficiency

**Resource Management**:
- Automatic pod scheduling across available GPU nodes
- Persistent volume sharing for model outputs
- Network configuration for inter-node communication
- Resource quotas and limits enforcement

## Project Details

See [claude.md](claude.md) for complete project documentation, distributed training technical specifications, L40S GPU optimization details for multi-node setups, and OpenShift AI distributed integration information.

## Troubleshooting Distributed Training

**Common Issues**:
1. **Communication Timeouts**: Check NCCL_DEBUG logs and network connectivity
2. **Uneven Pod Scheduling**: Verify GPU node availability and tolerations
3. **Storage Access**: Ensure PVCs are accessible from all nodes
4. **Memory Issues**: Monitor VRAM usage across all GPUs

**Debug Commands**:
```bash
# Check all distributed training pods
kubectl get pods -l pytorch-job-name=smollm3-distributed-finetuning

# Check PyTorchJob status
kubectl describe pytorchjob smollm3-distributed-finetuning

# View logs from specific worker
kubectl logs smollm3-distributed-finetuning-worker-0

# Check GPU allocation
kubectl describe nodes -l node.kubernetes.io/instance-type=gpu
```