# Architecture Diagrams: Single-Node vs Distributed Training

## Single-Node Training Architecture

```mermaid
graph TB
    subgraph "Kubernetes Cluster"
        subgraph "PyTorchJob: pythia-finetuning-demo"
            subgraph "Master Pod (Single Node)"
                Container[Training Container<br/>pythia-finetune:v1]
                GPU1[NVIDIA L40S GPU<br/>48GB VRAM]
                Model[SmolLM3-3B Model<br/>Batch Size: 8]

                Container --> GPU1
                Container --> Model
            end

            PVC1[PVC: trained-models]
            PVC2[PVC: workspace]

            Container --> PVC1
            Container --> PVC2
        end
    end

    style Container fill:#e1f5ff
    style GPU1 fill:#c8e6c9
    style Model fill:#fff9c4
```

**Characteristics:**
- Single pod with 1 GPU
- Batch size: 8 (memory constrained to single GPU)
- No inter-process communication needed
- Simple deployment

---

## Distributed Training Architecture

```mermaid
graph TB
    subgraph "Kubernetes Cluster"
        subgraph "PyTorchJob: smollm3-distributed-finetuning"
            subgraph "Master Pod - Rank 0"
                MasterContainer[Training Container<br/>smollm3-distributed-finetune:v1]
                MasterGPU[NVIDIA L40S GPU<br/>48GB VRAM]
                MasterModel[SmolLM3-3B Model<br/>Batch Size: 28]
                MasterEnv[ENV: RANK=0<br/>MASTER_ADDR=0.0.0.0<br/>WORLD_SIZE=2]

                MasterContainer --> MasterGPU
                MasterContainer --> MasterModel
                MasterContainer --> MasterEnv
            end

            subgraph "Worker Pod - Rank 1"
                WorkerContainer[Training Container<br/>smollm3-distributed-finetune:v1]
                WorkerGPU[NVIDIA L40S GPU<br/>48GB VRAM]
                WorkerModel[SmolLM3-3B Model<br/>Batch Size: 28]
                WorkerEnv[ENV: RANK=1<br/>MASTER_ADDR=master-0<br/>WORLD_SIZE=2]

                WorkerContainer --> WorkerGPU
                WorkerContainer --> WorkerModel
                WorkerContainer --> WorkerEnv
            end

            MasterContainer <-->|NCCL<br/>Port 23456| WorkerContainer

            PVC1[PVC: model-storage<br/>Shared by Master]
            PVC2[PVC: aim-runs<br/>Shared by Master & Worker]

            MasterContainer --> PVC1
            MasterContainer --> PVC2
            WorkerContainer --> PVC2
        end
    end

    style MasterContainer fill:#e1f5ff
    style WorkerContainer fill:#e1f5ff
    style MasterGPU fill:#c8e6c9
    style WorkerGPU fill:#c8e6c9
    style MasterModel fill:#fff9c4
    style WorkerModel fill:#fff9c4
    style MasterEnv fill:#ffe0b2
    style WorkerEnv fill:#ffe0b2
```

**Characteristics:**
- 2 pods, each with 1 GPU (2 GPUs total)
- Batch size: 28 per GPU (3.5x larger effective batch)
- NCCL communication over network
- Master coordinates, both train
- Shared Aim tracking across nodes

---

## Training Coordination Flow

```mermaid
sequenceDiagram
    participant K8s as Kubernetes/PyTorchJob Operator
    participant Master as Master Pod (Rank 0)
    participant Worker as Worker Pod (Rank 1)
    participant NCCL as NCCL Backend

    K8s->>Master: Create pod with RANK=0
    K8s->>Worker: Create pod with RANK=1

    Master->>Master: setup_distributed()<br/>Listen on port 23456
    Worker->>Worker: setup_distributed()<br/>Connect to master-0:23456

    Master->>NCCL: Initialize process group
    Worker->>NCCL: Join process group

    NCCL-->>Master: Group ready
    NCCL-->>Worker: Group ready

    loop Training Loop
        Master->>Master: Load batch (28 samples)
        Worker->>Worker: Load batch (28 samples)

        Master->>Master: Forward pass
        Worker->>Worker: Forward pass

        Master->>NCCL: Compute gradients
        Worker->>NCCL: Compute gradients

        NCCL->>NCCL: All-reduce gradients<br/>(average across GPUs)

        NCCL-->>Master: Synchronized gradients
        NCCL-->>Worker: Synchronized gradients

        Master->>Master: Update model weights
        Worker->>Worker: Update model weights
    end

    Master->>Master: Save model (rank 0 only)
    Worker->>Worker: Skip save (rank != 0)

    Master->>NCCL: Cleanup process group
    Worker->>NCCL: Cleanup process group
```

---

## Component Changes Overview

```mermaid
graph LR
    subgraph "File Changes Required"
        A[Dockerfile<br/>+1 line] -->|Add Aim| B[requirements.txt<br/>No change]
        B --> C[deploy/pytorchjob.yaml<br/>+Worker spec<br/>+NCCL env vars]
        C --> D[training.py<br/>+60 lines<br/>distributed setup]
    end

    subgraph "What Stays the Same"
        E[Model Architecture<br/>SmolLM3-3B]
        F[Training Loop<br/>SFTTrainer handles DDP]
        G[Dataset Format<br/>qa_dataset.parquet]
        H[Base Image<br/>quay.io/modh/training]
    end

    style A fill:#ffcdd2
    style B fill:#c8e6c9
    style C fill:#ffcdd2
    style D fill:#ffcdd2
    style E fill:#c8e6c9
    style F fill:#c8e6c9
    style G fill:#c8e6c9
    style H fill:#c8e6c9
```

**Legend:**
- 🔴 Red: Files that need changes
- 🟢 Green: No changes required

---

## Data Flow: Single-Node vs Distributed

### Single-Node Data Flow

```mermaid
flowchart TD
    Dataset[Dataset: 8 samples/batch]

    Dataset --> GPU[GPU 0]

    GPU --> Forward[Forward Pass]
    Forward --> Loss[Calculate Loss]
    Loss --> Backward[Backward Pass]
    Backward --> Update[Update Weights]
    Update --> GPU
```

**Effective batch size: 8**

### Distributed Data Flow

```mermaid
flowchart TD
    Dataset[Dataset: Split into 2 shards]

    Dataset --> GPU0[GPU 0<br/>28 samples]
    Dataset --> GPU1[GPU 1<br/>28 samples]

    GPU0 --> Forward0[Forward Pass]
    GPU1 --> Forward1[Forward Pass]

    Forward0 --> Loss0[Calculate Loss]
    Forward1 --> Loss1[Calculate Loss]

    Loss0 --> Backward0[Backward Pass]
    Loss1 --> Backward1[Backward Pass]

    Backward0 --> Grads0[Gradients]
    Backward1 --> Grads1[Gradients]

    Grads0 --> AllReduce[NCCL All-Reduce<br/>Average Gradients]
    Grads1 --> AllReduce

    AllReduce --> SyncGrad[Synchronized<br/>Gradients]

    SyncGrad --> Update0[Update Weights GPU 0]
    SyncGrad --> Update1[Update Weights GPU 1]

    Update0 --> GPU0
    Update1 --> GPU1
```

**Effective batch size: 56 (28 × 2)**

---

## Environment Variables Comparison

```mermaid
graph TD
    subgraph "Single-Node Env"
        SN1[EPOCHS=4]
        SN2[BATCH_SIZE=8]
        SN3[LEARNING_RATE=5e-5]
        SN4[OUTPUT_DIR=/shared/models]
    end

    subgraph "Distributed Master Env"
        DM1[EPOCHS=4]
        DM2[BATCH_SIZE=28]
        DM3[LEARNING_RATE=5e-5]
        DM4[OUTPUT_DIR=/persistent/models]
        DM5[WORLD_SIZE=2]
        DM6[RANK=0]
        DM7[MASTER_ADDR=0.0.0.0]
        DM8[MASTER_PORT=23456]
        DM9[NCCL_DEBUG=INFO]
        DM10[NCCL_SOCKET_IFNAME=eth0]
        DM11[AIM_REPO=/aim]
    end

    subgraph "Distributed Worker Env"
        DW1[EPOCHS=4]
        DW2[BATCH_SIZE=28]
        DW3[LEARNING_RATE=5e-5]
        DW4[WORLD_SIZE=2]
        DW5[RANK=1 auto-set]
        DW6[MASTER_ADDR=master-0]
        DW7[MASTER_PORT=23456]
        DW8[NCCL_DEBUG=INFO]
        DW9[AIM_REPO=/aim]
    end

    style DM5 fill:#ffcdd2
    style DM6 fill:#ffcdd2
    style DM7 fill:#ffcdd2
    style DM8 fill:#ffcdd2
    style DM9 fill:#ffcdd2
    style DM10 fill:#ffcdd2
    style DM11 fill:#ffcdd2
    style DW4 fill:#ffcdd2
    style DW5 fill:#ffcdd2
    style DW6 fill:#ffcdd2
    style DW7 fill:#ffcdd2
    style DW8 fill:#ffcdd2
    style DW9 fill:#ffcdd2
```

**Red highlighted**: New variables required for distributed training

---

## Key Takeaway: Minimal Changes

```mermaid
mindmap
  root((Distributed<br/>Training))
    Dockerfile
      +1 dependency aim
    requirements.txt
      No changes
    Deployment YAML
      Add Worker spec
      Add NCCL env vars
      Increase batch size
    training.py
      +setup_distributed 27 lines
      +cleanup_distributed 4 lines
      +rank checks ~10 lines
      +DDP config ~8 lines
      +callbacks ~15 lines
    What Stays Same
      Training loop code
      Model architecture
      Dataset processing
      SFTTrainer usage
```

**Total code changes: ~60 lines to unlock distributed training across multiple GPUs!**
