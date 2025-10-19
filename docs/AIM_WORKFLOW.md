# Aim Experiment Tracking Workflow Guide

This guide explains how to use Aim for tracking your distributed training experiments. If you're coming from Weights & Biases (W&B), this will help you understand the key differences and workflow.

## TL;DR - Quick Start for Continuous Loss Plot

1. **Port-forward Aim**: `kubectl port-forward -n lyric-professor svc/aim-service 43800:43800`
2. **Open**: http://localhost:43800
3. **Click "Metrics"** in the left sidebar
4. **Click "+ Metrics"** → select **"loss"**
5. **Group by Color** → select **"context.subset"** (separates train/eval)
6. **Apply smoothing** (0.6-0.8) for a cleaner line
7. **Refresh browser (F5)** every 1-2 minutes to see new data points
8. **Done!** You now have a continuous line plot of your training loss

Read on for detailed explanations...

## Key Differences from Weights & Biases

| Feature | Weights & Biases | Aim |
|---------|-----------------|-----|
| Live Updates | Auto-refreshes in real-time | Manual browser refresh needed |
| Dashboard | Auto-created with common metrics | You build custom views in explorers |
| Metric Detection | Auto-detects many metrics | Tracks what callbacks explicitly log |
| UI Paradigm | Streaming dashboard | Snapshot-based explorer |

## What Metrics Are Being Tracked

Your training code automatically tracks these metrics via `AimCallback`:

### Training Metrics (logged every 2 steps)
- `loss` - Training loss with context `subset='train'`
- `learning_rate` - Current learning rate with context `subset='train'`
- `epoch` - Current epoch number
- `perplexity` - Exponential of training loss (custom metric)
- `grad_norm` - Gradient norm (helps detect training instability)

### Evaluation Metrics (logged every 10 steps)
- `loss` - Validation loss with context `subset='eval'`
- `perplexity` - Exponential of validation loss (custom metric)
- `runtime` - Time taken for evaluation
- `samples_per_second` - Throughput metric
- `steps_per_second` - Processing speed metric

### Understanding Training Steps

With your current configuration:
- **Batch size**: 28 samples
- **Gradient accumulation**: 4 steps
- **World size**: 2 GPUs (distributed training)
- **Effective batch size**: 28 × 4 × 2 = **224 samples per training step**
- **Dataset size**: 2,560 training samples
- **Steps per epoch**: 2560 ÷ 224 ≈ **11-12 steps**
- **Total steps** (4 epochs): ~**48 steps**

This means:
- With `logging_steps=2`: ~**24 training log points** across 4 epochs
- With `eval_steps=10`: ~**4-5 evaluation runs** across training
- Much more data than the previous config (which only had 12 total steps!)

**Note:** Unlike classification models, causal language models don't automatically track accuracy. Loss and perplexity are the primary metrics for language model training.

## Accessing the Aim UI

### If Running Locally
```bash
aim up --repo /aim
```
Then open: http://localhost:43800

### If Running in Kubernetes (OpenShift)
```bash
# Port forward the Aim service
kubectl port-forward -n lyric-professor svc/aim-service 43800:43800

# Or if you need to find the pod name first:
kubectl get pods -n lyric-professor | grep aim
kubectl port-forward -n lyric-professor pod/<aim-pod-name> 43800:43800
```
Then open: http://localhost:43800

## Viewing Continuous Loss Plots - Step by Step

This is the most common use case: **seeing your training loss as one continuous line from start to finish**.

### Quick Start - View Training Loss

1. **Open Aim UI** at http://localhost:43800 (after port-forwarding)

2. **Click "Metrics"** in the left sidebar to open the Metrics Explorer

3. **Select the loss metric:**
   - Click the **"+ Metrics"** button at the top
   - In the dropdown, find and click **"loss"**
   - You should immediately see a line plot appear

4. **You now have a continuous line plot!** The x-axis shows training steps, y-axis shows loss value.

### Separating Training vs Evaluation Loss

By default, both train and eval loss might be shown together or overlapping. To separate them:

1. **Click "Group by"** in the toolbar
2. **Select "Color"** → then select **"context.subset"**
3. Now you'll see:
   - One color line for training loss (`subset='train'`)
   - Different color line for evaluation loss (`subset='eval'`)
4. The legend will show which color corresponds to which subset

### Customizing Your View

**Change the X-axis:**
- Click on the **X-axis dropdown** (default is "step")
- Options:
  - **"step"** - Training step number (recommended for most cases)
  - **"epoch"** - Training epoch (good for comparing across different runs)
  - **"relative time"** - How long training has been running
  - **"absolute time"** - Calendar timestamps

**Apply Smoothing:**
- Use the **"Smoothing"** slider (usually in the toolbar)
- Recommended: **0.6-0.8** for loss curves
- This removes noise and shows the overall trend more clearly

**Change Y-axis Scale:**
- Click the **Y-axis settings**
- Switch between:
  - **Linear** (default) - Normal scale
  - **Logarithmic** - Useful if your loss starts very high and drops significantly

### Viewing Multiple Metrics Together

To see loss AND perplexity on the same chart:

1. Click **"+ Metrics"** again
2. Add **"perplexity"**
3. Both metrics will appear on the same chart
4. Use **"Group by Chart"** → **"run.metric_name"** to put them in separate subplots if they have very different scales

### What You Should See

**During Training:**
- **Training loss** (logged every 2 steps): Should show a smooth downward trend with ~24 points
- **Evaluation loss** (logged every 10 steps): Should show ~4-5 points tracking close to training loss

**Healthy Training Pattern:**
- Both train and eval loss decreasing over time
- Eval loss tracking close to train loss (not much higher)
- Smooth curves without sudden spikes

**Warning Signs:**
- Eval loss much higher than train loss = overfitting
- Loss suddenly spiking = learning rate might be too high
- Loss not decreasing = learning rate might be too low or model not learning

### Refreshing to See New Data

**IMPORTANT:** While training is running:
1. Keep the Metrics Explorer open in your browser
2. Every 1-2 minutes, **press F5** or **Cmd+R** to refresh
3. New data points will appear as training progresses
4. The line will "grow" as more steps complete

## Using the Aim UI to Monitor Training

### Step 1: Navigate to Metrics Explorer

1. Open the Aim UI in your browser
2. From the **home page**, you'll see:
   - Overview section with active runs
   - Statistics of your training activities
   - Contributions heatmap showing experiment frequency
3. Click **"Metrics"** in the left sidebar to open the **Metrics Explorer**

### Step 2: Select Metrics to Visualize

In the Metrics Explorer:

1. **Click the "Metrics" dropdown** at the top
2. **Select the metrics you want to view:**
   - `loss` - This will show both train and eval loss
   - `perplexity` - This will show both train and eval perplexity
   - `learning_rate` - Shows learning rate over training
   - `samples_per_second` - Shows throughput metrics

3. **The metrics will appear as charts** in the explorer

### Step 3: Group and Filter Metrics

To separate training vs evaluation metrics:

1. **Click "Group by"** in the toolbar
2. **Select "context.subset"** - This separates `train` vs `eval` metrics
3. **Apply "Color by"** to distinguish different runs:
   - Useful if you have multiple experiments
   - Can color by hyperparameters like learning rate or batch size

### Step 4: Customize Visualization

**Smoothing:**
- Use the **"Smoothing"** slider to reduce noise in your loss curves
- Higher smoothing = smoother trends but less detail
- Recommended: 0.6-0.8 for loss curves

**Axis Options:**
- **X-axis**: Can align by step, epoch, or time
- **Y-axis**: Can use linear or log scale (log scale useful for loss)

**Highlight Runs:**
- Hover over a line to highlight it
- Click on a run in the legend to toggle visibility

### Step 5: Monitor in "Real-time"

**IMPORTANT:** This is the biggest difference from W&B:

- Aim data **is being written to `/aim` in real-time** as training progresses
- However, the UI **does NOT auto-refresh**
- **You must manually refresh your browser** (F5 or Cmd+R) to see new data points

**Recommended workflow:**
1. Open Metrics Explorer and set up your view (select metrics, grouping, smoothing)
2. Bookmark this view using the "Bookmark" button
3. Every 1-2 minutes, refresh your browser to see new training progress
4. Watch for:
   - Training loss decreasing
   - Eval loss tracking with train loss (no overfitting)
   - Perplexity decreasing (lower is better)
   - Learning rate schedule (if using a scheduler)

### Step 6: Comparing Multiple Runs

When you run multiple experiments:

1. All runs appear in the same Metrics Explorer
2. Use **"Runs"** dropdown to select/deselect specific runs
3. Use **"Color by"** to distinguish runs by hyperparameters
4. Create **custom queries** with AimQL:
   ```
   run.learning_rate > 1e-5 and run.batch_size == 28
   ```

## Understanding Your Training Progress

### What Good Training Looks Like in Aim

**Loss Metrics:**
- `train/loss` should steadily decrease
- `eval/loss` should decrease and track close to train loss
- If `eval/loss` >> `train/loss`, you're overfitting

**Perplexity Metrics:**
- Perplexity is `exp(loss)`, so it's more interpretable
- Lower perplexity = better model
- For language models, perplexity in the range of 10-50 is typical for fine-tuned models
- Your base model's perplexity gives you a baseline

**Learning Rate:**
- Should follow your scheduler (constant in your case at 5e-5)
- In distributed training, it's scaled by world_size (5e-5 * 2 = 1e-4)

### Troubleshooting

**Problem: No continuous line - just seeing scattered points**
- This is normal! Each point represents a logged step
- With your config: ~24 training points and ~4-5 eval points
- Apply **smoothing** (0.6-0.8) to make the line appear more continuous
- The line connects all the points chronologically by step number

**Problem: Line looks "jumpy" or has gaps**
- **Training loss**: Updates every 2 steps, so you get frequent points
- **Eval loss**: Only updates every 10 steps, so it appears more sparse
- Use **"Group by Color"** → **"context.subset"** to separate them
- Apply smoothing to see the overall trend

**Problem: Can't see the full training run - plot is cut off**
- Check the **x-axis range** - make sure it's not zoomed in
- Click the **"Reset zoom"** button if available
- Verify training actually completed: `kubectl logs -n lyric-professor <master-pod-name> | tail`

**Problem: No metrics showing up at all**
- Check that your training job is actually running: `kubectl get pods -n lyric-professor`
- Verify the `/aim` volume is mounted correctly
- Look at pod logs for any callback errors
- Make sure you've refreshed the browser after training started

**Problem: Metrics stopped updating mid-training**
- **First, refresh your browser!** (Aim doesn't auto-refresh)
- Check if training job crashed: `kubectl logs -n lyric-professor <pod-name>`
- Verify training is still running: `kubectl get pods -n lyric-professor`

**Problem: Only seeing hardware metrics, no training metrics**
- This usually means the training hasn't started logging yet
- Wait for first logging step (step 1, then every 2 steps)
- Check that `AimCallback` is in the callbacks list in the code
- Verify you're looking at the **Metrics Explorer**, not a different tab

**Problem: Train and eval loss on different scales - can't compare**
- Use **"Group by Chart"** → **"context.subset"** to create separate subplots
- Or use **logarithmic y-axis** if the scales differ significantly
- Alternatively, normalize the metrics by plotting them in separate views

## Advanced: Using the Home Page

The **Home Page** provides an overview:

1. **Active Runs table** - Shows currently running experiments
   - You should see your experiment while training is in progress
2. **Contributions heatmap** - Shows when you've been running experiments
   - Click on a cell to filter runs from that date
3. **Recent activity** - Quick access to recent runs and searches

## Advanced: Bookmarks and Views

Save your custom visualizations:

1. Set up your Metrics Explorer view (metrics, grouping, smoothing)
2. Click **"Bookmark"** button
3. Give it a name like "Training Loss Monitoring"
4. Access it later from the **"Bookmarks"** section on home page

## Quick Reference

### Common Keyboard Shortcuts
- **F5** or **Cmd+R** - Refresh to see new data (most important!)
- **Esc** - Close modals and dropdowns

### Expected Update Frequency
- Training metrics: Every 2 steps (~24 log points across 4 epochs)
- Evaluation metrics: Every 10 steps (~4-5 evaluation runs across training)
- First log: Immediately at step 1 (thanks to `logging_first_step=True`)
- Total training steps: ~48 steps (for 4 epochs with your dataset size)

### Metrics Naming Convention
- Hugging Face uses prefixes: `train_loss`, `eval_loss`
- Aim callback converts these to contexts: `loss` with `subset='train'` or `subset='eval'`
- Group by `context.subset` to separate them in charts

## Questions?

- **Aim Documentation**: https://aimstack.readthedocs.io/
- **Aim GitHub**: https://github.com/aimhubio/aim
- **Hugging Face Integration**: Check `aim.sdk.adapters.hugging_face.AimCallback` source code
