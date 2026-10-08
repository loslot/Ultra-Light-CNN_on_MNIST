# Ultra-Light CNN on MNIST

A very small convolutional neural network (about 23K parameters) that recognizes handwritten digits from only 2,500 or 5,000 training images, and still reaches about **98.8–98.9% accuracy** on test images with 15% gradient noise. Built with PyTorch.

## Setup (from the assignment)

| Item | Setting |
|---|---|
| Dataset | MNIST, 70,000 images (60,000 train + 10,000 test merged) |
| Subset sizes | 2,500 and 5,000 images |
| Seed | 42 |
| Split | 70% train / 20% test (with 15% gradient noise) / 10% validation |
| Epochs | 50 and 100 |
| Learning rate | 0.01 |
| Model | Ultra-Light CNN |

Every combination of size and epochs is trained, so there are 4 experiments.

## Model

Three small convolution blocks (16 → 32 → 48 channels, with batch norm, ReLU and max pooling), a 3×3 pooled feature map, dropout, and one linear layer.

How accuracy is improved while keeping the model tiny:
- **Augmentation (training only):** small shifts, rotations, scale and shear, plus gradient noise and blur that match the test noise.
- **Optimizer:** SGD with learning rate 0.01, momentum, weight decay and cosine learning-rate decay.
- **Test-time augmentation (TTA):** each test image's prediction is averaged over 9 one-pixel shifts.

## Results

Accuracy values are approximate readings from `graph_summary_comparison.png`. The exact numbers are in `output/results.csv`.

| Size / epochs | Test acc (15% noise) | Test acc (15% noise + TTA) |
|---|---|---|
| 2500 / 50 | about 98.2% | about 98.8% |
| 2500 / 100 | about 98.4% | about 98.8% |
| 5000 / 50 | about 98.7% | about 98.8% |
| 5000 / 100 | about 98.6% | about 98.9% |

The best run is 5,000 images with 100 epochs. The test sets are small (500 and 1,000 images), so differences of a few tenths of a percent are within noise.

## How to run

1. Use **Python 3.12** and create a virtual environment:
   ```bash
   uv venv --python 3.12 .venv
   source .venv/bin/activate
   uv pip install torch torchvision numpy pandas matplotlib scikit-learn ipykernel
   ```
2. Open `Ultra_Light_CNN_v6.ipynb` in VS Code or Jupyter and select the `.venv` kernel.
3. Click **Run All**. MNIST downloads automatically into `data/` on the first run. Training takes about 10 to 15 minutes on one CPU core.

## Output (saved in `output/`)

For each of the 4 runs:
- `epochs_size*_epochs*.csv`: loss, MAE, MSE, gap and time for every epoch
- `graph_loss_*.png`: training vs validation loss
- `graph_confusion_*.png`: confusion matrix
- `graph_samples_*.png`: noisy test samples with predictions
- `result_10_digits_*.png`: Original / Test / Result for digits 0–9

Overall:
- `graph_summary_comparison.png`: accuracy and loss for all runs
- `results.csv`: summary table
- `ultra_light_cnn_size*_ep*.pt`: the best trained model
