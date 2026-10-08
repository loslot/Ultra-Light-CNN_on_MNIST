# Ultra-Light CNN on MNIST

Goal: reach about 97.5% - 98.5% accuracy on a 15% gradient-noise test set using only 2500 / 5000 images.

How the accuracy is pushed up (while keeping the model ultra-light, ~23K parameters):
- Small 3-block CNN that keeps spatial information (3x3 pooled feature map) instead of collapsing to 1 value per channel
- Test-time augmentation: predictions are averaged over the image and 8 one-pixel shifts
- Training-only augmentation (small shifts, rotations, scale, shear) plus gradient noise and blur that match the test-set corruption, so the model is robust to noisy test images
- SGD (learning rate 0.01, momentum, weight decay) with cosine learning-rate decay, plus dropout and batch norm

Ultra-Light CNN (~23K parameters) trained on MNIST with the following setup:

| Item | Setting |
|---|---|
| Dataset | MNIST (70,000 = 60,000 train + 10,000 test merged) |
| Subset size | 2500 and 5000 |
| Seeding | 42 |
| Split | Train 70% / Test 20% (+15% gradient noise) / Validate 10% |
| Epochs | 50 and 100 |
| Learning rate | 0.01 |
| Model | Ultra-Light CNN (chosen model, no duplicate) |
| Evaluation | Training vs validation loss, result plot of 10 digits, confusion matrix |

