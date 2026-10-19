# Project Plan: CNN Image Classifier

## Goal
Build and train a small convolutional neural network (CNN) in Python that
classifies images into 10 object categories. The classifier is a simple
example of robot vision: a robot camera needs to recognise what it sees
before it can decide what to do.

## Dataset
CIFAR-10, a standard benchmark dataset:
- 60,000 colour images, each 32x32 pixels
- 50,000 for training and 10,000 for testing
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Downloads automatically through torchvision

## Tools
- Python 3, PyTorch and torchvision (model and data)
- NumPy and Matplotlib (arrays and plots)
- scikit-learn (confusion matrix and metrics)
- Git and GitHub (version control)
- Trello (project management)

## Approach
1. Load and inspect the dataset
2. Preprocess: normalise pixel values, split into train and test sets
3. Build and train a baseline CNN
4. Evaluate with accuracy and a confusion matrix
5. Improve with data augmentation
6. Predict on my own photos

## Milestones
| Date | Milestone |
|---|---|
| 21 Oct | Dataset loading and sample images |
| 23 Oct | Preprocessing |
| 26 Oct | Baseline CNN trained |
| 29 Oct | Evaluation (accuracy, confusion matrix) |
| 2 Nov | Predictions on my own photos |
| 6 Nov | Data augmentation |
| 10 Nov | Final README, cleanup and retrospective |

## Success criteria
- The model trains without errors and the results are reproducible
- Test accuracy clearly above random guessing (10%)
- Results recorded in docs/results.md
- Regular commits, with one Pull Request per Trello card

## Risks
- Training may be slow on a laptop (fallback: Google Colab)
- Time is short, so data augmentation is the first feature to drop

## Project management
Tasks are tracked on Trello (one card, one branch, one Pull Request).
Code is stored on GitHub.
