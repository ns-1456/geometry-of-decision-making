# The Geometry of Decision Making 🌀

> **"How does a neural network actually *think*?"**

This project isn't about getting 99% accuracy on MNIST. It's about **seeing** the math.

We take a notoriously difficult dataset—the **Two Spirals**—and pit four different algorithms against it. The goal? To visualize exactly how they carve up space to make decisions.

## The Challenge: Two Spirals
Imagine two interleaving spirals of points, Red and Blue.
- A **Linear Model** tries to draw a straight line between them. (Spoiler: It fails hilariously).
- A **Deep Neural Network** tries to *fold* the paper itself until the spirals can be separated by a straight cut.

## The Contestants 🥊

1.  **The Linear Baseline (Logistic Regression)**
    *   *The Optimist.* It believes everything in life can be solved with a straight line. It is wrong.
2.  **The Kernel Machine (SVM with RBF)**
    *   *The Mathematician.* It projects the data into infinite dimensions to find a solution. Elegant, but expensive.
3.  **The Wide Shallow Net (1000 Neurons)**
    *   *The Brute Force.* It uses 1000 tiny straight lines to approximate the curve. It works, but it's jagged and inefficient.
4.  **The Deep Net (3 Layers x 20 Neurons)**
    *   *The Origami Master.* It uses a series of small folds to untangle the spirals smoothly. This is the power of **depth**.

## How to Run It 🏃‍♂️

You'll need Python, PyTorch, and Matplotlib.

```bash
# 1. Clone the repo
git clone https://github.com/ns-1456/geometry-of-decision-making.git
cd geometry-of-decision-making

# 2. Install dependencies (if you haven't already)
pip install torch numpy matplotlib scikit-learn opencv-python

# 3. Run the magic
python main.py
```

## What You'll Get 🎁

The script will generate:
1.  **`two_spirals_preview.png`**: The raw data.
2.  **`1_linear_baseline.png`**: The failed attempt.
3.  **`2_kernel_machine.png`**: The SVM's solution.
4.  **`3_shallow_net.png`** & **`3_shallow_net.mp4`**: The shallow net's jagged solution + training animation.
5.  **`4_deep_net.png`** & **`4_deep_net.mp4`**: The deep net's smooth solution + training animation.

## The Theory 📚
If you want to nerd out, this project is grounded in:
*   **Universal Approximation Theorem** (Why shallow nets *can* work).
*   **Telgarsky’s "Benefits of Depth"** (Why deep nets work *better*).
*   **Manifold Hypothesis** (The idea of "unfolding" data).

---
*Built with 🧠 and PyTorch.*
