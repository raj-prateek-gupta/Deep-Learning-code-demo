# Perceptron from Scratch 🧠

This project demonstrates the implementation of a **Perceptron** (the simplest neural network) from scratch using **Python** and **NumPy**.  

The perceptron is a **binary linear classifier** and serves as the foundation of modern deep learning.  
This repository is designed for **students, beginners, and enthusiasts** who want to understand how neural networks work **from the ground up** without relying on external ML libraries.

---

## 📚 Table of Contents
1. [Introduction](#-introduction)  
2. [Theory](#-theory)  
3. [Algorithm](#-algorithm)  
4. [Project Structure](#-project-structure)  
5. [Installation](#-installation)  
6. [Results](#-results)  

---

## 🔍 Introduction
The **Perceptron** is a linear binary classifier introduced by Frank Rosenblatt in 1958.  

It consists of:
- **Input features** (x)  
- **Weights** (w)  
- **Bias** (b)  
- **Activation function** (Step function)  

The perceptron updates weights iteratively based on misclassified points until it finds a decision boundary that separates the classes (if linearly separable).

---

## 🧮 Theory
The perceptron works as follows:

1. **Weighted Sum**:  
   \[
   z = \mathbf{w} \cdot \mathbf{x} + b
   \]

2. **Activation (Step function)**:  
   \[
   \hat{y} =
   \begin{cases}
   1 & \text{if } z \geq 0 \\
   0 & \text{otherwise}
   \end{cases}
   \]

3. **Update Rule**:  
   \[
   w = w + \eta \cdot (y - \hat{y}) \cdot x
   \]  
   \[
   b = b + \eta \cdot (y - \hat{y})
   \]  

Where:
- \( \eta \) = learning rate  
- \( y \) = true label  
- \( \hat{y} \) = predicted label  

---

## ⚙️ Algorithm
**Step 1:** Initialize weights and bias as zeros.  
**Step 2:** For each training sample:
- Compute prediction \( \hat{y} \)  
- Compare with true label \( y \)  
- Update weights and bias if prediction is wrong  

**Step 3:** Repeat for multiple epochs until convergence or no errors remain.  

---

## 📂 Project Structure

perceptron/
│
├── perceptron.py # Main Perceptron implementation with demo dataset
├── README.md # Documentation (this file)
└── requirements.txt # Python dependencies

---

## 💻 Installation
Clone the repository and install dependencies:

```bash
# Clone the main repository
git clone https://github.com/raj-prateek-gupta/Deep-Learning-code-demo.git

# Navigate to perceptron folder
cd Deep-Learning-code-demo/perceptron

# Install required packages
pip install -r requirements.txt

---

## 🖥️ Example Output

When running the demo, you will see output like:

```yaml
Accuracy: 1.0
Weights: [0.9288 0.8256]
Bias: -0.5
Mistakes per epoch: [10, 3, 1, 0]


