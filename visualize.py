import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_decision_boundary(model, X, y, title, filename):
    """
    Plots the decision boundary of a model.
    
    Args:
        model: The trained model (must have a predict method or be a PyTorch module).
        X: Feature data.
        y: Labels.
        title: Plot title.
        filename: Output filename.
    """
    # Define the grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Flatten the grid to feed into the model
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Get predictions
    if hasattr(model, 'predict'):
        # Scikit-learn models
        Z = model.predict(grid_points)
    else:
        # PyTorch models
        model.eval()
        with torch.no_grad():
            inputs = torch.FloatTensor(grid_points)
            outputs = model(inputs)
            Z = (outputs > 0.5).float().numpy().flatten()
            
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu_r, edgecolors='k', s=20)
    plt.title(title)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot to {filename}")

def save_frame(model, X, y, epoch, folder):
    """
    Saves a single frame of the decision boundary.
    """
    filename = f"{folder}/frame_{epoch:04d}.png"
    plot_decision_boundary(model, X, y, f"Epoch {epoch}", filename)

def create_animation(folder, output_filename):
    """
    Creates an MP4 video from a folder of images using OpenCV.
    """
    import cv2
    import os
    
    filenames = sorted([f for f in os.listdir(folder) if f.endswith('.png')])
    if not filenames:
        print(f"No images found in {folder}")
        return

    # Read the first image to get dimensions
    first_frame = cv2.imread(os.path.join(folder, filenames[0]))
    height, width, layers = first_frame.shape
    
    # Define the codec and create VideoWriter object
    # mp4v is a safe choice for .mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_filename, fourcc, 10.0, (width, height))
    
    for filename in filenames:
        img = cv2.imread(os.path.join(folder, filename))
        out.write(img)
        
    out.release()
    print(f"Saved animation to {output_filename}")
