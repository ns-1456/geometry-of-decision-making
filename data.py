import numpy as np
import matplotlib.pyplot as plt

def generate_two_spirals(n_points=2000, noise=0.5):
    """
    Generates a two-spirals dataset.
    
    Args:
        n_points (int): Total number of points to generate.
        noise (float): Standard deviation of Gaussian noise added to the data.
        
    Returns:
        X (np.ndarray): (n_points, 2) array of feature coordinates.
        y (np.ndarray): (n_points,) array of class labels (0 or 1).
    """
    n = n_points // 2
    theta = np.sqrt(np.random.rand(n)) * 4 * np.pi  # theta ranges from 0 to 4pi
    
    # Spiral 1 (Class 0)
    r_a = theta + np.pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
    x_a = data_a + np.random.randn(n, 2) * noise
    
    # Spiral 2 (Class 1)
    r_b = -theta - np.pi
    data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
    x_b = data_b + np.random.randn(n, 2) * noise
    
    res_a = np.append(x_a, np.zeros((n, 1)), axis=1)
    res_b = np.append(x_b, np.ones((n, 1)), axis=1)
    
    res = np.append(res_a, res_b, axis=0)
    np.random.shuffle(res)
    
    X = res[:, :2]
    y = res[:, 2]
    
    return X, y

if __name__ == "__main__":
    # Quick test to visualize
    X, y = generate_two_spirals()
    plt.figure(figsize=(8, 8))
    plt.scatter(X[y==0, 0], X[y==0, 1], c='r', s=10, label='Class 0')
    plt.scatter(X[y==1, 0], X[y==1, 1], c='b', s=10, label='Class 1')
    plt.title("Two Spirals Dataset")
    plt.legend()
    plt.savefig("two_spirals_preview.png")
    print("Preview saved to two_spirals_preview.png")
