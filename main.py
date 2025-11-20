import data
import models
import visualize
import numpy as np

def main():
    print("Phase 1: Generating Data...")
    X, y = data.generate_two_spirals(n_points=2000, noise=0.5)
    
    print("Phase 2: Training Models...")
    
    # 1. Linear Baseline
    print("Training Linear Model...")
    linear_model = models.LinearModel()
    linear_model.fit(X, y)
    visualize.plot_decision_boundary(linear_model, X, y, "Linear Baseline (Logistic Regression)", "1_linear_baseline.png")
    
    # 2. Kernel Machine
    print("Training Kernel Machine (SVM)...")
    kernel_model = models.KernelModel(gamma='scale', C=10) # Increased C for tighter fit
    kernel_model.fit(X, y)
    visualize.plot_decision_boundary(kernel_model, X, y, "Kernel Machine (SVM RBF)", "2_kernel_machine.png")
    
    # 3. Wide Shallow Net
    print("Training Wide Shallow Net...")
    import os
    if not os.path.exists("shallow_frames"):
        os.makedirs("shallow_frames")
        
    def shallow_callback(model, epoch):
        visualize.save_frame(model, X, y, epoch, "shallow_frames")
        
    shallow_net = models.ShallowNet()
    shallow_net = models.train_torch_model(shallow_net, X, y, epochs=2000, lr=0.01, callback=shallow_callback)
    visualize.plot_decision_boundary(shallow_net, X, y, "Wide Shallow Net (1000 neurons)", "3_shallow_net.png")
    visualize.create_animation("shallow_frames", "3_shallow_net.mp4")
    
    # 4. Deep Net
    print("Training Deep Net...")
    if not os.path.exists("deep_frames"):
        os.makedirs("deep_frames")
        
    def deep_callback(model, epoch):
        visualize.save_frame(model, X, y, epoch, "deep_frames")

    deep_net = models.DeepNet()
    deep_net = models.train_torch_model(deep_net, X, y, epochs=2000, lr=0.01, callback=deep_callback)
    visualize.plot_decision_boundary(deep_net, X, y, "Deep Net (3x20 neurons)", "4_deep_net.png")
    visualize.create_animation("deep_frames", "4_deep_net.mp4")
    
    print("Phase 3: Visualization Complete. Check the output images.")

if __name__ == "__main__":
    main()
