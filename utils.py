import numpy as np
import matplotlib.pyplot as plt
import math

def average_euclidean_distance(points1, points2):
    """
    Calculate the Euclidean distances between two sets of 2D points
    and return the average distance.
    """
    # Ensure the points are numpy arrays
    points1 = np.asarray(points1)
    points2 = np.asarray(points2)
    
    # Compute the squared differences
    squared_diff = (points1 - points2) ** 2
    
    # Sum over the x and y dimensions
    distances = np.sqrt(np.sum(squared_diff, axis=1))
    
    # Compute the average distance
    avg_distance = np.mean(distances)
    
    return avg_distance


def visualize(samples,estimated_centers):
    """
    Visualize results, given samples as well as estimated centers.
    """
    
    num_visualizations = len(samples)

    # Ensure half as many columns as rows
    rows = math.ceil(math.sqrt(num_visualizations / 2))  # Rows should be sqrt(2 * num_visualizations)
    cols = math.ceil(rows * 2)  # Columns are half of rows (rounded up)
    
    # Dynamically adjust rows if there are fewer plots than spaces
    actual_rows = math.ceil(num_visualizations / cols)

    fig, axes = plt.subplots(actual_rows, cols, figsize=(15, 15))  # Adjust figsize for better visualization
    axes = axes.flatten()  # Flatten in case of multi-dimensional axes
    
    for i, (sample, estimated_center) in enumerate(zip(samples,estimated_centers)):
        ground_truth = sample["patch_center_crop"]
        
        ax = axes[i]
        ax.imshow(sample["overlay"])
        ax.scatter(estimated_center[0], estimated_center[1], c="red", edgecolors="black")
        ax.scatter(ground_truth[0], ground_truth[1], c="blue", edgecolors="black")
        ax.set_title(f"Visualization {i+1}")
        ax.legend(["Estimated Center", "Actual Center"], fontsize="small")
    
    # remove unused subplots if num_visualizations doesn't fill the grid
    for i in range(num_visualizations, len(axes)):
        axes[i].remove()  # Completely remove unused axes
    
    plt.tight_layout()  # Adjust spacing
    plt.show()