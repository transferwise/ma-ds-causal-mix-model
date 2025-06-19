"""
Utils for the WISE project.
"""
import numpy as np

# Calculate distribution overlap
def calculate_overlap(dist1, dist2, bins=30):
    # Create a common range for both distributions
    min_val = min(np.min(dist1), np.min(dist2))
    max_val = max(np.max(dist1), np.max(dist2))
    
    # Create histograms with the same bins
    hist1, bin_edges = np.histogram(dist1, bins=bins, range=(min_val, max_val), density=True)
    hist2, _ = np.histogram(dist2, bins=bins, range=(min_val, max_val), density=True)
    
    # Calculate the overlap
    overlap = np.sum(np.minimum(hist1, hist2)) * (bin_edges[1] - bin_edges[0])
    return overlap * 100  # Convert to percentage