import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Load Aachen data
data = np.load('data/synthetic-traffic/tmap_model-aachen.npy')
# data shape: [time, height, width]
time_steps, height, width = data.shape

print(f"Data shape: {data.shape}")
print(f"Time steps: {time_steps}, Height: {height}, Width: {width}")

# Create heatmap where:
# - X-axis: time (hours)
# - Y-axis: traffic volume (pixel average)
# - Value: traffic for each pixel at each time

# Flatten the spatial dimensions: each pixel becomes a row
flattened_data = data.reshape(time_steps, height * width)  # [time_steps, num_pixels]
print(f"Flattened data shape: {flattened_data.shape}")

# Normalize each pixel independently so peak traffic is bright (1.0) for each pixel
pixel_mins = np.min(flattened_data, axis=0, keepdims=True)
pixel_maxs = np.max(flattened_data, axis=0, keepdims=True)
flattened_data = (flattened_data - pixel_mins) / (pixel_maxs - pixel_mins + 1e-8)

# Also normalize the original 3D data for individual pixel plots
for i in range(height):
    for j in range(width):
        pixel_data = data[:, i, j]
        pixel_min = np.min(pixel_data)
        pixel_max = np.max(pixel_data)
        if pixel_max > pixel_min:
            data[:, i, j] = (pixel_data - pixel_min) / (pixel_max - pixel_min)

print(f"Data normalized per pixel (peak=1.0 for each pixel)")

# Calculate average traffic for each pixel
pixel_avg = np.mean(flattened_data, axis=0)

# Sort pixels by average traffic (descending)
sorted_indices = np.argsort(-pixel_avg)  # negative for descending order
sorted_data = flattened_data[:, sorted_indices]  # [time_steps, num_pixels_sorted]
sorted_avg = pixel_avg[sorted_indices]

print(f"Sorted by traffic volume (descending)")

# Create figure
fig, ax = plt.subplots(figsize=(18, 10))

# Plot heatmap
# Transpose so time is on x-axis and pixels (sorted by traffic) on y-axis
im = ax.imshow(sorted_data.T, aspect='auto', cmap='viridis', 
               interpolation='nearest', origin='upper')

# Set labels
ax.set_xlabel('Hour (Time Step)', fontsize=12, fontweight='bold')
ax.set_ylabel('Traffic Volume (Pixel Average)', fontsize=12, fontweight='bold')

# Set x-axis ticks to show hours
ax.set_xticks(np.arange(0, time_steps, max(1, time_steps // 24)))
ax.set_xticklabels(np.arange(0, time_steps, max(1, time_steps // 24)))

# Set y-axis ticks and labels with traffic volume values
num_labels = 20  # Show 20 labels along y-axis
label_indices = np.linspace(0, len(sorted_avg) - 1, num_labels, dtype=int)
ax.set_yticks(label_indices)
ax.set_yticklabels([f'{sorted_avg[i]:.4f}' for i in label_indices], fontsize=8)

# Add colorbar
cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
cbar.set_label('Traffic Value', fontsize=12, fontweight='bold')

plt.title('Aachen - Traffic Time Series per Pixel (Sorted by Traffic Volume)\n(X-axis: Hour, Y-axis: Traffic Volume, Color: Traffic)', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('aachen_pixel_traffic_timeseries.pdf', dpi=150, bbox_inches='tight')
print("\nSaved: aachen_pixel_traffic_timeseries.pdf")

# Also create a version with spatial structure preserved
fig, axes = plt.subplots(height, width, figsize=(28, 20))
fig.suptitle('Aachen - Time Series for Each Spatial Pixel', fontsize=16, fontweight='bold', y=0.995)

for i in range(height):
    for j in range(width):
        ax = axes[i, j]
        time_series = data[:, i, j]
        
        # Plot time series
        ax.plot(time_series, linewidth=1, color='steelblue')
        ax.fill_between(range(time_steps), time_series, alpha=0.3, color='steelblue')
        
        # Add title with pixel coordinates and average traffic
        avg_traffic = np.mean(time_series)
        ax.set_title(f'({i},{j})\nAvg:{avg_traffic:.4f}', fontsize=7, fontweight='bold')
        
        # Set minimal ticks
        ax.tick_params(labelsize=5)
        ax.set_xlim(0, time_steps - 1)
        
        # Add grid for easier reading
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('aachen_pixel_traffic_timeseries_detailed.pdf', dpi=150, bbox_inches='tight')
print("Saved: aachen_pixel_traffic_timeseries_detailed.pdf")

# Create a combined visualization: heatmap + selected pixel time series
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Top left: full heatmap
ax1 = fig.add_subplot(gs[0, :])
im = ax1.imshow(flattened_data.T, aspect='auto', cmap='viridis', 
                 interpolation='nearest', origin='upper')
ax1.set_xlabel('Hour (Time Step)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Pixel Index', fontsize=11, fontweight='bold')
ax1.set_title('Complete Heatmap: Traffic per Pixel over Time', fontsize=12, fontweight='bold')
cbar1 = plt.colorbar(im, ax=ax1, orientation='vertical', pad=0.02)
cbar1.set_label('Traffic', fontsize=10, fontweight='bold')

# Bottom left: selected pixel with max average traffic
pixel_avg = np.mean(flattened_data, axis=0)
max_pixel_idx = np.argmax(pixel_avg)
max_i, max_j = max_pixel_idx // width, max_pixel_idx % width

ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(data[:, max_i, max_j], linewidth=2, color='red', label=f'Pixel ({max_i}, {max_j})')
ax2.fill_between(range(time_steps), data[:, max_i, max_j], alpha=0.3, color='red')
ax2.set_xlabel('Hour', fontsize=11, fontweight='bold')
ax2.set_ylabel('Traffic', fontsize=11, fontweight='bold')
ax2.set_title(f'Highest Traffic Pixel: ({max_i}, {max_j})', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

# Bottom right: selected pixel with min average traffic
min_pixel_idx = np.argmin(pixel_avg)
min_i, min_j = min_pixel_idx // width, min_pixel_idx % width

ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(data[:, min_i, min_j], linewidth=2, color='blue', label=f'Pixel ({min_i}, {min_j})')
ax3.fill_between(range(time_steps), data[:, min_i, min_j], alpha=0.3, color='blue')
ax3.set_xlabel('Hour', fontsize=11, fontweight='bold')
ax3.set_ylabel('Traffic', fontsize=11, fontweight='bold')
ax3.set_title(f'Lowest Traffic Pixel: ({min_i}, {min_j})', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)

plt.savefig('aachen_pixel_traffic_combined.pdf', dpi=150, bbox_inches='tight')
print("Saved: aachen_pixel_traffic_combined.pdf")

print(f"\nStatistics:")
print(f"  Average traffic per pixel: {np.mean(flattened_data):.6f}")
print(f"  Max pixel average: {np.max(pixel_avg):.6f}")
print(f"  Min pixel average: {np.min(pixel_avg):.6f}")
