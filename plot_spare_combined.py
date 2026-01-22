import numpy as np
import matplotlib.pyplot as plt

# Select one city's data
city = 'aachen'
data = np.load(f'data/synthetic-traffic/tmap_model-{city}.npy')

# data shape: [time_steps, height, width]
time_steps, height, width = data.shape

print(f"Data shape: {data.shape}")
print(f"Time steps: {time_steps}, Height: {height}, Width: {width}")

# Calculate spare capacity for each grid cell
spare_percentage_map = np.zeros((height, width))

for i in range(height):
    for j in range(width):
        # Get traffic values for this grid across all time steps
        grid_traffic = data[:, i, j]
        
        # Peak traffic for this grid
        peak_traffic = np.max(grid_traffic)
        
        if peak_traffic > 0:  # Avoid division by zero
            # Average spare capacity
            avg_spare = np.mean(peak_traffic - grid_traffic)
            
            # Spare as percentage of peak
            spare_percentage = (avg_spare / peak_traffic) * 100
            spare_percentage_map[i, j] = spare_percentage
        else:
            spare_percentage_map[i, j] = 0

# Flatten the spare capacity map for CDF
all_spare_values = spare_percentage_map.flatten()

# Sort for CDF
sorted_spare = np.sort(all_spare_values)

# Calculate CDF
cdf = np.arange(1, len(sorted_spare) + 1) / len(sorted_spare)

# Create figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# ============ LEFT: Heatmap ============
im = ax1.imshow(spare_percentage_map, 
                cmap='YlOrRd',
                vmin=0, vmax=100,
                aspect='auto',
                interpolation='nearest')

# Add colorbar for heatmap
cbar1 = plt.colorbar(im, ax=ax1)
cbar1.set_label('Spare Capacity (%)', fontsize=32, fontweight='bold')
cbar1.ax.tick_params(labelsize=32)

# Heatmap styling
ax1.set_xlabel('Width (Grid X)', fontsize=32, fontweight='bold')
ax1.set_ylabel('Height (Grid Y)', fontsize=32, fontweight='bold')
ax1.set_title(f'(a) Spare Capacity Distribution\n{city.upper()} ({time_steps} timesteps)', 
              fontsize=32, fontweight='bold', pad=20)
ax1.tick_params(axis='both', which='major', labelsize=32)

# ============ RIGHT: CDF ============
ax2.plot(sorted_spare, cdf, linewidth=3, color='#2E86AB', label='CDF')
ax2.fill_between(sorted_spare, 0, cdf, alpha=0.3, color='#6CB4EE')

# Add median line
median_spare = np.median(all_spare_values)
ax2.axvline(median_spare, color='red', linestyle='--', linewidth=2, 
            label=f'Median: {median_spare:.1f}%', alpha=0.8)
ax2.axhline(0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)

# CDF styling
ax2.set_xlabel('Spare Capacity (%)', fontsize=32, fontweight='bold')
ax2.set_ylabel('Cumulative Probability', fontsize=32, fontweight='bold')
ax2.set_title(f'(b) CDF of Spare Capacity\nAcross All {height}×{width} Grid Cells', 
              fontsize=32, fontweight='bold', pad=20)
ax2.legend(loc='lower right', fontsize=24, framealpha=0.95)
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax2.set_xlim(0, 100)
ax2.set_ylim(0, 1)
ax2.tick_params(axis='both', which='major', labelsize=32)

# Add statistics text box
stats_text = f'Min: {np.min(all_spare_values):.1f}%\n'
stats_text += f'Max: {np.max(all_spare_values):.1f}%\n'
stats_text += f'Mean: {np.mean(all_spare_values):.1f}%\n'
stats_text += f'Std: {np.std(all_spare_values):.1f}%'

ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
         fontsize=24, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('spare_capacity_combined.pdf', dpi=300, bbox_inches='tight')
plt.savefig('spare_capacity_combined.png', dpi=300, bbox_inches='tight')

print(f"\n✓ Saved: spare_capacity_combined.pdf and .png")
print(f"✓ Combined heatmap and CDF in one figure")
print(f"✓ Spare capacity range: {np.min(all_spare_values):.2f}% - {np.max(all_spare_values):.2f}%")
print(f"✓ Average spare capacity: {np.mean(all_spare_values):.2f}%")
print(f"✓ Median spare capacity: {np.median(all_spare_values):.2f}%")

plt.show()
