import numpy as np
import matplotlib.pyplot as plt

# Select one city's data
city = 'aachen'
data = np.load(f'data/synthetic-traffic/tmap_model-{city}.npy')

# Calculate statistics for each timestep
time_steps = data.shape[0]

# Select a subset of timesteps to show (e.g., 24 hours representing a day)
if time_steps >= 24:
    start_idx = 0
    end_idx = 24
else:
    start_idx = 0
    end_idx = time_steps

# Prepare distribution data for each timestep (flatten spatial dimensions)
distributions = []
median_values = []
max_values = []
for t in range(start_idx, end_idx):
    # Get all spatial positions' traffic values at this timestep
    traffic_values = data[t, :, :].flatten()
    distributions.append(traffic_values)
    median_values.append(np.median(traffic_values))
    max_values.append(np.max(traffic_values))

subset_time = list(range(len(distributions)))
median_values = np.array(median_values)
max_values = np.array(max_values)

# Calculate peak traffic level (max of all violin peaks)
peak_traffic = np.max(max_values)
peak_idx = np.argmax(max_values)

# Create figure
fig, ax = plt.subplots(figsize=(18, 9))

# Plot violin plot for traffic distribution first
parts = ax.violinplot(distributions, positions=subset_time,
                      showmeans=False, showmedians=True, widths=0.7)

# Set violin plot colors
for pc in parts['bodies']:
    pc.set_facecolor('#6CB4EE')
    pc.set_alpha(0.7)
    pc.set_edgecolor('#2E86AB')
    pc.set_linewidth(1.8)
    pc.set_zorder(3)

# Set median line color
parts['cmedians'].set_color('#2E86AB')
parts['cmedians'].set_linewidth(2.5)
parts['cmedians'].set_zorder(4)

# Set whisker colors
for partname in ('cbars', 'cmins', 'cmaxes'):
    if partname in parts:
        parts[partname].set_color('steelblue')
        parts[partname].set_linewidth(1.5)
        parts[partname].set_zorder(3)

# Draw a line connecting all violin peaks (max values)
ax.plot(subset_time, max_values, color='#DC7000', linewidth=2.5, 
        alpha=0.8, zorder=4, linestyle='-')

# Draw peak traffic horizontal line
ax.axhline(y=peak_traffic, color='#DC143C', linestyle='--', 
           linewidth=2.5, alpha=0.9, label='Peak Traffic Level', zorder=5)

# Fill spare compute opportunity area (from max values line to peak line)
ax.fill_between(subset_time, max_values, peak_traffic, 
                alpha=0.5, color='#F18F01', 
                hatch='///', edgecolor='#DC7000', linewidth=0.8,
                label='Spare Compute Opportunity', zorder=2)

# Mark peak with a red star
ax.scatter([subset_time[peak_idx]], [max_values[peak_idx]], 
           color='red', s=200, zorder=5, marker='*', 
           edgecolors='darkred', linewidths=1.5,
           label=f'Peak (Hour {subset_time[peak_idx]})')

# Styling
ax.set_xlabel('Time (Hour)', fontsize=32, fontweight='bold')
ax.set_ylabel('Traffic Volume', fontsize=32, fontweight='bold')
ax.set_title(f'Traffic Distribution and Spare Capacity - {city.upper()}', 
             fontsize=32, fontweight='bold', pad=20)
ax.legend(loc='upper center', fontsize=24, 
          framealpha=0.95, edgecolor='gray', fancybox=True, ncol=3)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
ax.set_xlim(-0.5, len(subset_time) - 0.5)
# Adjust y-axis to not have too much space above
ax.set_ylim(0, peak_traffic * 1.15)

# Set tick label font sizes to be larger
ax.tick_params(axis='both', which='major', labelsize=32)

# Set x-axis ticks to show every hour
ax.set_xticks(subset_time[::2])  # Show every 2 hours to avoid crowding
ax.set_xticklabels([f'{t}h' for t in range(start_idx, end_idx, 2)])

plt.tight_layout()
plt.savefig('traffic_peak_spare_capacity.pdf', dpi=300, bbox_inches='tight')
plt.savefig('traffic_peak_spare_capacity.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: traffic_peak_spare_capacity.pdf and .png")
print(f"✓ Time range: {start_idx}-{end_idx-1} hours")
print(f"✓ Peak traffic (max): {peak_traffic:.4f} at hour {subset_time[peak_idx]}")
print(f"✓ Average spare capacity: {np.mean(peak_traffic - median_values):.4f}")

plt.show()
