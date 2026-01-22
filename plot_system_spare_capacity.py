import numpy as np
import matplotlib.pyplot as plt

# City list
cities = ['aachen', 'bonn', 'dresden', 'frankfurt', 'munich']

# Load data for all cities
city_data = {}
city_max_traffic = {}
city_traffic_by_time = {}

print("Loading data for all cities...")
for city in cities:
    data = np.load(f'data/synthetic-traffic/tmap_model-{city}.npy')
    city_data[city] = data
    
    # Calculate total traffic for each timestep (sum over all grid cells)
    time_steps = data.shape[0]
    traffic_by_time = np.array([np.sum(data[t, :, :]) for t in range(time_steps)])
    city_traffic_by_time[city] = traffic_by_time
    
    # Find max total traffic for this city
    city_max_traffic[city] = np.max(traffic_by_time)
    
    print(f"  {city}: shape={data.shape}, max_traffic={city_max_traffic[city]:.4f}")

# Calculate total max capacity across all cities
total_max_capacity = sum(city_max_traffic.values())
print(f"\nTotal max capacity (sum of all city maxes): {total_max_capacity:.4f}")

# Select first 24 hours for analysis
hours = 24
spare_percentage = []

for t in range(hours):
    # Calculate current total traffic across all cities at time t
    current_total = sum([city_traffic_by_time[city][t] for city in cities])
    
    # Calculate spare capacity
    spare = total_max_capacity - current_total
    
    # Convert to percentage
    spare_pct = (spare / total_max_capacity) * 100
    spare_percentage.append(spare_pct)

spare_percentage = np.array(spare_percentage)

# Create figure
fig, ax = plt.subplots(figsize=(18, 9))

# Plot spare capacity over 24 hours
time_hours = range(hours)
ax.plot(time_hours, spare_percentage, linewidth=4, color='#E63946', 
        marker='o', markersize=8, markerfacecolor='white', 
        markeredgewidth=3, markeredgecolor='#E63946')

# Fill area under the curve
ax.fill_between(time_hours, 0, spare_percentage, alpha=0.3, color='#F18F01')

# Add grid
ax.grid(True, alpha=0.4, linestyle='--', linewidth=1.2, axis='both')

# Styling
ax.set_xlabel('Time (Hour)', fontsize=32, fontweight='bold')
ax.set_ylabel('Spare Capacity (%)', fontsize=32, fontweight='bold')
ax.set_title(f'System-wide Spare Capacity Across {len(cities)} Cities', 
             fontsize=32, fontweight='bold', pad=20)

# Set tick parameters
ax.tick_params(axis='both', which='major', labelsize=32)

# Set x-axis ticks
ax.set_xticks(range(0, hours, 2))
ax.set_xticklabels([f'{h}h' for h in range(0, hours, 2)])

# Set y-axis limits
ax.set_xlim(-0.5, hours - 0.5)
ax.set_ylim(0, max(spare_percentage) * 1.1)

# Add statistics annotations
min_spare = np.min(spare_percentage)
max_spare = np.max(spare_percentage)
avg_spare = np.mean(spare_percentage)
min_idx = np.argmin(spare_percentage)
max_idx = np.argmax(spare_percentage)

# Add statistics text box
stats_text = f'Min: {min_spare:.1f}% (Hour {min_idx})\n'
stats_text += f'Max: {max_spare:.1f}% (Hour {max_idx})\n'
stats_text += f'Avg: {avg_spare:.1f}%'

ax.text(0.5, 0.6, stats_text, transform=ax.transAxes,
        fontsize=22, verticalalignment='top', horizontalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

plt.tight_layout()
plt.savefig('system_spare_capacity.pdf', dpi=300, bbox_inches='tight')
plt.savefig('system_spare_capacity.png', dpi=300, bbox_inches='tight')

print(f"\n✓ Saved: system_spare_capacity.pdf and .png")
print(f"✓ Spare capacity statistics:")
print(f"   Min: {min_spare:.2f}% at hour {min_idx}")
print(f"   Max: {max_spare:.2f}% at hour {max_idx}")
print(f"   Average: {avg_spare:.2f}%")
print(f"   Range: {max_spare - min_spare:.2f}%")

plt.show()
