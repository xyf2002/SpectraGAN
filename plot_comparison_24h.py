#!/usr/bin/env python3
"""
Plot 24-hour traffic pattern for SpectraGAN and IMDEA datasets side by side.
Both datasets are normalized by their peak values.

X-axis: Time (0-24 hours)  
Y-axis: Traffic (normalized by peak, 0-1)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

# Set font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_spectragen_data(city='aachen', pixel_x=10, pixel_y=10, day=0):
    """Load 24-hour data from SpectraGAN dataset."""
    data_file = f'data/synthetic-traffic/tmap_model-{city}.npy'
    
    if not os.path.exists(data_file):
        print(f"❌ SpectraGAN data not found: {data_file}")
        return None
    
    data = np.load(data_file)
    
    # Check coordinate validity
    if pixel_x >= data.shape[1] or pixel_y >= data.shape[2]:
        print(f"❌ SpectraGAN coordinates out of range!")
        print(f"  Valid range: x=[0, {data.shape[1]-1}], y=[0, {data.shape[2]-1}]")
        return None
    
    # Extract 24-hour data
    start_idx = day * 24
    end_idx = start_idx + 24
    
    if end_idx > data.shape[0]:
        print(f"❌ SpectraGAN time index out of range!")
        return None
    
    traffic_24h = data[start_idx:end_idx, pixel_x, pixel_y]
    
    # Normalize by peak
    peak = np.max(traffic_24h)
    traffic_24h_normalized = traffic_24h / peak if peak > 0 else traffic_24h
    
    return {
        'data': traffic_24h_normalized,
        'raw_data': traffic_24h,
        'peak': peak,
        'city': city,
        'pixel': (pixel_x, pixel_y),
        'day': day
    }

def load_imdea_data(zone='I', frequency='f1815', metric='uplink', day_index=0):
    """Load 24-hour data from IMDEA dataset."""
    
    metric_map = {'uplink': 'uplink', 'downlink': 'downlink', 'users': 'users'}
    metric_name = metric_map.get(metric, 'uplink')
    zone_folder = f'zoneI' if zone == 'I' else f'zoneII'
    
    file_path = f'imdea_dataset/{zone_folder}/{frequency}/{metric_name}_{zone}{frequency[-4:]}_s.csv'
    
    if not os.path.exists(file_path):
        print(f"❌ IMDEA data not found: {file_path}")
        return None
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ Error reading IMDEA file: {e}")
        return None
    
    if df.empty:
        print(f"❌ IMDEA data is empty")
        return None
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Get data range
    min_time = df['timestamp'].min()
    max_time = df['timestamp'].max()
    
    print(f"  IMDEA data range: {min_time} to {max_time}")
    print(f"  Data points: {len(df)}")
    
    # Extract 24-hour window (starting from a day_index 24-hour period)
    # We'll use the first 24*3600 seconds worth of data, grouped by hour
    start_time = min_time
    end_time = min_time + timedelta(days=1)
    
    df_24h = df[(df['timestamp'] >= start_time) & (df['timestamp'] < end_time)].copy()
    
    if len(df_24h) == 0:
        print(f"❌ No data found in 24-hour window")
        return None
    
    # Resample to hourly data
    df_24h.set_index('timestamp', inplace=True)
    df_hourly = df_24h['tbs_sum'].resample('h').sum()
    
    # Get 24 hours
    traffic_24h = df_hourly.values[:24]
    
    if len(traffic_24h) < 24:
        print(f"⚠ Only {len(traffic_24h)} hours available, padding with NaN")
        traffic_24h = np.pad(traffic_24h, (0, 24 - len(traffic_24h)), constant_values=np.nan)
    
    # Normalize by peak
    peak = np.nanmax(traffic_24h)
    traffic_24h_normalized = traffic_24h / peak if peak > 0 else traffic_24h
    
    return {
        'data': traffic_24h_normalized,
        'raw_data': traffic_24h,
        'peak': peak,
        'zone': zone,
        'frequency': frequency,
        'metric': metric,
        'start_time': start_time
    }

def plot_comparison(spectragen_config=None, imdea_config=None):
    """Plot 24-hour traffic from both datasets side by side."""
    
    # Default configs
    if spectragen_config is None:
        spectragen_config = {
            'city': 'aachen',
            'pixel_x': 6,
            'pixel_y': 5,  # Peak traffic pixel
            'day': 0
        }
    
    if imdea_config is None:
        imdea_config = {
            'zone': 'I',
            'frequency': 'f1815',
            'metric': 'uplink',
            'day_index': 0
        }
    
    # Load both datasets
    print("Loading SpectraGAN data...")
    sg_data = load_spectragen_data(
        spectragen_config['city'],
        spectragen_config['pixel_x'],
        spectragen_config['pixel_y'],
        spectragen_config['day']
    )
    
    if sg_data is None:
        print("❌ Failed to load SpectraGAN data")
        return
    
    print(f"✓ SpectraGAN loaded: {spectragen_config['city']}")
    print(f"  Peak (before normalization): {sg_data['peak']:.6f}")
    
    print("\nLoading IMDEA data...")
    imdea_data = load_imdea_data(
        imdea_config['zone'],
        imdea_config['frequency'],
        imdea_config['metric'],
        imdea_config['day_index']
    )
    
    if imdea_data is None:
        print("❌ Failed to load IMDEA data")
        return
    
    print(f"✓ IMDEA loaded: Zone {imdea_config['zone']}, {imdea_config['frequency']}")
    print(f"  Peak (before normalization): {imdea_data['peak']:.0f}")
    
    # Create comparison plot
    hours = np.arange(0, 24)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ===== SpectraGAN subplot =====
    ax1.plot(hours, sg_data['data'], linewidth=2.5, color='#2E86AB', 
             marker='o', markersize=5)
    ax1.fill_between(hours, sg_data['data'], alpha=0.3, color='#2E86AB')
    
    ax1.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Normalized Traffic', fontsize=12, fontweight='bold')
    ax1.set_xlim(-0.5, 23.5)
    ax1.set_ylim(0, 1.1)
    ax1.set_xticks(np.arange(0, 24, 2))
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # ===== IMDEA subplot =====
    ax2.plot(hours, imdea_data['data'], linewidth=2.5, color='#E63946', 
             marker='s', markersize=5)
    ax2.fill_between(hours, imdea_data['data'], alpha=0.3, color='#E63946')
    
    ax2.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax2.set_xlim(-0.5, 23.5)
    ax2.set_ylim(0, 1.1)
    ax2.set_xticks(np.arange(0, 24, 2))
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # Align y-axes
    ax1.set_ylim(0, 1.1)
    ax2.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    # Save figure
    output_file = f'plot_single_cell_24h_comparison_{spectragen_config["city"]}_imdea.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure saved: {output_file}")
    
    plt.show()

if __name__ == '__main__':
    # Default comparison: SpectraGAN Aachen peak pixel vs IMDEA Zone I
    sg_config = {
        'city': 'aachen',
        'pixel_x': 6,      # Peak traffic pixel
        'pixel_y': 5,      # Peak traffic pixel
        'day': 0
    }
    
    imdea_config = {
        'zone': 'I',
        'frequency': 'f1815',
        'metric': 'uplink',
        'day_index': 0
    }
    
    plot_comparison(sg_config, imdea_config)
    
    # You can also customize the plot by changing the configs:
    # sg_config = {'city': 'frankfurt', 'pixel_x': 25, 'pixel_y': 39, 'day': 0}
    # imdea_config = {'zone': 'I', 'frequency': 'f2650', 'metric': 'downlink', 'day_index': 0}
    # plot_comparison(sg_config, imdea_config)
