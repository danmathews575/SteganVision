import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def generate_graph():
    # Path to the metrics CSV
    csv_path = r'C:\MajorP\image_steganography\outputs\evaluation\results\metrics_summary.csv'
    output_path = r'C:\MajorP\image_steganography\comparison_graph.png'

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    # Load data
    df = pd.read_csv(csv_path)
    
    # Filter for relevant metrics
    metrics_to_plot = ['psnr', 'secret_psnr', 'ssim']
    df_filtered = df[df['metric'].isin(metrics_to_plot)].copy()

    # Rename metrics for better labels
    label_map = {
        'psnr': 'Cover PSNR (dB)',
        'secret_psnr': 'Secret PSNR (dB)',
        'ssim': 'SSIM'
    }
    df_filtered['Metric Label'] = df_filtered['metric'].map(label_map)

    # Set up the plot
    sns.set_theme(style="whitegrid")
    
    # Create figure with 2 subplots (one for PSNRs, one for SSIM)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Data for PSNRs
    df_psnr = df_filtered[df_filtered['metric'].isin(['psnr', 'secret_psnr'])]
    
    # Melt for seaborn
    df_psnr_melted = df_psnr.melt(
        id_vars=['Metric Label'], 
        value_vars=['cnn_mean', 'gan_mean'], 
        var_name='Model', 
        value_name='Value'
    )
    df_psnr_melted['Model'] = df_psnr_melted['Model'].map({'cnn_mean': 'CNN Baseline', 'gan_mean': 'SteganVision (GAN)'})

    # Plot PSNRs
    sns.barplot(data=df_psnr_melted, x='Metric Label', y='Value', hue='Model', ax=ax1, palette=['#95a5a6', '#2ecc71'])
    ax1.set_title('Quality Metrics (PSNR)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('')
    ax1.set_ylabel('PSNR (dB)')
    ax1.legend(title='Model')
    
    # Add value labels
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.1f')

    # Data for SSIM
    df_ssim = df_filtered[df_filtered['metric'] == 'ssim']
    df_ssim_melted = df_ssim.melt(
        id_vars=['Metric Label'], 
        value_vars=['cnn_mean', 'gan_mean'], 
        var_name='Model', 
        value_name='Value'
    )
    df_ssim_melted['Model'] = df_ssim_melted['Model'].map({'cnn_mean': 'CNN Baseline', 'gan_mean': 'SteganVision (GAN)'})

    # Plot SSIM
    sns.barplot(data=df_ssim_melted, x='Metric Label', y='Value', hue='Model', ax=ax2, palette=['#95a5a6', '#2ecc71'])
    ax2.set_title('Structural Similarity (SSIM)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('')
    ax2.set_ylabel('SSIM Score')
    ax2.set_ylim(0.9, 1.0) # Zoom in to show difference
    ax2.legend(title='Model')

    # Add value labels
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.3f')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Graph saved to {output_path}")

if __name__ == "__main__":
    generate_graph()
