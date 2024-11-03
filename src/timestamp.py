from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline


@dataclass
class ModernChartStyle:
    """Configuration for modern chart appearance."""
    figure_size: tuple[int, int] = (14, 8)
    primary_color: str = '#4361ee'
    secondary_color: str = '#e74c3c'
    highlight_color: str = '#2ecc71'
    background_color: str = '#f8f9fa'
    grid_color: str = '#dee2e6'
    text_color: str = '#2d3436'
    title_size: int = 20
    label_size: int = 14
    tick_size: int = 12
    annotation_size: int = 10
    dpi: int = 300


def create_non_negative_trend(x, y, smoothing_factor=300):
    """Create a smooth, non-negative trend line."""
    # Create a smoother line by adding points before and after
    x_extended = np.concatenate(([x[0] - 1], x, [x[-1] + 1]))
    y_extended = np.concatenate(([y[0]], y, [y[-1]]))

    # Create the spline function
    spl = make_interp_spline(x_extended, y_extended, k=3)

    # Generate smooth points
    x_smooth = np.linspace(x[0], x[-1], smoothing_factor)
    y_smooth = spl(x_smooth)

    # Ensure non-negative values
    y_smooth = np.maximum(y_smooth, 0)

    return x_smooth, y_smooth


def visualize_hourly_activity(df: pd.DataFrame,
                                     output_path: str,
                                     style: Optional[ModernChartStyle] = None) -> None:
    """
    Create a modern yet clear visualization of hourly activity patterns.

    Args:
        df: DataFrame containing message data with 'timestamp' column
        output_path: Path to save the visualization
        style: Optional styling configuration
    """
    style = style or ModernChartStyle()

    # Prepare the data
    df = df.copy()
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    hourly_counts = df['hour'].value_counts().sort_index().reset_index()
    hourly_counts.columns = ['hour', 'count']

    # Calculate statistics
    total_messages = len(df)
    peak_idx = hourly_counts['count'].idxmax()
    peak_hour = hourly_counts.loc[peak_idx, 'hour']
    peak_count = hourly_counts.loc[peak_idx, 'count']
    start_date = pd.to_datetime(df['timestamp']).min()
    end_date = pd.to_datetime(df['timestamp']).max()

    # Set up the plot with modern styling
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=style.figure_size,
                           facecolor=style.background_color)
    ax.set_facecolor(style.background_color)

    # Create the main bar plot with gradient color
    bars = ax.bar(hourly_counts['hour'],
                  hourly_counts['count'],
                  color=style.primary_color,
                  alpha=0.7)

    # Add a smooth, non-negative trend line
    x_smooth, y_smooth = create_non_negative_trend(
        hourly_counts['hour'].values,
        hourly_counts['count'].values
    )
    ax.plot(x_smooth, y_smooth,
            color=style.secondary_color,
            linewidth=2,
            label='Activity Trend')

    # Highlight peak activity
    peak_bar = bars[peak_hour]
    peak_bar.set_color(style.highlight_color)
    peak_bar.set_alpha(0.9)

    # Add peak annotation
    ax.annotate(f'Peak Activity: {peak_count} messages',
                xy=(peak_hour, peak_count),
                xytext=(10, 10),
                textcoords='offset points',
                ha='left',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.5',
                          fc=style.background_color,
                          alpha=0.8,
                          ec=style.highlight_color),
                color=style.text_color,
                fontsize=style.annotation_size,
                arrowprops=dict(arrowstyle='->',
                                connectionstyle='arc3,rad=0.2',
                                color=style.highlight_color))

    # Customize grid
    ax.grid(True, axis='y', alpha=0.3, color=style.grid_color)
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(style.grid_color)
    ax.spines['bottom'].set_color(style.grid_color)

    # Customize title and labels
    ax.set_title('Message Activity Throughout the Day',
                 pad=20,
                 fontsize=style.title_size,
                 color=style.text_color)
    ax.set_xlabel('Hour of Day',
                  fontsize=style.label_size,
                  color=style.text_color)
    ax.set_ylabel('Number of Messages',
                  fontsize=style.label_size,
                  color=style.text_color)

    # Customize ticks
    ax.set_xticks(range(24))
    ax.set_xticklabels([f'{hour:02d}:00' for hour in range(24)],
                       rotation=45,
                       ha='right',
                       fontsize=style.tick_size)
    ax.tick_params(axis='both', colors=style.text_color)

    # Add summary statistics
    summary_text = (
        f'Total Messages: {total_messages:,}\n'
        f'Period: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}'
    )
    plt.figtext(0.99, 0.02,
                summary_text,
                ha='right',
                va='bottom',
                fontsize=style.annotation_size,
                color=style.text_color)

    # Add legend
    ax.legend(fontsize=style.annotation_size)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path,
                dpi=style.dpi,
                bbox_inches='tight',
                facecolor=style.background_color)
    plt.close()

    print(f"Modern visualization saved as '{output_path}'")