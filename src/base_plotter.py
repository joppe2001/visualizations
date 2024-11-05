# base_plotter.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DimReductionData:
    """Container for dimensionality reduction results."""
    embedded_data: np.ndarray
    labels: np.ndarray
    explained_variance: Optional[float] = None



class BasePlotter:
    """Utility class for creating consistent visualizations across the project"""

    STYLE_PRESETS = {
        'default': {
            'style': 'seaborn-v0_8-darkgrid',
            'figure_size': (12, 8),
            'title_size': 14,
            'dpi': 300,
            'colors': {
                'positive': 'lightgreen',
                'negative': 'lightcoral',
                'neutral': 'lightgray',
                'line': 'black'
            }
        },
        'minimal': {
            'style': 'seaborn-v0_8-whitegrid',
            'figure_size': (10, 6),
            'title_size': 12,
            'dpi': 300,
            'colors': {
                'positive': '#90EE90',
                'negative': '#F08080',
                'neutral': '#D3D3D3',
                'line': '#333333'
            }
        },
        'dark': {
            'style': 'seaborn-v0_8-dark',
            'figure_size': (12, 8),
            'title_size': 14,
            'dpi': 300,
            'colors': {
                'positive': '#00FF00',
                'negative': '#FF0000',
                'neutral': '#808080',
                'line': '#FFFFFF'
            }
        }
    }

    def __init__(self,
                 preset: str = 'default',
                 figure_size: Optional[Tuple[int, int]] = None,
                 dpi: Optional[int] = None,
                 style: Optional[str] = None):
        """
        Initialize the plotter with given settings or preset.

        Args:
            preset: Style preset ('default', 'minimal', or 'dark')
            figure_size: Optional override for figure size
            dpi: Optional override for DPI
            style: Optional override for matplotlib style
        """
        self.settings = self.STYLE_PRESETS[preset].copy()

        if figure_size:
            self.settings['figure_size'] = figure_size
        if dpi:
            self.settings['dpi'] = dpi
        if style:
            self.settings['style'] = style

        plt.style.use(self.settings['style'])

    def setup_figure(self, title: str) -> Tuple[plt.Figure, plt.Axes]:
        """Create and setup a new figure with consistent styling"""
        fig, ax = plt.subplots(figsize=self.settings['figure_size'])
        fig.suptitle(title, fontsize=self.settings['title_size'])
        return fig, ax

    def save_plot(self, path: str, tight: bool = True) -> None:
        """Save the current plot to file with consistent settings"""
        if tight:
            plt.tight_layout()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=self.settings['dpi'], bbox_inches='tight')
        plt.close()

    def create_time_series(self,
                           data: Union[pd.Series, pd.DataFrame],
                           title: str,
                           ylabel: str,
                           columns: Optional[List[str]] = None,
                           output_path: Optional[str] = None) -> None:
        """
        Create a time series plot with consistent styling.

        Args:
            data: Series or DataFrame to plot
            title: Plot title
            ylabel: Y-axis label
            columns: List of columns to plot (if DataFrame)
            output_path: Optional path to save the plot
        """
        fig, ax = self.setup_figure(title)

        # Handle different input types
        if isinstance(data, pd.Series):
            plot_data = data
            if isinstance(data.index, pd.PeriodIndex):
                plot_index = data.index.astype('datetime64[ns]')
            else:
                plot_index = data.index

            # Plot line
            line = ax.plot(plot_index, plot_data.values,
                           color=self.settings['colors']['line'],
                           linewidth=2)

            # Add zero line if data contains positive and negative values
            if (plot_data > 0).any() and (plot_data < 0).any():
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

                # Fill between
                ax.fill_between(plot_index, plot_data.values, 0,
                                where=(plot_data.values > 0),
                                color=self.settings['colors']['positive'],
                                alpha=0.3)
                ax.fill_between(plot_index, plot_data.values, 0,
                                where=(plot_data.values <= 0),
                                color=self.settings['colors']['negative'],
                                alpha=0.3)

        else:  # DataFrame
            if columns is None:
                columns = data.columns
            for column in columns:
                ax.plot(data.index, data[column], label=column)
            ax.legend()

        # Customize
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=45)

        if output_path:
            self.save_plot(output_path)

    def create_stacked_bar(self,
                           data: pd.DataFrame,
                           labels: List[str],
                           title: str,
                           xlabel: str,
                           horizontal: bool = True,
                           output_path: Optional[str] = None) -> None:
        """Create a stacked bar chart with consistent styling"""
        fig, ax = self.setup_figure(title)

        positions = np.arange(len(data))

        if horizontal:
            plot_func = ax.barh
            ax.set_yticks(positions)
            if isinstance(data.index, (pd.Index, pd.MultiIndex)):
                ax.set_yticklabels(data.index)
        else:
            plot_func = ax.bar
            ax.set_xticks(positions)
            if isinstance(data.index, (pd.Index, pd.MultiIndex)):
                ax.set_xticklabels(data.index, rotation=45)

        left = np.zeros(len(data))
        for label in labels:
            plot_func(positions, data[label], left=left, label=label)
            left += data[label]

        ax.set_xlabel(xlabel)
        ax.legend()

        if output_path:
            self.save_plot(output_path)

    def create_distribution(self,
                            data: pd.Series,
                            title: Union[str, dict[str, str]],
                            xlabel: str,
                            ylabel: str = 'Frequency',
                            bins: int = 50,
                            kde: bool = True,
                            output_path: Optional[str] = None) -> None:
        """Create a distribution plot with consistent styling and sentiment indicator"""
        fig, ax = self.setup_figure(title if isinstance(title, str) else title['title'])

        # Plot the distribution
        sns.histplot(data=data, bins=bins, kde=kde, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Add subtitle with sentiment indicator if provided
        if isinstance(title, dict) and 'subtitle' in title:
            ax.text(0.5, 1.05, title['subtitle'],
                    horizontalalignment='center',
                    transform=ax.transAxes,
                    fontsize=10,
                    fontweight='bold')

        if output_path:
            self.save_plot(output_path)

    def create_heatmap(self,
                       data: pd.DataFrame,
                       title: str,
                       cmap: str = 'YlOrRd',
                       annot: bool = True,
                       fmt: str = '.2f',
                       output_path: Optional[str] = None) -> None:
        """Create a heatmap with consistent styling"""
        fig, ax = self.setup_figure(title)
        sns.heatmap(data, cmap=cmap, annot=annot, fmt=fmt, ax=ax)

        if output_path:
            self.save_plot(output_path)

    def create_barchart(self,
                        data: pd.DataFrame,
                        title: str,
                        xlabel: str,
                        ylabel: str,
                        output_path: Optional[str] = None) -> None:
        """Create bar chart with consistent styling"""
        fig, ax = self.setup_figure(title)

        # Ensure we're using the correct column names from the data
        sns.barplot(
            data=data,
            x='Author',  # Use actual column name from DataFrame
            y='Percentage',  # Use actual column name from DataFrame
            ax=ax
        )

        # Set custom labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if output_path:
            self.save_plot(output_path)

            # Add this method at the end of the BasePlotter class

    # In base_plotter.py or wherever your BasePlotter is defined
    class BasePlotter:
        def create_dim_reduction_plot(self, data, title: str, output_path: str):
            """Create dimensionality reduction plot with error handling."""
            try:
                print(f"Starting to create plot: {title}")
                print(f"Data shape: {data.embedded_data.shape}")
                print(f"Output path: {output_path}")

                plt.figure(figsize=self.figure_size)

                if hasattr(data, 'clusters') and data.clusters is not None:
                    scatter = plt.scatter(
                        data.embedded_data[:, 0],
                        data.embedded_data[:, 1],
                        c=data.clusters,
                        cmap='tab20',
                        alpha=0.6
                    )
                    plt.colorbar(scatter, label='Clusters')
                else:
                    plt.scatter(
                        data.embedded_data[:, 0],
                        data.embedded_data[:, 1],
                        alpha=0.6
                    )

                plt.title(title)
                plt.xlabel('Dimension 1')
                plt.ylabel('Dimension 2')

                if hasattr(data, 'explained_variance') and data.explained_variance is not None:
                    plt.suptitle(f'Explained variance: {data.explained_variance:.2%}')

                plt.tight_layout()

                # Ensure the directory exists
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                print(f"Saving plot to: {output_path}")
                plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
                print(f"Plot saved successfully to: {output_path}")
                plt.close()

                # Verify file was created
                if output_path.exists():
                    print(f"Verified: File exists at {output_path}")
                    print(f"File size: {output_path.stat().st_size} bytes")
                else:
                    print(f"Warning: File was not created at {output_path}")

            except Exception as e:
                print(f"Error in create_dim_reduction_plot: {str(e)}")
                import traceback
                traceback.print_exc()
                raise  # Re-raise the exception for the calling code

    def create_boxplot(self,
                       data: pd.DataFrame,
                       x: str,
                       y: str,
                       title: str,
                       ylabel: str,
                       palette: str = 'husl',
                       output_path: Optional[str] = None) -> None:
        """
        Create a boxplot with consistent styling.

        Args:
            data: DataFrame containing the data
            x: Column name for x-axis (categories)
            y: Column name for y-axis (values)
            title: Plot title
            ylabel: Y-axis label
            palette: Color palette for the boxes
            output_path: Optional path to save the plot
        """
        fig, ax = self.setup_figure(title)

        # Create boxplot using seaborn
        sns.boxplot(
            data=data,
            x=x,
            y=y,
            ax=ax,
            palette=palette,
            width=0.7,
            fliersize=5
        )

        # Customize
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=45)

        # Add median values on top of each box
        medians = data.groupby(x)[y].median()
        for i, median in enumerate(medians):
            ax.text(
                i,
                median,
                f'{median:.2f}',
                horizontalalignment='center',
                verticalalignment='bottom',
                fontweight='bold'
            )

        if output_path:
            self.save_plot(output_path)

    # Add these methods to your existing BasePlotter class

    def create_scatter_plot(self,
                            data: pd.DataFrame,
                            x: str,
                            y: str,
                            title: str,
                            xlabel: Optional[str] = None,
                            ylabel: Optional[str] = None,
                            hue: Optional[str] = None,
                            output_path: Optional[str] = None) -> None:
        """Create scatter plot with optional trend line."""
        fig, ax = self.setup_figure(title)

        sns.scatterplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            alpha=0.6,
            ax=ax
        )

        # Add trend line
        sns.regplot(
            data=data,
            x=x,
            y=y,
            scatter=False,
            color='red',
            line_kws={'linestyle': '--'},
            ax=ax
        )

        ax.set_xlabel(xlabel or x)
        ax.set_ylabel(ylabel or y)

        if output_path:
            self.save_plot(output_path)

    def create_line_plot(self,
                         data: Union[pd.Series, List],
                         title: str,
                         xlabel: Optional[str] = None,
                         ylabel: Optional[str] = None,
                         output_path: Optional[str] = None) -> None:
        """Create line plot for temporal data."""
        fig, ax = self.setup_figure(title)

        if isinstance(data, pd.Series):
            ax.plot(data.index, data.values,
                    color=self.settings['colors']['line'],
                    linewidth=2)
        else:
            ax.plot(range(len(data)), data,
                    color=self.settings['colors']['line'],
                    linewidth=2)

        ax.set_xlabel(xlabel or "Index")
        ax.set_ylabel(ylabel or "Value")

        if output_path:
            self.save_plot(output_path)

    def create_bar_plot(self,
                        data: Union[pd.Series, dict],
                        title: str,
                        xlabel: Optional[str] = None,
                        ylabel: Optional[str] = None,
                        horizontal: bool = False,
                        output_path: Optional[str] = None) -> None:
        """Create bar plot with consistent styling."""
        fig, ax = self.setup_figure(title)

        if isinstance(data, pd.Series):
            plot_data = data
        else:
            plot_data = pd.Series(data)

        if horizontal:
            plot_data.plot(
                kind='barh',
                ax=ax,
                color=self.settings['colors']['positive']
            )
        else:
            plot_data.plot(
                kind='bar',
                ax=ax,
                color=self.settings['colors']['positive']
            )

        ax.set_xlabel(xlabel or "Category")
        ax.set_ylabel(ylabel or "Value")

        if not horizontal:
            plt.xticks(rotation=45, ha='right')

        if output_path:
            self.save_plot(output_path)

    def create_joint_plot(self,
                          data: pd.DataFrame,
                          x: str,
                          y: str,
                          title: str,
                          kind: str = 'scatter',
                          output_path: Optional[str] = None) -> None:
        """Create joint plot (scatter with distributions)."""
        g = sns.jointplot(
            data=data,
            x=x,
            y=y,
            kind=kind,
            height=self.settings['figure_size'][0] * 0.8
        )

        g.fig.suptitle(title, y=1.02)

        if output_path:
            g.savefig(output_path, dpi=self.settings['dpi'], bbox_inches='tight')
            plt.close()

    def create_raincloud_distribution(self,
                                      data: pd.DataFrame,
                                      x: str,
                                      y: str,
                                      title: Union[str, dict[str, str]],
                                      xlabel: str,
                                      output_path: Optional[str] = None) -> None:
        """Create an enhanced distribution plot combining violin, boxplot, and points"""

        # Setup the figure
        fig, ax = self.setup_figure(title if isinstance(title, str) else title['title'])

        # Create violin plot
        sns.violinplot(data=data, x=x, y=y, ax=ax, inner=None, color='lightgrey', alpha=0.5)

        # Add individual points with jitter
        sns.stripplot(data=data, x=x, y=y, ax=ax, size=3, alpha=0.3, jitter=0.2, color='darkblue')

        # Add boxplot
        sns.boxplot(data=data, x=x, y=y, ax=ax, width=0.1, color='white',
                    showfliers=False, showbox=False, showcaps=False)

        # Customize the plot
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Frequency')

        # Add subtle grid
        ax.grid(True, axis='y', alpha=0.2)

        # Add zero line for reference
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)

        # Add sentiment regions
        ax.axhspan(0, 1, alpha=0.1, color='green', label='Positive')
        ax.axhspan(-1, 0, alpha=0.1, color='red', label='Negative')

        # Add legend
        ax.legend(loc='upper right')

        # Add subtitle with sentiment indicator if provided
        if isinstance(title, dict) and 'subtitle' in title:
            ax.text(0.5, 1.05, title['subtitle'],
                    horizontalalignment='center',
                    transform=ax.transAxes,
                    fontsize=10,
                    fontweight='bold')

        # Remove x-axis label since we're only showing one category
        ax.set_xticks([])

        if output_path:
            self.save_plot(output_path)