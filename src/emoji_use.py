from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Optional, List
from base_plotter import BasePlotter


@dataclass
class ModernChartStyle:
    """Configuration for modern chart appearance following visualization principles."""
    figure_size: Tuple[int, int] = (14, 8)
    primary_color: str = '#4361ee'
    secondary_color: str = '#e74c3c'
    background_color: str = '#f8f9fa'
    grid_color: str = '#dee2e6'
    text_color: str = '#2d3436'
    title_size: int = 20
    label_size: int = 14
    tick_size: int = 12
    annotation_size: int = 10
    dpi: int = 300


@dataclass
class ColumnConfigEmoji:
    """Configuration for DataFrame column names"""
    author: str = 'author'
    has_emoji: str = 'has_emoji'
    message: str = 'message'  # Added for emoji pattern analysis


@dataclass
class ChartConfig:
    """Configuration for chart appearance and settings"""
    title: str = 'Emoji Usage Patterns in Communication'
    xlabel: str = 'Participant'
    ylabel: str = 'Percentage of Messages with Emojis'
    annotation_settings: Dict = None

    def __post_init__(self):
        self.annotation_settings = self.annotation_settings or {
            'percentage': {
                'fontweight': 'bold',
                'va': 'bottom',
                'ha': 'center',
                'fontsize': 12
            },
            'count': {
                'fontsize': 10,
                'va': 'top',
                'ha': 'center'
            }
        }


@dataclass
class EmojiStats:
    """Container for comprehensive emoji statistics"""
    total_messages: int
    emoji_messages: int
    percentage: float
    by_author: pd.Series
    most_used_emojis: Dict[str, int]
    emoji_frequency: Dict[str, float]
    patterns: Dict[str, any]


class EmojiAnalyzer:
    """Analyze emoji usage patterns in communication"""

    def __init__(self,
                 style: Optional[ModernChartStyle] = None,
                 columns: Optional[ColumnConfigEmoji] = None,
                 chart_config: Optional[ChartConfig] = None):
        """Initialize analyzer with modern styling and configurations"""
        self.style = style or ModernChartStyle()
        self.cols = columns or ColumnConfigEmoji()
        self.chart_config = chart_config or ChartConfig()

        # Set the visual style
        plt.style.use('seaborn-v0_8-whitegrid')

    def analyze_emoji_patterns(self, df: pd.DataFrame) -> Dict[str, any]:
        """Analyze deeper patterns in emoji usage"""
        patterns = {
            'time_of_day': self._analyze_time_patterns(df),
            'message_length': self._analyze_message_length_correlation(df),
            # 'response_time': self._analyze_response_patterns(df)
        }
        return patterns

    def _analyze_time_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze emoji usage patterns by time of day"""
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        emoji_by_hour = df[df[self.cols.has_emoji]].groupby('hour').size()
        total_by_hour = df.groupby('hour').size()
        percentage_by_hour = (emoji_by_hour / total_by_hour * 100).round(1)

        return {
            'peak_hour': percentage_by_hour.idxmax(),
            'peak_percentage': percentage_by_hour.max(),
            'low_hour': percentage_by_hour.idxmin(),
            'low_percentage': percentage_by_hour.min()
        }

    def _analyze_message_length_correlation(self, df: pd.DataFrame) -> Dict:
        """Analyze correlation between message length and emoji usage"""
        df['message_length'] = df[self.cols.message].str.len()
        avg_length_with_emoji = df[df[self.cols.has_emoji]]['message_length'].mean()
        avg_length_without_emoji = df[~df[self.cols.has_emoji]]['message_length'].mean()

        return {
            'avg_length_with_emoji': round(avg_length_with_emoji, 1),
            'avg_length_without_emoji': round(avg_length_without_emoji, 1)
        }

    def create_visualization(self, df: pd.DataFrame, output_path: str) -> EmojiStats:
        """Create comprehensive emoji usage visualization with modern styling"""
        """Create comprehensive emoji usage visualization with modern styling"""
        # Debug prints
        print("ColumnConfig attributes:", dir(self.cols))
        print("has_emoji attribute value:", getattr(self.cols, 'has_emoji', None))
        # Calculate basic statistics
        total_by_author = df.groupby(self.cols.author).size()
        emoji_by_author = df[df[self.cols.has_emoji]].groupby(self.cols.author).size()
        percentage_by_author = (emoji_by_author / total_by_author * 100).round(1)

        # Create figure with modern styling
        fig, ax = plt.subplots(figsize=self.style.figure_size,
                               facecolor=self.style.background_color)
        ax.set_facecolor(self.style.background_color)

        # Create enhanced bar chart
        bars = ax.bar(percentage_by_author.index,
                      percentage_by_author.values,
                      color=self.style.primary_color,
                      alpha=0.7)

        # Add detailed annotations
        self._add_enhanced_annotations(ax, percentage_by_author, total_by_author, emoji_by_author)

        # Customize appearance
        self._customize_chart(ax)

        # Add informative title and labels
        plt.title(self.chart_config.title,
                  pad=20,
                  fontsize=self.style.title_size,
                  color=self.style.text_color)
        plt.xlabel(self.chart_config.xlabel,
                   fontsize=self.style.label_size,
                   color=self.style.text_color)
        plt.ylabel(self.chart_config.ylabel,
                   fontsize=self.style.label_size,
                   color=self.style.text_color)

        # Add summary statistics
        patterns = self.analyze_emoji_patterns(df)
        self._add_summary_statistics(df, emoji_by_author, patterns)

        # Save visualization
        plt.tight_layout(rect=[0, 0.1, 1, 1])
        plt.savefig(output_path,
                    dpi=self.style.dpi,
                    bbox_inches='tight',
                    facecolor=self.style.background_color)
        plt.close()

        return EmojiStats(
            total_messages=len(df),
            emoji_messages=emoji_by_author.sum(),
            percentage=(emoji_by_author.sum() / len(df) * 100).round(1),
            by_author=emoji_by_author,
            most_used_emojis=self._get_most_used_emojis(df),
            emoji_frequency=self._calculate_emoji_frequency(df),
            patterns=patterns
        )

    def _add_enhanced_annotations(self, ax, percentage_by_author, total_by_author, emoji_by_author):
        """Add detailed annotations to the visualization"""
        for i, v in enumerate(percentage_by_author):
            author = percentage_by_author.index[i]
            ax.text(i, v + 1, f'{v}%',
                    **self.chart_config.annotation_settings['percentage'])
            ax.text(i, -3,
                    f'{emoji_by_author[author]} of {total_by_author[author]}',
                    **self.chart_config.annotation_settings['count'])

    def _customize_chart(self, ax):
        """Apply modern chart styling"""
        ax.grid(True, axis='y', alpha=0.3, color=self.style.grid_color)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(self.style.grid_color)
        ax.spines['bottom'].set_color(self.style.grid_color)
        ax.tick_params(axis='both', colors=self.style.text_color)
        ax.margins(y=0.2)

    def _add_summary_statistics(self, df, emoji_by_author, patterns):
        """Add comprehensive summary statistics to the visualization"""
        summary_text = (
            f'Total Messages: {len(df):,}\n'
            f'Messages with Emojis: {emoji_by_author.sum():,} ({(emoji_by_author.sum() / len(df) * 100):.1f}%)\n'
            f'Peak Emoji Usage: {patterns["time_of_day"]["peak_hour"]:02d}:00 ({patterns["time_of_day"]["peak_percentage"]:.1f}%)\n'
            f'Avg Message Length with Emoji: {patterns["message_length"]["avg_length_with_emoji"]:.1f} chars'
        )
        plt.figtext(0.99, 0.05,
                    summary_text,
                    ha='right',
                    va='bottom',
                    fontsize=self.style.annotation_size,
                    color=self.style.text_color)