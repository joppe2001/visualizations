from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import pandas as pd
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from base_plotter import BasePlotter


@dataclass
class ColumnConfig:
    """Configuration for DataFrame column names"""
    timestamp: str = 'timestamp'
    message: str = 'message'
    author: str = 'author'
    sentiment: str = 'sentiment'
    has_sentiment: str = 'has_sentiment'  # New column to flag messages with valid sentiment


@dataclass
class TimeFeatures:
    """Time-based feature column names"""
    hour: str = 'hour'
    time_of_day: str = 'time_of_day'
    day_of_week: str = 'day_of_week'
    is_weekend: str = 'is_weekend'


@dataclass
class TimeConfig:
    """Configuration for time periods"""
    time_periods: Dict[str, Tuple[int, int]] = None
    sentiment_threshold: float = 0.05  # Minimum absolute sentiment score to consider

    def __post_init__(self):
        self.time_periods = self.time_periods or {
            'Late Night': (0, 5),
            'Early Morning': (5, 9),
            'Morning': (9, 12),
            'Afternoon': (12, 17),
            'Evening': (17, 21),
            'Night': (21, 24)
        }


class SentimentTimeAnalyzer:
    """Analyze sentiment patterns in relation to time"""

    def __init__(self,
                 cols: Optional[ColumnConfig] = None,
                 time_features: Optional[TimeFeatures] = None,
                 time_config: Optional[TimeConfig] = None,
                 plotter: Optional[BasePlotter] = None):
        """Initialize the analyzer with configurations"""
        self.cols = cols or ColumnConfig()
        self.time_features = time_features or TimeFeatures()
        self.time_config = time_config or TimeConfig()
        self.plotter = plotter or BasePlotter(preset='minimal')
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def _calculate_sentiment(self, text: str) -> Tuple[Optional[float], bool]:
        """
        Calculate sentiment score for a message and determine if it has meaningful sentiment.
        Returns (sentiment_score, has_sentiment).
        """
        try:
            compound_score = self.sentiment_analyzer.polarity_scores(str(text))['compound']
            # Only consider messages with sentiment above threshold
            if abs(compound_score) >= self.time_config.sentiment_threshold:
                return compound_score, True
            return None, False
        except:
            return None, False

    def _get_time_period(self, hour: int) -> str:
        """Map hour to time period"""
        for period, (start, end) in self.time_config.time_periods.items():
            if start <= hour < end:
                return period
        return 'Late Night'  # Default period

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with time features and sentiment"""
        df = df.copy()

        # Ensure timestamp is datetime
        df[self.cols.timestamp] = pd.to_datetime(df[self.cols.timestamp])

        # Extract time features
        df[self.time_features.hour] = df[self.cols.timestamp].dt.hour
        df[self.time_features.day_of_week] = df[self.cols.timestamp].dt.dayofweek
        df[self.time_features.is_weekend] = df[self.time_features.day_of_week].isin([5, 6])
        df[self.time_features.time_of_day] = df[self.time_features.hour].apply(self._get_time_period)

        # Set categorical order for time_of_day
        time_periods_order = [
            'Early Morning', 'Morning',
            'Afternoon', 'Evening', 'Night', 'Late Night'
        ]
        df[self.time_features.time_of_day] = pd.Categorical(
            df[self.time_features.time_of_day],
            categories=time_periods_order,
            ordered=True
        )

        # Calculate sentiment
        sentiment_results = df[self.cols.message].apply(self._calculate_sentiment)
        df[self.cols.sentiment] = sentiment_results.apply(lambda x: x[0])
        df[self.cols.has_sentiment] = sentiment_results.apply(lambda x: x[1])

        return df

    def analyze(self, df: pd.DataFrame, output_dir: Path) -> Dict:
        """Perform the complete analysis"""
        print("\nAnalyzing sentiment patterns across time...")
        output_dir.mkdir(exist_ok=True)

        # Prepare the data
        df = self.prepare_data(df)

        # Print basic statistics
        total_messages = len(df)
        sentiment_messages = df[self.cols.has_sentiment].sum()
        print(f"\nMessage Analysis:")
        print(f"Total messages: {total_messages:,}")
        print(f"Messages with sentiment: {sentiment_messages:,} ({sentiment_messages / total_messages * 100:.1f}%)")
        print(f"Messages without sentiment: {total_messages - sentiment_messages:,} ({(total_messages - sentiment_messages) / total_messages * 100:.1f}%)")

        # Create visualizations using only messages with sentiment
        self._create_hourly_trends(df, output_dir)
        self._create_period_distribution(df, output_dir)
        self._create_author_time_heatmap(df, output_dir)
        self._create_weekend_comparison(df, output_dir)

        # Calculate summary statistics
        results = self._calculate_statistics(df)
        self._print_insights(results)

        return results

    def _create_hourly_trends(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Create hourly sentiment trend visualization"""
        sentiment_df = df[df[self.cols.has_sentiment]]
        hourly_avg = sentiment_df.groupby(self.time_features.hour)[self.cols.sentiment].mean()

        self.plotter.create_line_plot(
            data=hourly_avg,
            title="Sentiment Throughout the Day (Messages with Sentiment)",
            xlabel="Hour of Day",
            ylabel="Average Sentiment Score",
            output_path=output_dir / "hourly_sentiment.png"
        )

    def _create_period_distribution(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Create time period sentiment distribution"""
        sentiment_df = df[df[self.cols.has_sentiment]]
        self.plotter.create_boxplot(
            data=sentiment_df,
            x=self.time_features.time_of_day,
            y=self.cols.sentiment,
            title="Sentiment Distribution by Time of Day (Messages with Sentiment)",
            ylabel="Sentiment Score",
            output_path=output_dir / "period_sentiment.png"
        )

    def _create_author_time_heatmap(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Create heatmap of sentiment by author and time"""
        sentiment_df = df[df[self.cols.has_sentiment]]
        pivot_data = sentiment_df.pivot_table(
            values=self.cols.sentiment,
            index=self.time_features.time_of_day,
            columns=self.cols.author,
            aggfunc='mean'
        )

        self.plotter.create_heatmap(
            data=pivot_data,
            title="Sentiment Patterns: Authors × Time of Day ( 1 = pos, 0 = neut )",
            output_path=output_dir / "author_time_heatmap.png"
        )

    def _create_weekend_comparison(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Create weekend vs weekday comparison"""
        sentiment_df = df[df[self.cols.has_sentiment]]
        weekend_labels = {False: 'Weekday', True: 'Weekend'}
        sentiment_df['day_type'] = sentiment_df[self.time_features.is_weekend].map(weekend_labels)

        self.plotter.create_boxplot(
            data=sentiment_df,
            x='day_type',
            y=self.cols.sentiment,
            title="Weekend vs Weekday Sentiment (Messages with Sentiment)",
            ylabel="Sentiment Score",
            output_path=output_dir / "weekend_sentiment.png"
        )

    def _calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate summary statistics"""
        sentiment_df = df[df[self.cols.has_sentiment]]
        return {
            'hourly': sentiment_df.groupby(self.time_features.hour)[self.cols.sentiment].agg(['mean', 'std']),
            'period': sentiment_df.groupby(self.time_features.time_of_day)[self.cols.sentiment].agg(['mean', 'std']),
            'weekend': sentiment_df.groupby(self.time_features.is_weekend)[self.cols.sentiment].agg(['mean', 'std']),
            'author_time': sentiment_df.pivot_table(
                values=self.cols.sentiment,
                index=self.time_features.time_of_day,
                columns=self.cols.author,
                aggfunc=['mean', 'count']
            )
        }

    def _print_insights(self, results: Dict) -> None:
        """Print key insights from the analysis"""
        print("\nKey Sentiment Patterns:")
        print("-" * 50)

        # Most positive hour
        hourly_means = results['hourly']['mean']
        best_hour = hourly_means.idxmax()
        print(f"Peak positivity: {best_hour:02d}:00 (Score: {hourly_means[best_hour]:.3f})")

        # Most consistent period
        period_stats = results['period']
        most_consistent = period_stats['std'].idxmin()
        print(f"\nMost consistent mood: {most_consistent}")
        print(f"  Average: {period_stats.loc[most_consistent, 'mean']:.3f}")
        print(f"  Std Dev: {period_stats.loc[most_consistent, 'std']:.3f}")

        # Weekend effect
        weekend_stats = results['weekend']
        weekend_diff = weekend_stats.loc[True, 'mean'] - weekend_stats.loc[False, 'mean']
        print(f"\nWeekend effect: {weekend_diff:+.3f}")
        print(f"  Weekend avg: {weekend_stats.loc[True, 'mean']:.3f}")
        print(f"  Weekday avg: {weekend_stats.loc[False, 'mean']:.3f}")