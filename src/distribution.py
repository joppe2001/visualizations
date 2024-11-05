import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import os
from textblob import TextBlob
from base_plotter import BasePlotter
import re


@dataclass
class ColumnConfig:
    """Configuration for DataFrame column names."""
    timestamp: str = 'timestamp'
    message: str = 'message'
    author: str = 'author'
    month: str = 'month'
    sentiment: str = 'sentiment'
    has_sentiment: str = 'has_sentiment'  # New column to flag messages with valid sentiment


@dataclass
class WhatsAppConfig:
    """Configuration for WhatsApp-specific message handling."""
    # Messages to filter out (case insensitive)
    system_messages: List[str] = None
    # Regex patterns to identify non-content
    system_patterns: List[str] = None

    def __post_init__(self):
        self.system_messages = self.system_messages or [
            'media omitted',
            'message deleted',
            'this message was deleted',
            'missed voice call',
            'missed video call',
            'missed call',
            'location shared',
            'joined using this group\'s invite link',
            'left the group',
            'added',
            'removed',
            'changed the subject',
            'changed this group\'s icon',
            'changed the group description',
            'changed their phone number',
            'messages and calls are end-to-end encrypted',
            'gif omitted',
            'image omitted',
            'video omitted',
            'sticker omitted',
            'document omitted',
            'audio omitted',
            'contact card omitted'
        ]

        self.system_patterns = self.system_patterns or [
            r'^\<?\s*attached\s*:\s*\d*\s*file[s]?\s*\>?$',
            r'^\d+\s+attachments?$',
            r'^created group',
            r'security code changed',
            r'messages to this group are now',
            r'^[<>]?[\w\s]+ omitted[>]?$',
            r'[\u2068\u2069]',  # Invisible formatting characters
            r'^\s*$',  # Empty messages
            r'^https?://',  # URLs only
            r'^\+\d{10,}$',  # Phone numbers only
            r'^[\u2714\u2611\u2705\u274c]$'  # Single emoji checkmarks
        ]


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    titles: Dict[str, str] = None
    labels: Dict[str, str] = None
    output_files: Dict[str, str] = None

    def __post_init__(self):
        self.titles = self.titles or {
            'monthly': 'Monthly Average Sentiment (Messages with Sentiment)',
            'distribution': 'Is the conversation positive?',
            'boxplot': 'Sentiment Distribution by Author',
            'heatmap': 'Sentiment Heatmap by Author and Month'
        }

        self.labels = self.labels or {
            'sentiment': 'Sentiment Score (1.0 = Most Positive, -1.0 = Most Negative)',
            'sentiment_dist': 'Sentiment Score (1.0 = Most Positive, -1.0 = Most Negative)'
        }

        self.output_files = self.output_files or {
            'monthly': 'monthly_sentiment.png',
            'distribution': 'sentiment_distribution.png',
            'boxplot': 'sentiment_by_author.png',
            'heatmap': 'sentiment_heatmap.png'
        }


@dataclass
class AnalysisConfig:
    """Configuration for analysis settings."""
    round_digits: int = 3
    summary_metrics: List[str] = None
    summary_columns: List[str] = None
    min_words: int = 2  # Minimum words for a message to be considered content

    def __post_init__(self):
        self.summary_metrics = self.summary_metrics or ['mean', 'count']
        self.summary_columns = self.summary_columns or [
            'avg_sentiment', 'message_count', 'first_message', 'last_message'
        ]


class SentimentAnalyzer:
    """
    Analyze sentiment in WhatsApp messages with smart filtering.

    Features:
    - Filters out system messages and non-content
    - Handles WhatsApp-specific message types
    - Uses TextBlob for sentiment analysis
    - Only includes messages with valid sentiment scores
    """

    def __init__(self,
                 columns: Optional[ColumnConfig] = None,
                 whatsapp_config: Optional[WhatsAppConfig] = None,
                 viz_config: Optional[VisualizationConfig] = None,
                 analysis_config: Optional[AnalysisConfig] = None,
                 plotter: Optional[BasePlotter] = None):
        """Initialize analyzer with custom configurations."""
        self.cols = columns or ColumnConfig()
        self.whatsapp_config = whatsapp_config or WhatsAppConfig()
        self.viz_config = viz_config or VisualizationConfig()
        self.analysis_config = analysis_config or AnalysisConfig()
        self.plotter = plotter or BasePlotter(preset='dark')

        # Compile regex patterns for efficiency
        self.system_patterns = [re.compile(pattern, re.IGNORECASE)
                              for pattern in self.whatsapp_config.system_patterns]

    def is_content_message(self, message: str) -> bool:
        """
        Determine if a message contains actual content worth analyzing.
        Returns False for system messages, notifications, and non-content.
        """
        if not isinstance(message, str) or not message.strip():
            return False

        # Check against system messages list
        if any(sys_msg.lower() in message.lower()
               for sys_msg in self.whatsapp_config.system_messages):
            return False

        # Check against regex patterns
        if any(pattern.search(message) for pattern in self.system_patterns):
            return False

        # Check minimum word count (excluding very short messages)
        words = message.split()
        if len(words) < self.analysis_config.min_words:
            return False

        return True

    def analyze_sentiment(self, text: str) -> Tuple[Optional[float], bool]:
        """
        Calculate sentiment score for a single message.
        Returns (sentiment_score, has_sentiment).
        """
        if not self.is_content_message(text):
            return None, False

        try:
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity
            # Only consider messages with non-zero sentiment
            if sentiment != 0:
                return sentiment, True
            return None, False
        except:
            return None, False

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for analysis."""
        df = df.copy()
        df[self.cols.timestamp] = pd.to_datetime(df[self.cols.timestamp])

        # Analyze sentiment and flag messages with sentiment
        sentiment_results = df[self.cols.message].apply(self.analyze_sentiment)
        df[self.cols.sentiment] = sentiment_results.apply(lambda x: x[0])
        df[self.cols.has_sentiment] = sentiment_results.apply(lambda x: x[1])

        df[self.cols.month] = df[self.cols.timestamp].dt.to_period('M')
        return df

    def create_user_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create user sentiment summary (messages with sentiment only)."""
        # Filter for messages with sentiment
        sentiment_df = df[df[self.cols.has_sentiment]]

        metrics = {
            self.cols.sentiment: self.analysis_config.summary_metrics,
            self.cols.timestamp: ['min', 'max']
        }

        summary = sentiment_df.groupby(self.cols.author).agg(metrics).round(
            self.analysis_config.round_digits
        )
        summary.columns = self.analysis_config.summary_columns

        # Add sentiment message percentage
        total_messages = df.groupby(self.cols.author).size()
        sentiment_messages = sentiment_df.groupby(self.cols.author).size()
        sentiment_percentage = (sentiment_messages / total_messages * 100).round(1)
        summary['messages_with_sentiment_pct'] = sentiment_percentage

        return summary

    def _get_sentiment_summary(self, sentiment_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate overall sentiment summary statistics."""
        sentiment_values = sentiment_df[self.cols.sentiment]
        return {
            'mean': sentiment_values.mean(),
            'positive_pct': (sentiment_values > 0).mean() * 100,
            'negative_pct': (sentiment_values < 0).mean() * 100
        }

    def create_visualizations(self, df: pd.DataFrame, output_dir: str) -> None:
        """Create visualizations using only messages with sentiment."""
        os.makedirs(output_dir, exist_ok=True)

        # Filter for messages with sentiment
        sentiment_df = df[df[self.cols.has_sentiment]].copy()

        # Calculate sentiment summary for the subtitle
        stats = self._get_sentiment_summary(sentiment_df)
        sentiment_indicator = "MOSTLY POSITIVE" if stats['mean'] > 0 else "MOSTLY NEGATIVE"
        score_info = f"({stats['positive_pct']:.1f}% positive, {stats['negative_pct']:.1f}% negative)"

        # Update distribution title with sentiment indicator
        distribution_title = {
            'title': self.viz_config.titles['distribution'],
            'subtitle': f"{sentiment_indicator} {score_info}"
        }

        # 1. Monthly sentiment trend
        monthly_sentiment = sentiment_df.groupby(self.cols.month)[self.cols.sentiment].mean()
        self.plotter.create_time_series(
            monthly_sentiment,
            self.viz_config.titles['monthly'],
            self.viz_config.labels['sentiment'],
            output_path=os.path.join(output_dir, self.viz_config.output_files['monthly'])
        )

        # 2. Sentiment distribution with indicator
        self.plotter.create_distribution(
            sentiment_df[self.cols.sentiment],
            distribution_title,
            self.viz_config.labels['sentiment_dist'],
            output_path=os.path.join(output_dir, self.viz_config.output_files['distribution'])
        )

        # 3. Sentiment heatmap
        pivot_data = sentiment_df.pivot_table(
            values=self.cols.sentiment,
            index=self.cols.month,
            columns=self.cols.author,
            aggfunc='mean'
        )
        self.plotter.create_heatmap(
            pivot_data,
            self.viz_config.titles['heatmap'],
            output_path=os.path.join(output_dir, self.viz_config.output_files['heatmap'])
        )


    def analyze(self, df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
        """Main analysis pipeline."""
        df = self.prepare_data(df)

        # Print basic statistics
        total_messages = len(df)
        sentiment_messages = df[self.cols.has_sentiment].sum()
        print(f"\nMessage Analysis:")
        print(f"Total messages: {total_messages:,}")
        print(f"Messages with sentiment: {sentiment_messages:,} ({sentiment_messages / total_messages * 100:.1f}%)")
        print(
            f"Messages without sentiment: {total_messages - sentiment_messages:,} ({(total_messages - sentiment_messages) / total_messages * 100:.1f}%)")

        self.create_visualizations(df, output_dir)
        user_summary = self.create_user_summary(df)
        print("\nUser Summary (Messages with Sentiment Only):")
        print(user_summary.sort_values('avg_sentiment', ascending=False))

        return df