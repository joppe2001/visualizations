import pandas as pd
import click
from config_handler import ConfigHandler
from visualization import create_simple_message_frequency_plot
from emoji_use import EmojiAnalyzer, ModernChartStyle, ChartConfig, ColumnConfigEmoji, EmojiStats
from timestamp import visualize_hourly_activity
from distribution import SentimentAnalyzer, BasePlotter, ColumnConfig
from dimensionality import DimensionalityAnalyzer, ClusteringConfig, VectorizerConfig, VisualizationConfig, KeywordExtractionConfig


def load_data(file_path):
    df = pd.read_csv(file_path) if file_path.suffix == '.csv' else pd.read_parquet(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def create_message_frequency(df, image_dir):
    """Generate message frequency visualization."""
    fig = create_simple_message_frequency_plot(df)
    output_path = image_dir / 'message_frequency.jpg'
    fig.write_image(str(output_path), scale=2)
    click.echo(f"Message frequency plot saved as {output_path}")


def create_emoji_usage(df: pd.DataFrame, image_dir) -> EmojiStats:
    """Generate comprehensive emoji usage visualization with modern styling."""
    print("Available columns:", df.columns.tolist())
    # Create modern styling configuration
    style = ModernChartStyle(
        figure_size=(14, 8),
        primary_color='#4361ee',
        secondary_color='#e74c3c',
        background_color='#f8f9fa',
        grid_color='#dee2e6',
        text_color='#2d3436',
        title_size=20,
        label_size=14,
        tick_size=12,
        annotation_size=10,
        dpi=300
    )

    # Custom chart configuration
    chart_config = ChartConfig(
        title='Emoji Usage Patterns in Communication',
        xlabel='Chat Participant',
        ylabel='Percentage of Messages with Emojis'
    )

    # Initialize analyzer with configurations
    columns = ColumnConfigEmoji()
    analyzer = EmojiAnalyzer(
        style=style,
        columns=columns,
        chart_config=chart_config
    )

    # Set output path and run analysis
    output_path = image_dir / 'emoji_usage.png'
    stats = analyzer.create_visualization(df, str(output_path))

    # Print insights
    click.echo(f"\nEmoji Usage Analysis Results:")
    click.echo("-" * 30)
    click.echo(f"Visualization saved as: {output_path}")
    click.echo(f"Total messages analyzed: {stats.total_messages:,}")
    click.echo(f"Messages containing emojis: {stats.emoji_messages:,} ({stats.percentage:.1f}%)")

    return stats

def create_hourly_activity(df, image_dir):
    """Generate hourly activity visualization."""
    output_path = image_dir / 'hourly_activity.jpg'
    visualize_hourly_activity(df, str(output_path))
    click.echo(f"Hourly activity chart saved as {output_path}")


def create_sentiment_analysis(df, image_dir):
    """Generate sentiment analysis visualizations."""
    plotter = BasePlotter(
        figure_size=(12, 8),
        style='seaborn-v0_8-darkgrid'
    )

    analyzer = SentimentAnalyzer(
        columns=ColumnConfig(
            timestamp='timestamp',
            message='message',
            author='author'
        ),
        plotter=plotter
    )

    sentiment_dir = image_dir / 'sentiment'
    sentiment_dir.mkdir(exist_ok=True)

    results = analyzer.analyze(df, str(sentiment_dir))
    click.echo(f"Sentiment analysis visualizations saved in {sentiment_dir}")
    return results


def create_text_clusters(df: pd.DataFrame, image_dir) -> None:
    """Generate text clustering visualizations with main conversation topics."""
    try:
        print("\nGenerating topic analysis visualizations...")

        # Create directory for cluster visualizations
        cluster_dir = image_dir / 'text_clusters'
        cluster_dir.mkdir(exist_ok=True, parents=True)

        # Initialize analyzer with chat-optimized settings
        analyzer = DimensionalityAnalyzer(
            clustering_config=ClusteringConfig(
                n_clusters=6,  # Create 6 main topics
                random_state=42
            ),
            vectorizer_config=VectorizerConfig(
                min_df=10,  # Words must appear in at least 10 messages
                max_features=3000,
                stop_words='english',
                ngram_range=(1, 2)  # Use both single words and pairs
            ),
            viz_config=VisualizationConfig(
                figure_size=(15, 10),
                dpi=300,
                cmap='tab10',
                alpha=0.6
            ),
            keyword_config=KeywordExtractionConfig(
                n_keywords=8,  # Show top 8 keywords per topic
                min_df=5
            )
        )

        # Run analysis
        analyzer.analyze_and_visualize(df, cluster_dir)

        print(f"\nVisualizations saved in: {cluster_dir}")
        print("Generated files:")
        print(f"- PCA topic visualization: {cluster_dir / 'pca_topics.png'}")
        print(f"- t-SNE topic visualization: {cluster_dir / 'tsne_topics.png'}")

    except Exception as e:
        print(f"\nError generating topic analysis: {str(e)}")
        import traceback
        traceback.print_exc()


@click.group()
def cli():
    """WhatsApp Chat Analysis Tool - Choose a visualization to generate."""
    pass


@cli.command()
@click.option('--all', is_flag=True, help='Generate all visualizations')
def visualize(all):
    """Generate visualizations based on WhatsApp chat data."""
    config = ConfigHandler()
    config.ensure_directories()

    data_path = config.get_processed_file_path()
    df = load_data(data_path)
    image_dir = config.get_image_dir()

    visualizations = {
        1: ("Message Frequency Plot", create_message_frequency),
        2: ("Emoji Usage Chart", create_emoji_usage),
        3: ("Hourly Activity Visualization", create_hourly_activity),
        4: ("Sentiment Analysis", create_sentiment_analysis),
        5: ("Text Clustering", create_text_clusters),  # New option
    }

    if all:
        click.echo("Generating all visualizations...")
        for _, func in visualizations.values():
            func(df, image_dir)
        click.echo("All visualizations completed!")
        return

    click.echo("Available visualizations:")
    for num, (name, _) in visualizations.items():
        click.echo(f"{num}. {name}")

    choice = click.prompt(
        "Please select a visualization (1-5)",  # Updated range
        type=click.IntRange(1, len(visualizations))
    )

    if choice in visualizations:
        click.echo(f"\nGenerating {visualizations[choice][0]}...")
        visualizations[choice][1](df, image_dir)
        click.echo("Visualization completed!")
    else:
        click.echo("Invalid choice. Please select a number between 1 and 5.")


@cli.command()
def info():
    """Display information about the current configuration."""
    config = ConfigHandler()
    click.echo("\nConfiguration Information:")
    click.echo(f"Data file: {config.get_processed_file_path()}")
    click.echo(f"Images directory: {config.get_image_dir()}")


@cli.command()
@click.option('--detailed', is_flag=True, help='Show detailed sentiment statistics')
def sentiment(detailed):
    """Analyze sentiment in chat messages."""
    config = ConfigHandler()
    config.ensure_directories()

    data_path = config.get_processed_file_path()
    df = load_data(data_path)
    image_dir = config.get_image_dir()

    click.echo("Running sentiment analysis...")
    results = create_sentiment_analysis(df, image_dir)

    if detailed:
        click.echo("\nDetailed Sentiment Statistics:")
        stats = results.groupby('author')['sentiment'].agg(['mean', 'std', 'count'])
        click.echo(stats.round(3).to_string())


if __name__ == "__main__":
    cli()