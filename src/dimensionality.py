from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import Memory
from tqdm import tqdm
import nltk
from base_plotter import BasePlotter


@dataclass
class ClusterTheme:
    theme_name: str
    confidence: str  # 'high', 'medium', 'low', or 'unclear'
    description: str
    representative_messages: List[str] = field(default_factory=list)


@dataclass
class ClusteringConfig:
    n_clusters: int = 8
    random_state: int = 42
    sample_size: int = 50  # Number of messages to sample for theme analysis


@dataclass
class VectorizerConfig:
    min_df: int = 10
    max_features: int = 3000
    stop_words: str = 'english'
    ngram_range: Tuple[int, int] = (1, 2)


@dataclass
class VisualizationConfig:
    figure_size: Tuple[int, int] = (18, 12)  # Increased from (15, 10)
    dpi: int = 1600  # Increased from 300 for higher resolution
    cmap: str = 'tab10'
    alpha: float = 0.7
    style: str = 'seaborn-v0_8-darkgrid'

    # Added font configurations
    title_fontsize: int = 14
    label_fontsize: int = 12
    legend_fontsize: int = 14
    legend_title_fontsize: int = 16


@dataclass
class KeywordExtractionConfig:
    n_keywords: int = 8
    min_df: int = 5
    stop_words: str = 'english'


@dataclass
class AnalysisResults:
    embedded_data: np.ndarray
    clusters: np.ndarray
    cluster_keywords: Dict[int, List[str]]
    cluster_themes: Dict[int, ClusterTheme]
    cluster_sizes: Dict[int, int]


class DimensionalityAnalyzer:
    def __init__(
            self,
            clustering_config: ClusteringConfig = None,
            vectorizer_config: VectorizerConfig = None,
            viz_config: VisualizationConfig = None,
            keyword_config: KeywordExtractionConfig = None
    ):
        self.clustering_config = clustering_config or ClusteringConfig()
        self.vectorizer_config = vectorizer_config or VectorizerConfig()
        self.viz_config = viz_config or VisualizationConfig()
        self.keyword_config = keyword_config or KeywordExtractionConfig()

        self.memory = Memory(Path.home() / '.cache' / 'whatsapp_analysis', verbose=0)
        self.vectorizer = None

        # Initialize NLTK data
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("Downloading required NLTK data...")
            nltk.download('stopwords', quiet=True)

    def _preprocess_texts(self, texts: List[str], desc="Preprocessing texts") -> List[str]:
        """Preprocess texts with progress bar."""
        processed_texts = []
        for text in tqdm(texts, desc=desc):
            text = text.lower()
            if len(text.split()) > 3 and not text.startswith(('<image', '<video', '<media')):
                processed_texts.append(text)
        return processed_texts

    def _extract_cluster_keywords(self, texts: List[str], clusters: np.ndarray) -> Dict[int, List[str]]:
        """Extract keywords that characterize each cluster."""
        print("Extracting keywords for each topic...")
        cluster_keywords = {}
        unique_clusters = np.unique(clusters)

        count_vectorizer = CountVectorizer(
            stop_words=self.keyword_config.stop_words,
            min_df=self.keyword_config.min_df
        )

        for cluster_id in tqdm(unique_clusters, desc="Analyzing topics"):
            cluster_texts = [text for text, cluster in zip(texts, clusters) if cluster == cluster_id]
            if not cluster_texts:
                continue

            word_counts = count_vectorizer.fit_transform(cluster_texts)
            words = count_vectorizer.get_feature_names_out()

            total_counts = word_counts.sum(axis=0).A1
            top_indices = total_counts.argsort()[-self.keyword_config.n_keywords:][::-1]
            top_keywords = [words[i] for i in top_indices]

            cluster_keywords[cluster_id] = top_keywords

        return cluster_keywords

    def _identify_cluster_theme(self,
                              cluster_texts: List[str],
                              keywords: List[str],
                              sample_size: int = 50) -> ClusterTheme:
        """Analyze cluster keywords and messages to identify themes with improved keyword prioritization."""
        # Sample messages for analysis
        sample = np.random.choice(cluster_texts,
                                size=min(sample_size, len(cluster_texts)),
                                replace=False)

        top_3_keywords = set(keywords[:3])
        keywords_str = ', '.join(top_3_keywords)

        # Theme identification with categories based on actual conversation patterns
        affectionate_words = {
            'love', 'cutie', 'lovvie', 'baby', 'bby', 'handsome', 'cute',
            'miss', 'beautiful', 'perfect', 'amazing'
        }

        daily_life_words = {
            'home', 'work', 'office', 'train', 'metro', 'walking', 'safely',
            'tickets', 'jacket', 'bed', 'sleep'
        }

        planning_words = {
            'tonight', 'dinner', 'ramen', 'birthday', 'bbq', 'free', 'next week',
            'monday', 'visit'
        }

        missing_together_words = {
            'miss', 'come back', 'wish', 'together', 'cant wait', 'come to me',
            'back home', 'want you'
        }

        # Check message themes
        if any(word in affectionate_words for word in top_3_keywords):
            return ClusterTheme(
                theme_name="Love & Affection",
                confidence="high",
                description=f"Expressions of love and affection. Key terms: {keywords_str}",
                representative_messages=sample[:3]
            )

        if any(word in daily_life_words for word in top_3_keywords):
            return ClusterTheme(
                theme_name="Daily Life Together",
                confidence="high",
                description=f"Sharing daily activities and updates. Key terms: {keywords_str}",
                representative_messages=sample[:3]
            )

        if any(word in planning_words for word in top_3_keywords):
            return ClusterTheme(
                theme_name="Making Plans",
                confidence="high",
                description=f"Planning activities and events. Key terms: {keywords_str}",
                representative_messages=sample[:3]
            )

        if any(word in missing_together_words for word in top_3_keywords):
            return ClusterTheme(
                theme_name="Missing Each Other",
                confidence="high",
                description=f"Expressions of missing each other. Key terms: {keywords_str}",
                representative_messages=sample[:3]
            )

        # For messages in other languages (Portuguese/German)
        if any(word in {'eu', 'você', 'te', 'wir', 'du', 'ist'} for word in top_3_keywords):
            return ClusterTheme(
                theme_name="Language Practice",
                confidence="medium",
                description=f"Conversations in other languages. Key terms: {keywords_str}",
                representative_messages=sample[:3]
            )

        # Analyze message patterns for remaining cases
        msg_lengths = [len(msg.split()) for msg in sample]
        avg_length = sum(msg_lengths) / len(msg_lengths)

        if avg_length > 15:
            return ClusterTheme(
                theme_name="💬 Extended Messages",
                confidence="medium",
                description=f"Longer conversations about: {keywords_str}",
                representative_messages=sample[:3]
            )
        else:
            # Filter out very common words for theme naming
            common_words = {'just', 'like', 'want', 'gonna', 'im', 'dont', 'know',
                            'get', 'got', 'yeah', 'yes', 'okay', 'sure', 'well'}
            theme_words = [word for word in top_3_keywords if len(word) > 3
                           and word not in common_words]

            if theme_words:
                theme_name = f"Chat about {' & '.join(theme_words)}"
            else:
                theme_name =  "Quick Updates"

            return ClusterTheme(
                theme_name=theme_name,
                confidence="medium",
                description=f"Short message exchanges about: {keywords_str}",
                representative_messages=sample[:3]
            )

    def _get_text_vectors(self, texts: List[str]) -> np.ndarray:
        """Convert texts to TF-IDF vectors."""
        self.vectorizer = TfidfVectorizer(
            min_df=self.vectorizer_config.min_df,
            max_features=self.vectorizer_config.max_features,
            stop_words=self.vectorizer_config.stop_words,
            ngram_range=self.vectorizer_config.ngram_range
        )
        return self.vectorizer.fit_transform(texts).toarray()

    def analyze_and_visualize(self, df: pd.DataFrame, output_dir: Path) -> AnalysisResults:
        """Main analysis and visualization function with theme identification."""
        print("\n=== Starting Theme Analysis ===")
        output_dir.mkdir(exist_ok=True)

        texts = self._preprocess_texts(df['message'].tolist())
        print(f"Analyzing {len(texts)} messages...")

        vectors = self._get_text_vectors(texts)

        kmeans = KMeans(
            n_clusters=self.clustering_config.n_clusters,
            random_state=self.clustering_config.random_state,
            n_init=10
        )
        clusters = kmeans.fit_predict(vectors)

        cluster_keywords = self._extract_cluster_keywords(texts, clusters)

        # Identify themes for each cluster
        cluster_themes = {}
        print("\nIdentifying conversation themes...")
        for cluster_id in tqdm(range(self.clustering_config.n_clusters)):
            cluster_texts = [text for text, c in zip(texts, clusters) if c == cluster_id]
            keywords = cluster_keywords[cluster_id]
            theme = self._identify_cluster_theme(
                cluster_texts,
                keywords,
                self.clustering_config.sample_size
            )
            cluster_themes[cluster_id] = theme

        cluster_sizes = {
            cluster_id: np.sum(clusters == cluster_id)
            for cluster_id in range(self.clustering_config.n_clusters)
        }

        results = AnalysisResults(
            embedded_data=vectors,
            clusters=clusters,
            cluster_keywords=cluster_keywords,
            cluster_themes=cluster_themes,
            cluster_sizes=cluster_sizes
        )

        print("\nGenerating visualizations...")
        self._create_visualizations(results, texts, output_dir)

        print("\n=== Analysis Complete ===")
        self._print_theme_summary(results)

        return results

    def _create_visualizations(self, results: AnalysisResults, texts: List[str], output_dir: Path) -> None:
        """Create all visualizations using BasePlotter."""
        plotter = BasePlotter(
            preset='default',
            figure_size=(self.viz_config.figure_size[0] + 3, self.viz_config.figure_size[1]),
            dpi=self.viz_config.dpi,
            style=self.viz_config.style
        )

        for method in ["t-SNE", "PCA"]:
            self._create_single_visualization(results, method, plotter, output_dir)

    def _create_single_visualization(self, results: AnalysisResults, method: str,
                                     plotter: BasePlotter, output_dir: Path) -> None:
        """Create a single visualization with cluster overlaps highlighted."""
        print(f"Generating {method} visualization...")

        # Setup figure with high-quality settings
        plt.rcParams.update({
            'figure.dpi': self.viz_config.dpi,
            'savefig.dpi': self.viz_config.dpi,
            'font.size': 14,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'lines.markersize': 10,
            'lines.linewidth': 2.5,
            'font.family': 'sans-serif',
            'font.weight': 'medium',
            'axes.labelweight': 'bold',
            'figure.figsize': self.viz_config.figure_size
        })

        if method == "t-SNE":
            embedded = TSNE(n_components=2, random_state=42).fit_transform(results.embedded_data)
            title = 'Conversation Themes (t-SNE)'
        else:
            pca = PCA(n_components=2, random_state=42)
            embedded = pca.fit_transform(results.embedded_data)
            title = f'Conversation Themes (PCA)\nTotal Variance Explained: {sum(pca.explained_variance_ratio_):.1%}'

        fig, ax = plotter.setup_figure(title)

        # Set higher quality figure properties
        plt.rcParams['figure.dpi'] = self.viz_config.dpi
        plt.rcParams['savefig.dpi'] = self.viz_config.dpi
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['lines.markersize'] = 8
        plt.rcParams['lines.linewidth'] = 2

        # Create a consistent color mapping
        cmap = plt.get_cmap(self.viz_config.cmap)
        colors = [cmap(i / self.clustering_config.n_clusters)
                  for i in range(self.clustering_config.n_clusters)]

        # Create a color map dictionary that maps cluster IDs to their colors
        sorted_topics = sorted(results.cluster_sizes.items(), key=lambda x: x[1], reverse=True)
        color_map = {cluster_id: colors[cluster_id] for cluster_id, _ in sorted_topics}

        # Calculate cluster centroids and covariances
        cluster_info = {}
        for cluster_id in range(self.clustering_config.n_clusters):
            mask = results.clusters == cluster_id
            if np.sum(mask) > 0:
                points = embedded[mask]
                centroid = np.mean(points, axis=0)

                if len(points) > 1:
                    covariance = np.cov(points.T)
                    eigenvals, eigenvects = np.linalg.eigh(covariance)
                    cluster_info[cluster_id] = {
                        'centroid': centroid,
                        'eigenvals': eigenvals,
                        'eigenvects': eigenvects,
                        'points': points
                    }

        # Create color array for scatter plot that matches the cluster colors
        point_colors = [color_map[cluster_id] for cluster_id in results.clusters]

        # Draw the scatter plot using the mapped colors
        scatter = ax.scatter(
            embedded[:, 0], embedded[:, 1],
            c=point_colors,
            alpha=self.viz_config.alpha,
            s=50
        )

        def confidence_ellipse(points, ax, n_std=2.0, **kwargs):
            """Create a covariance confidence ellipse of the points."""
            if len(points) < 2:
                return None

            cov = np.cov(points, rowvar=False)
            pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

            ell_radius_x = np.sqrt(1 + pearson)
            ell_radius_y = np.sqrt(1 - pearson)

            ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                              **kwargs)

            scale_x = np.sqrt(cov[0, 0]) * n_std
            scale_y = np.sqrt(cov[1, 1]) * n_std

            transf = transforms.Affine2D() \
                .rotate_deg(45) \
                .scale(scale_x, scale_y) \
                .translate(np.mean(points[:, 0]), np.mean(points[:, 1]))

            ellipse.set_transform(transf + ax.transData)
            return ellipse

        # Draw ellipses using the same color mapping
        for cluster_id, info in cluster_info.items():
            color = color_map[cluster_id]
            ellipse = confidence_ellipse(
                info['points'],
                ax,
                n_std=2.0,
                facecolor=color,
                alpha=0.1,
                edgecolor=color,
                linewidth=1,
                linestyle='--'
            )
            if ellipse is not None:
                ax.add_patch(ellipse)

        # Create legend with consistent colors
        legend_elements = []
        legend_labels = []
        total_messages = sum(results.cluster_sizes.values())

        for cluster_id, size in sorted_topics:
            theme = results.cluster_themes[cluster_id]
            patch = plt.Circle((0, 0), radius=1, color=color_map[cluster_id],
                               alpha=self.viz_config.alpha)
            legend_elements.append(patch)

            percentage = (size / total_messages) * 100
            confidence_marker = "high" if theme.confidence == "high" else "medium"

            # More concise and clearer label format
            label = (f"{theme.theme_name}\n"
                     f"({percentage:.1f}% of messages)\n"
                     f"Common words: {', '.join(results.cluster_keywords[cluster_id][:3])}")
            legend_labels.append(label)

        # Add legend with better formatting
        ax.legend(
            legend_elements,
            legend_labels,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            fontsize=12,
            title="Conversation Topics",
            title_fontsize=14,
            borderaxespad=0,
            frameon=True,
            fancybox=True,
            shadow=True,
            markerscale=2.0
        )

        # Adjusted layout with more space for legend
        plt.tight_layout(rect=(0, 0.1, 0.82, 1))

        # Higher quality notes text
        plt.figtext(0.98, 0.02, "Dashed lines show where topics overlap",
                    ha='right', fontsize=8, style='italic', weight='medium')

        # Save with maximum quality
        output_path = output_dir / f'{method.lower()}_themes.png'
        plt.savefig(
            str(output_path),
            dpi=self.viz_config.dpi,
            bbox_inches='tight',
            pad_inches=0.2,
            format='png',
            transparent=False,
            facecolor='white'
        )
        plt.close()

    def _print_theme_summary(self, results: AnalysisResults) -> None:
        """Print a detailed summary of the identified themes."""
        print("\nConversation Theme Summary:")
        print("=" * 60)

        sorted_topics = sorted(
            results.cluster_sizes.items(),
            key=lambda x: x[1],
            reverse=True
        )

        total_messages = sum(results.cluster_sizes.values())

        for cluster_id, size in sorted_topics:
            theme = results.cluster_themes[cluster_id]
            percentage = (size / total_messages) * 100
            confidence_marker = "high" if theme.confidence == "high" else "medium"

            print(f"\nTheme: {theme.theme_name} {confidence_marker}")
            print(f"Size: {size:,} messages ({percentage:.1f}%)")
            print(f"Description: {theme.description}")
            print("Sample messages:")
            for msg in theme.representative_messages[:2]:
                print(f"  - {msg[:100]}...")