from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import Memory
from tqdm import tqdm
import nltk
from base_plotter import BasePlotter


@dataclass
class ClusteringConfig:
    n_clusters: int = 8  # Number of main topics to identify
    random_state: int = 42


@dataclass
class VectorizerConfig:
    min_df: int = 10
    max_features: int = 3000
    stop_words: str = 'english'
    ngram_range: Tuple[int, int] = (1, 2)


@dataclass
class VisualizationConfig:
    figure_size: Tuple[int, int] = (15, 10)
    dpi: int = 300
    cmap: str = 'tab10'
    alpha: float = 0.6
    style: str = 'seaborn-v0_8-darkgrid'


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
    cluster_labels: Dict[int, str]
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

    def analyze_and_visualize(self, df: pd.DataFrame, output_dir: Path) -> AnalysisResults:
        """Main analysis and visualization function."""
        try:
            print("\n=== Starting Topic Analysis ===")
            output_dir.mkdir(exist_ok=True)

            texts = self._preprocess_texts(df['message'].tolist())
            print(f"Analyzing {len(texts)} messages...")

            print("Vectorizing texts...")
            vectors = self._get_text_vectors(texts)
            print(f"Vector shape: {vectors.shape}")

            print("Identifying main topics...")
            kmeans = KMeans(
                n_clusters=self.clustering_config.n_clusters,
                random_state=self.clustering_config.random_state,
                n_init=10
            )
            clusters = kmeans.fit_predict(vectors)

            cluster_keywords = self._extract_cluster_keywords(texts, clusters)

            cluster_sizes = {
                cluster_id: np.sum(clusters == cluster_id)
                for cluster_id in range(self.clustering_config.n_clusters)
            }

            cluster_labels = {
                cluster_id: f"Topic {cluster_id + 1}: {' & '.join(keywords[:2])}"
                for cluster_id, keywords in cluster_keywords.items()
            }

            results = AnalysisResults(
                embedded_data=vectors,
                clusters=clusters,
                cluster_keywords=cluster_keywords,
                cluster_labels=cluster_labels,
                cluster_sizes=cluster_sizes
            )

            print("\nGenerating visualizations...")
            self._create_visualizations(results, texts, output_dir)

            print("\n=== Analysis Complete ===")
            self._print_topic_summary(results)

            return results

        except Exception as e:
            print(f"\n❌ Error in analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

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
        """Create a single visualization using BasePlotter."""
        print(f"Generating {method} visualization...")

        # Perform dimensionality reduction
        if method == "t-SNE":
            embedded = TSNE(n_components=2, random_state=42).fit_transform(results.embedded_data)
            title = 'Message Topics (t-SNE)'
        else:  # PCA
            pca = PCA(n_components=2, random_state=42)
            embedded = pca.fit_transform(results.embedded_data)
            title = f'Message Topics (PCA)\nTotal Variance Explained: {sum(pca.explained_variance_ratio_):.1%}'

        fig, ax = plotter.setup_figure(title)

        scatter = ax.scatter(
            embedded[:, 0], embedded[:, 1],
            c=results.clusters,
            cmap=self.viz_config.cmap,
            alpha=self.viz_config.alpha,
            s=50
        )

        ax.set_xlabel(f'{method} Dimension 1')
        ax.set_ylabel(f'{method} Dimension 2')

        # Create legend
        sorted_topics = sorted(results.cluster_sizes.items(), key=lambda x: x[1], reverse=True)
        legend_elements = []
        legend_labels = []

        cmap = plt.get_cmap(self.viz_config.cmap)
        n_clusters = self.clustering_config.n_clusters
        colors = [cmap(i / n_clusters) for i in range(n_clusters)]
        total_messages = sum(results.cluster_sizes.values())

        for cluster_id, size in sorted_topics:
            patch = plt.Circle(
                (0, 0),
                radius=1,
                color=colors[cluster_id],
                alpha=self.viz_config.alpha
            )
            legend_elements.append(patch)

            percentage = (size / total_messages) * 100
            keywords = results.cluster_keywords[cluster_id][:3]
            label = f"Topic {cluster_id + 1}: {', '.join(keywords)}\n({size:,} msgs, {percentage:.1f}%)"
            legend_labels.append(label)

        ax.legend(
            legend_elements,
            legend_labels,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            fontsize=8,
            title="Topics by Size",
            title_fontsize=10,
            borderaxespad=0,
            frameon=True,
            fancybox=True,
            shadow=True
        )

        output_path = output_dir / f'{method.lower()}_topics.png'
        plotter.save_plot(str(output_path))

    def _print_topic_summary(self, results: AnalysisResults) -> None:
        """Print a summary of the topics found."""
        print("\nTopic Summary:")
        print("=" * 50)

        sorted_topics = sorted(
            results.cluster_sizes.items(),
            key=lambda x: x[1],
            reverse=True
        )

        total_messages = sum(results.cluster_sizes.values())

        for cluster_id, size in sorted_topics:
            percentage = (size / total_messages) * 100
            keywords = results.cluster_keywords[cluster_id]

            print(f"\n{results.cluster_labels[cluster_id]}")
            print(f"Size: {size:,} messages ({percentage:.1f}%)")
            print(f"Keywords: {', '.join(keywords)}")

    def _get_text_vectors(self, texts: List[str]) -> np.ndarray:
        """Convert texts to TF-IDF vectors."""
        self.vectorizer = TfidfVectorizer(
            min_df=self.vectorizer_config.min_df,
            max_features=self.vectorizer_config.max_features,
            stop_words=self.vectorizer_config.stop_words,
            ngram_range=self.vectorizer_config.ngram_range
        )
        return self.vectorizer.fit_transform(texts).toarray()