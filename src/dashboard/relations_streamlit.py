import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union, Literal, Dict, Any, Tuple
from config_handler import ConfigHandler

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import emoji
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler


@dataclass
class Config:
    """Configuration for the WhatsApp Analysis Dashboard."""
    config_handler: ConfigHandler = ConfigHandler()
    plot_height: int = 600
    plot_width: int = 1000
    template: str = "plotly_white"
    color_scale: str = 'Viridis'

    def get_data_path(self) -> Path:
        """Get the path to the data file using ConfigHandler."""
        self.config_handler.ensure_directories()
        return self.config_handler.get_processed_file_path()


@dataclass
class PlotSettings:
    """Settings for plot creation and display."""
    x_variable: str
    y_variable: str
    plot_type: Literal['scatter', 'line', 'bar']
    view_type: Literal['combined', 'separate']
    color_by: Optional[str] = None
    show_trendline: bool = False
    enable_zoom: bool = True


@dataclass
class DataProcessor:
    """Handles data loading and processing."""
    config: Config

    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the WhatsApp chat data."""
        data_path = self.config.get_data_path()
        df = pd.read_csv(data_path) if data_path.suffix == '.csv' else pd.read_parquet(data_path)
        return self.process_data(df)

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the loaded data with all required transformations."""
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['year'] = df['timestamp'].dt.year
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # Text-based features
        df['word_count'] = df['message'].apply(lambda x: len(re.findall(r'\w+', str(x).lower())))
        df['character_count'] = df['message'].str.len()
        df['sentiment'] = df['message'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        df['emoji_count'] = df['message'].apply(lambda x: len(''.join(c for c in str(x) if c in emoji.EMOJI_DATA)))
        df['contains_question'] = df['message'].str.contains('\?').astype(int)
        df['contains_exclamation'] = df['message'].str.contains('!').astype(int)

        # Author-based features
        author_message_counts = df['author'].value_counts()
        df['author_message_count'] = df['author'].map(author_message_counts)

        # Normalize numerical columns
        numeric_columns = ['word_count', 'character_count', 'emoji_count', 'author_message_count']
        df[numeric_columns] = MinMaxScaler().fit_transform(df[numeric_columns])

        return df


@dataclass
class Visualizer:
    """Handles plot creation and visualization."""
    config: Config

    def create_plot(self, df: pd.DataFrame, settings: PlotSettings) -> Union[go.Figure, None]:
        """Create a plot based on the provided settings."""
        try:
            color_values, color_scale = None, None
            if settings.color_by:
                color_values, color_scale = self._handle_color_settings(
                    df, settings.x_variable, settings.y_variable, settings.color_by
                )

            if settings.view_type == 'combined':
                fig = self._create_combined_plot(df, settings, color_values, color_scale)
            else:
                fig = self._create_separate_plot(df, settings)

            self._update_layout(fig, settings)
            return fig

        except Exception as e:
            return self._create_error_plot(str(e))

    def _handle_color_settings(self, df: pd.DataFrame, x_var: str, y_var: str, color_by: str) -> Tuple[
        pd.Series, Optional[str]]:
        """Handle color settings based on variable types."""
        is_x_continuous = df[x_var].dtype in ['float64', 'float32']
        is_y_continuous = df[y_var].dtype in ['float64', 'float32']

        if (is_x_continuous or is_y_continuous) and df[color_by].dtype == 'object':
            return df[color_by].astype('category').cat.codes, self.config.color_scale
        return df[color_by], None

    def _create_combined_plot(self, df: pd.DataFrame, settings: PlotSettings,
                              color_values: Optional[pd.Series], color_scale: Optional[str]) -> go.Figure:
        """Create a combined view plot."""
        plot_args = {
            'data_frame': df,
            'x': settings.x_variable,
            'y': settings.y_variable,
            'color': color_values,
            'color_continuous_scale': color_scale
        }

        if settings.plot_type == 'scatter':
            return px.scatter(
                **plot_args,
                trendline="lowess" if settings.show_trendline else None
            )
        elif settings.plot_type == 'line':
            return px.line(**plot_args)
        else:  # bar
            if settings.color_by:
                grouped = df.groupby([settings.x_variable, settings.color_by])[settings.y_variable].mean().reset_index()
            else:
                grouped = df.groupby(settings.x_variable)[settings.y_variable].mean().reset_index()
            return px.bar(grouped, x=settings.x_variable, y=settings.y_variable, color=settings.color_by)

    def _create_separate_plot(self, df: pd.DataFrame, settings: PlotSettings) -> go.Figure:
        """Create a separate view plot."""
        fig = make_subplots(rows=1, cols=2, subplot_titles=df['author'].unique())

        for i, author in enumerate(df['author'].unique(), start=1):
            author_data = df[df['author'] == author]

            if settings.plot_type == 'scatter':
                trace = go.Scatter(
                    x=author_data[settings.x_variable],
                    y=author_data[settings.y_variable],
                    mode='markers',
                    name=author
                )
            elif settings.plot_type == 'line':
                trace = go.Scatter(
                    x=author_data[settings.x_variable],
                    y=author_data[settings.y_variable],
                    mode='lines',
                    name=author
                )
            else:  # bar
                grouped = author_data.groupby(settings.x_variable)[settings.y_variable].mean().reset_index()
                trace = go.Bar(
                    x=grouped[settings.x_variable],
                    y=grouped[settings.y_variable],
                    name=author
                )

            fig.add_trace(trace, row=1, col=i)

        return fig

    def _update_layout(self, fig: go.Figure, settings: PlotSettings) -> None:
        """Update the layout of the plot."""
        fig.update_layout(
            height=self.config.plot_height,
            width=self.config.plot_width,
            template=self.config.template,
            hovermode='closest',
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            dragmode='zoom' if settings.enable_zoom else 'select'
        )

    def _create_error_plot(self, error_message: str) -> go.Figure:
        """Create an error message plot."""
        fig = go.Figure()
        fig.add_annotation(
            text=f"Could not create plot with current settings.<br>Error: {error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="red")
        )
        fig.update_layout(
            height=self.config.plot_height,
            width=self.config.plot_width,
            template=self.config.template,
            showlegend=False
        )
        return fig


@dataclass
class DashboardApp:
    """Main dashboard application class."""
    config: Config
    processor: DataProcessor
    visualizer: Visualizer

    @classmethod
    def create(cls) -> 'DashboardApp':
        """Create a new dashboard application instance."""
        config = Config()
        return cls(
            config=config,
            processor=DataProcessor(config),
            visualizer=Visualizer(config)
        )

    def initialize_session_state(self, df: pd.DataFrame) -> None:
        """Initialize Streamlit session state variables."""
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        if 'show_trendline' not in st.session_state:
            st.session_state.show_trendline = False
        if 'enable_zoom' not in st.session_state:
            st.session_state.enable_zoom = True
        if 'x_variable' not in st.session_state:
            st.session_state.x_variable = numeric_columns[0]
        if 'y_variable' not in st.session_state:
            st.session_state.y_variable = numeric_columns[1]
        if 'plot_type' not in st.session_state:
            st.session_state.plot_type = 'scatter'
        if 'view_type' not in st.session_state:
            st.session_state.view_type = 'combined'
        if 'color_by' not in st.session_state:
            st.session_state.color_by = 'None'

    def show_sidebar_controls(self, df: pd.DataFrame) -> None:
        """Display sidebar controls."""
        st.sidebar.subheader("Plot Controls")

        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Add warning for potentially problematic combinations
        selected_vars_continuous = (
                df[st.session_state.x_variable].dtype in ['float64', 'float32'] or
                df[st.session_state.y_variable].dtype in ['float64', 'float32']
        )

        if selected_vars_continuous and st.session_state.color_by != 'None':
            st.sidebar.warning(
                "⚠️ When using continuous variables (like sentiment), "
                "categorical coloring might not work well. Consider using a numerical column for coloring instead."
            )

        # Plot controls
        st.session_state.x_variable = st.sidebar.selectbox('Select X-axis variable', numeric_columns, key='x_select')
        y_options = [col for col in numeric_columns if col != st.session_state.x_variable]
        st.session_state.y_variable = st.sidebar.selectbox('Select Y-axis variable', y_options, key='y_select')

        st.session_state.plot_type = st.sidebar.radio('Select plot type', ['scatter', 'line', 'bar'],
                                                      key='plot_type_radio')
        st.session_state.view_type = st.sidebar.radio('Select view type', ['combined', 'separate'],
                                                      key='view_type_radio')

        if st.session_state.view_type == 'combined':
            st.session_state.color_by = st.sidebar.selectbox('Color by', ['None'] + categorical_columns,
                                                             key='color_select')
        else:
            st.session_state.color_by = 'None'

        st.session_state.show_trendline = st.sidebar.checkbox('Show trendline', value=st.session_state.show_trendline,
                                                              key='trendline_check')
        st.session_state.enable_zoom = st.sidebar.checkbox('Enable zoom', value=st.session_state.enable_zoom,
                                                           key='zoom_check')

    def show_main_content(self, df: pd.DataFrame) -> None:
        """Display main content area."""
        settings = PlotSettings(
            x_variable=st.session_state.x_variable,
            y_variable=st.session_state.y_variable,
            plot_type=st.session_state.plot_type,
            view_type=st.session_state.view_type,
            color_by=st.session_state.color_by if st.session_state.color_by != 'None' else None,
            show_trendline=st.session_state.show_trendline,
            enable_zoom=st.session_state.enable_zoom
        )

        col1, col2 = st.columns([3, 1])

        with col1:
            fig = self.visualizer.create_plot(df, settings)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Quick Statistics")
            correlation = df[settings.x_variable].corr(df[settings.y_variable])
            st.write(f"Correlation: {correlation:.2f}")

            if settings.plot_type == 'bar':
                avg_by_x = df.groupby(settings.x_variable)[settings.y_variable].mean()
                st.write(f"Max average {settings.y_variable}: {avg_by_x.max():.2f}")
                st.write(f"Min average {settings.y_variable}: {avg_by_x.min():.2f}")

    def run_streamlit(self) -> None:
        """Run the Streamlit dashboard."""
        st.set_page_config(
            page_title="WhatsApp Chat Analysis",
            page_icon="💬",
            layout="wide"
        )

        if 'data' not in st.session_state:
            st.session_state.data = self.processor.load_data()

        df = st.session_state.data
        self.initialize_session_state(df)

        st.title('💬 WhatsApp Chat Analysis')
        st.markdown("""
        Analyze and visualize WhatsApp chat patterns and relationships between different metrics.
        Use the sidebar to customize the visualization.
        """)

        with st.expander("📊 Basic Statistics", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Messages", len(df))
            with col2:
                st.metric("Date Range", f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
            with col3:
                st.metric("Participants", df['author'].nunique())

        self.show_sidebar_controls(df)
        self.show_main_content(df)

    def run_cli(self) -> None:
        """Run the command-line interface."""
        df = self.processor.load_data()
        print(f"Loaded {len(df)} messages")
        print(f"Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")


def main() -> None:
    """Main entry point for both Streamlit and command line usage."""
    import sys
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description='WhatsApp Chat Analysis Dashboard')
    parser.add_argument('--streamlit', action='store_true', help='Run as Streamlit dashboard')
    parser.add_argument('--data-file', type=str, help='Path to data file')
    parser.add_argument('--output-dir', type=str, help='Directory for output files')
    args = parser.parse_args()

    # Create application instance
    app = DashboardApp.create()

    # Override data file if specified
    if args.data_file:
        app.config.data_file = args.data_file

    if args.streamlit:
        # Get the current file's path
        current_file = str(Path(__file__).absolute())
        # Set up sys.argv for Streamlit run command
        sys.argv = ["streamlit", "run", current_file, "--"]
        import streamlit.web.cli as stcli
        sys.exit(stcli.main())
    else:
        app.run_cli()


if __name__ == "__main__":
    # Check if we're being run by Streamlit
    import streamlit.runtime.scriptrunner.script_runner as script_runner
    if script_runner.get_script_run_ctx():
        # We're being run by Streamlit, so just run the app
        app = DashboardApp.create()
        app.run_streamlit()
    else:
        # We're being run from the command line
        main()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()