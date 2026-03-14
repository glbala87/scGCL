"""HTML report generation for scGCL analysis."""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from datetime import datetime
import base64
import io
import warnings


@dataclass
class ReportSection:
    """A section of the HTML report."""

    title: str
    content: str
    order: int = 0


class HTMLReportGenerator:
    """Generate comprehensive HTML reports for scGCL analysis."""

    def __init__(self, title: str = "scGCL Analysis Report"):
        """
        Initialize report generator.

        Parameters
        ----------
        title : str
            Report title
        """
        self.title = title
        self.sections: List[ReportSection] = []
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def add_section(self, title: str, content: str, order: int = 0):
        """Add a section to the report."""
        self.sections.append(ReportSection(title=title, content=content, order=order))

    def add_summary(self, stats: Dict[str, Any]):
        """Add summary statistics section."""
        html = '<div class="summary-grid">'

        for key, value in stats.items():
            if isinstance(value, float):
                value_str = f"{value:.4f}"
            else:
                value_str = str(value)

            html += f'''
            <div class="stat-card">
                <div class="stat-label">{key}</div>
                <div class="stat-value">{value_str}</div>
            </div>
            '''

        html += '</div>'
        self.add_section("Summary Statistics", html, order=0)

    def add_table(self, title: str, df: pd.DataFrame, max_rows: int = 50, order: int = 10):
        """Add a DataFrame as a table section."""
        if len(df) > max_rows:
            df = df.head(max_rows)
            note = f'<p class="note">Showing first {max_rows} rows of {len(df)}</p>'
        else:
            note = ''

        # Format numeric columns
        formatted_df = df.copy()
        for col in formatted_df.select_dtypes(include=[np.floating]).columns:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "")

        table_html = formatted_df.to_html(classes='data-table', index=False)
        html = note + table_html

        self.add_section(title, html, order=order)

    def add_figure(self, title: str, fig, order: int = 5):
        """Add a matplotlib figure as embedded image."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib required for figure embedding")
            return

        # Save figure to bytes
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)

        # Encode as base64
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        html = f'<img src="data:image/png;base64,{img_base64}" class="report-figure" />'
        self.add_section(title, html, order=order)

        plt.close(fig)

    def add_plotly_figure(self, title: str, fig, order: int = 5):
        """Add a plotly figure as embedded HTML."""
        try:
            import plotly.io as pio
        except ImportError:
            warnings.warn("plotly required for interactive figures")
            return

        html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
        self.add_section(title, html, order=order)

    def add_text(self, title: str, text: str, order: int = 0):
        """Add a text section."""
        html = f'<p>{text}</p>'
        self.add_section(title, html, order=order)

    def add_markdown(self, title: str, markdown: str, order: int = 0):
        """Add markdown content (basic conversion)."""
        # Basic markdown to HTML conversion
        html = markdown

        # Headers
        lines = html.split('\n')
        converted = []
        for line in lines:
            if line.startswith('### '):
                converted.append(f'<h4>{line[4:]}</h4>')
            elif line.startswith('## '):
                converted.append(f'<h3>{line[3:]}</h3>')
            elif line.startswith('# '):
                converted.append(f'<h2>{line[2:]}</h2>')
            elif line.startswith('- '):
                converted.append(f'<li>{line[2:]}</li>')
            elif line.startswith('* '):
                converted.append(f'<li>{line[2:]}</li>')
            elif line.strip() == '':
                converted.append('<br/>')
            else:
                converted.append(f'<p>{line}</p>')

        html = '\n'.join(converted)

        # Bold
        import re
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'__(.+?)__', r'<strong>\1</strong>', html)

        # Italic
        html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
        html = re.sub(r'_(.+?)_', r'<em>\1</em>', html)

        # Code
        html = re.sub(r'`(.+?)`', r'<code>\1</code>', html)

        self.add_section(title, html, order=order)

    def _get_css(self) -> str:
        """Get CSS styles for the report."""
        return '''
        <style>
            :root {
                --primary-color: #2563eb;
                --secondary-color: #64748b;
                --bg-color: #f8fafc;
                --card-bg: #ffffff;
                --text-color: #1e293b;
                --border-color: #e2e8f0;
            }

            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }

            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                line-height: 1.6;
                color: var(--text-color);
                background-color: var(--bg-color);
                padding: 2rem;
            }

            .container {
                max-width: 1200px;
                margin: 0 auto;
            }

            header {
                text-align: center;
                margin-bottom: 2rem;
                padding: 2rem;
                background: linear-gradient(135deg, var(--primary-color), #3b82f6);
                color: white;
                border-radius: 12px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }

            header h1 {
                font-size: 2rem;
                margin-bottom: 0.5rem;
            }

            header .timestamp {
                opacity: 0.9;
                font-size: 0.9rem;
            }

            .section {
                background: var(--card-bg);
                border-radius: 8px;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                border: 1px solid var(--border-color);
            }

            .section h2 {
                color: var(--primary-color);
                font-size: 1.25rem;
                margin-bottom: 1rem;
                padding-bottom: 0.5rem;
                border-bottom: 2px solid var(--border-color);
            }

            .summary-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 1rem;
            }

            .stat-card {
                background: var(--bg-color);
                padding: 1rem;
                border-radius: 8px;
                text-align: center;
                border: 1px solid var(--border-color);
            }

            .stat-label {
                font-size: 0.85rem;
                color: var(--secondary-color);
                margin-bottom: 0.25rem;
            }

            .stat-value {
                font-size: 1.5rem;
                font-weight: 600;
                color: var(--primary-color);
            }

            .data-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 0.9rem;
            }

            .data-table th, .data-table td {
                padding: 0.75rem;
                text-align: left;
                border-bottom: 1px solid var(--border-color);
            }

            .data-table th {
                background: var(--bg-color);
                font-weight: 600;
                color: var(--secondary-color);
            }

            .data-table tr:hover {
                background: var(--bg-color);
            }

            .report-figure {
                max-width: 100%;
                height: auto;
                display: block;
                margin: 1rem auto;
                border-radius: 8px;
            }

            .note {
                font-size: 0.85rem;
                color: var(--secondary-color);
                font-style: italic;
                margin-bottom: 0.5rem;
            }

            code {
                background: var(--bg-color);
                padding: 0.2rem 0.4rem;
                border-radius: 4px;
                font-family: 'Monaco', 'Menlo', monospace;
                font-size: 0.9em;
            }

            footer {
                text-align: center;
                padding: 1rem;
                color: var(--secondary-color);
                font-size: 0.85rem;
            }

            @media print {
                body {
                    background: white;
                    padding: 0;
                }

                .section {
                    break-inside: avoid;
                    box-shadow: none;
                }
            }
        </style>
        '''

    def generate(self) -> str:
        """Generate the complete HTML report."""
        # Sort sections by order
        sorted_sections = sorted(self.sections, key=lambda x: x.order)

        sections_html = ''
        for section in sorted_sections:
            sections_html += f'''
            <div class="section">
                <h2>{section.title}</h2>
                {section.content}
            </div>
            '''

        html = f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{self.title}</title>
            {self._get_css()}
        </head>
        <body>
            <div class="container">
                <header>
                    <h1>{self.title}</h1>
                    <div class="timestamp">Generated: {self.timestamp}</div>
                </header>

                {sections_html}

                <footer>
                    Generated by scGCL v0.2.0 | Single-Cell Graph Contrastive Learning
                </footer>
            </div>
        </body>
        </html>
        '''

        return html

    def save(self, path: str):
        """Save report to file."""
        html = self.generate()
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)


def generate_clustering_report(
    expression: np.ndarray,
    labels: np.ndarray,
    embedding: Optional[np.ndarray] = None,
    gene_names: Optional[np.ndarray] = None,
    metrics: Optional[Dict] = None,
    markers: Optional[pd.DataFrame] = None,
    title: str = "scGCL Clustering Report",
    output_path: str = "report.html"
) -> str:
    """
    Generate a comprehensive clustering report.

    Parameters
    ----------
    expression : np.ndarray
        Expression matrix
    labels : np.ndarray
        Cluster labels
    embedding : np.ndarray, optional
        2D embedding for visualization
    gene_names : np.ndarray, optional
        Gene names
    metrics : Dict, optional
        Clustering metrics
    markers : pd.DataFrame, optional
        Marker gene results
    title : str
        Report title
    output_path : str
        Output file path

    Returns
    -------
    str
        Path to generated report
    """
    from .qc import compute_cluster_qc

    report = HTMLReportGenerator(title=title)

    # Summary statistics
    n_cells, n_genes = expression.shape
    n_clusters = len(np.unique(labels))

    summary_stats = {
        'Total Cells': n_cells,
        'Total Genes': n_genes,
        'Clusters': n_clusters,
        'Mean Cluster Size': int(n_cells / n_clusters)
    }

    if metrics:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                summary_stats[key.replace('_', ' ').title()] = value

    report.add_summary(summary_stats)

    # QC metrics
    qc_result = compute_cluster_qc(expression, labels, gene_names)
    report.add_table("Cluster Statistics", qc_result.cluster_stats, order=2)

    # Cluster size distribution
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        axes[0].bar(unique, counts, color='steelblue')
        axes[0].set_xlabel('Cluster')
        axes[0].set_ylabel('Number of Cells')
        axes[0].set_title('Cluster Size Distribution')

        # Embedding if available
        if embedding is not None:
            scatter = axes[1].scatter(
                embedding[:, 0], embedding[:, 1],
                c=labels, cmap='tab20', alpha=0.6, s=5
            )
            axes[1].set_xlabel('UMAP1')
            axes[1].set_ylabel('UMAP2')
            axes[1].set_title('UMAP Embedding')
            plt.colorbar(scatter, ax=axes[1], label='Cluster')
        else:
            axes[1].text(0.5, 0.5, 'Embedding not provided',
                        ha='center', va='center', transform=axes[1].transAxes)

        plt.tight_layout()
        report.add_figure("Cluster Overview", fig, order=1)

    except ImportError:
        pass

    # Marker genes if provided
    if markers is not None and len(markers) > 0:
        # Top markers per cluster
        if 'cluster' in markers.columns:
            top_markers = markers.groupby('cluster').head(5)
            report.add_table("Top Marker Genes", top_markers, order=3)

    # Methods section
    methods_text = """
    **Clustering Method:** scGCL (Single-Cell Graph Contrastive Learning)

    The scGCL framework uses graph neural networks combined with contrastive learning
    to learn meaningful representations of single-cell RNA sequencing data. The learned
    embeddings are then clustered using standard algorithms.

    **QC Metrics:**
    - Total counts: Sum of all gene expression values per cell
    - Genes detected: Number of genes with non-zero expression
    - Cluster purity: Homogeneity of cell types within clusters
    """
    report.add_markdown("Methods", methods_text, order=99)

    # Save report
    report.save(output_path)

    return output_path


def generate_comparison_report(
    results: List[Dict],
    labels_list: List[np.ndarray],
    method_names: List[str],
    embedding: Optional[np.ndarray] = None,
    title: str = "Clustering Comparison Report",
    output_path: str = "comparison_report.html"
) -> str:
    """
    Generate a report comparing multiple clustering results.

    Parameters
    ----------
    results : List[Dict]
        List of result dictionaries with metrics
    labels_list : List[np.ndarray]
        List of cluster label arrays
    method_names : List[str]
        Names of clustering methods
    embedding : np.ndarray, optional
        2D embedding for visualization
    title : str
        Report title
    output_path : str
        Output file path

    Returns
    -------
    str
        Path to generated report
    """
    report = HTMLReportGenerator(title=title)

    # Comparison table
    comparison_data = []

    for name, result, labels in zip(method_names, results, labels_list):
        row = {
            'Method': name,
            'Clusters': len(np.unique(labels))
        }

        for key, value in result.items():
            if isinstance(value, (int, float)):
                row[key] = value

        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)
    report.add_table("Method Comparison", comparison_df, order=0)

    # Visual comparison
    if embedding is not None:
        try:
            import matplotlib.pyplot as plt

            n_methods = len(method_names)
            fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4))

            if n_methods == 1:
                axes = [axes]

            for ax, name, labels in zip(axes, method_names, labels_list):
                scatter = ax.scatter(
                    embedding[:, 0], embedding[:, 1],
                    c=labels, cmap='tab20', alpha=0.6, s=5
                )
                ax.set_title(name)
                ax.set_xlabel('UMAP1')
                ax.set_ylabel('UMAP2')

            plt.tight_layout()
            report.add_figure("Visual Comparison", fig, order=1)

        except ImportError:
            pass

    # Save report
    report.save(output_path)

    return output_path
