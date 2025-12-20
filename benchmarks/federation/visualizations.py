# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)

"""
Visualizations for Federation Benchmarks.

Generates charts and HTML reports.
"""

import os
from typing import Dict, List, Any, Optional
import json
import base64
from io import BytesIO

from .metrics import BenchmarkResults

# Try to import matplotlib, but make it optional
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def plot_latency_comparison(
    results: Dict[str, BenchmarkResults],
    output_path: str,
) -> None:
    """
    Create box plot comparing latencies across configurations.

    Args:
        results: Dict of config_name -> BenchmarkResults
        output_path: Path to save PNG
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping chart generation")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    data = []
    labels = []
    for name, result in results.items():
        latencies = [q.latency_ms for q in result.queries if q.error is None]
        if latencies:
            data.append(latencies)
            labels.append(name)

    if data:
        bp = ax.boxplot(data, labels=labels, patch_artist=True)

        # Color boxes
        colors = plt.cm.Set3.colors
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_ylabel('Latency (ms)')
        ax.set_xlabel('Configuration')
        ax.set_title('Query Latency Distribution by Configuration')
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()


def plot_precision_latency_tradeoff(
    results: Dict[str, BenchmarkResults],
    output_path: str,
) -> None:
    """
    Create scatter plot of precision vs latency tradeoff.

    Args:
        results: Dict of config_name -> BenchmarkResults
        output_path: Path to save PNG
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping chart generation")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (name, result) in enumerate(results.items()):
        ax.scatter(
            result.avg_latency_ms,
            result.avg_precision,
            s=100,
            label=name,
            marker='o',
        )
        ax.annotate(
            name,
            (result.avg_latency_ms, result.avg_precision),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    ax.set_xlabel('Average Latency (ms)')
    ax.set_ylabel('Average Precision')
    ax.set_title('Precision vs Latency Tradeoff')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_scalability(
    results_by_size: Dict[int, Dict[str, BenchmarkResults]],
    output_path: str,
    metric: str = "avg_latency_ms",
) -> None:
    """
    Create line chart showing metric vs network size.

    Args:
        results_by_size: Dict of size -> config_name -> BenchmarkResults
        output_path: Path to save PNG
        metric: Which metric to plot
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping chart generation")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    sizes = sorted(results_by_size.keys())

    # Get all config names
    config_names = set()
    for size_results in results_by_size.values():
        config_names.update(size_results.keys())

    for config_name in sorted(config_names):
        values = []
        for size in sizes:
            if config_name in results_by_size.get(size, {}):
                result = results_by_size[size][config_name]
                values.append(getattr(result, metric, 0))
            else:
                values.append(None)

        ax.plot(sizes, values, marker='o', label=config_name)

    ax.set_xlabel('Network Size (nodes)')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'{metric.replace("_", " ").title()} vs Network Size')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_query_type_breakdown(
    results: Dict[str, BenchmarkResults],
    output_path: str,
) -> None:
    """
    Create grouped bar chart of metrics by query type.

    Args:
        results: Dict of config_name -> BenchmarkResults
        output_path: Path to save PNG
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping chart generation")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    query_types = ["specific", "exploratory", "consensus"]
    config_names = list(results.keys())
    x = range(len(query_types))
    width = 0.8 / len(config_names)

    # Latency chart
    for i, name in enumerate(config_names):
        result = results[name]
        latencies = [
            result.metrics_by_type.get(qt, {}).get("avg_latency_ms", 0)
            for qt in query_types
        ]
        axes[0].bar(
            [xi + i * width for xi in x],
            latencies,
            width,
            label=name,
        )

    axes[0].set_xlabel('Query Type')
    axes[0].set_ylabel('Average Latency (ms)')
    axes[0].set_title('Latency by Query Type')
    axes[0].set_xticks([xi + width * len(config_names) / 2 for xi in x])
    axes[0].set_xticklabels(query_types)
    axes[0].legend(loc='best')

    # Precision chart
    for i, name in enumerate(config_names):
        result = results[name]
        precisions = [
            result.metrics_by_type.get(qt, {}).get("avg_precision", 0)
            for qt in query_types
        ]
        axes[1].bar(
            [xi + i * width for xi in x],
            precisions,
            width,
            label=name,
        )

    axes[1].set_xlabel('Query Type')
    axes[1].set_ylabel('Average Precision')
    axes[1].set_title('Precision by Query Type')
    axes[1].set_xticks([xi + width * len(config_names) / 2 for xi in x])
    axes[1].set_xticklabels(query_types)
    axes[1].legend(loc='best')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_report(
    results: Dict[str, BenchmarkResults],
    output_dir: str,
    results_by_size: Optional[Dict[int, Dict[str, BenchmarkResults]]] = None,
) -> str:
    """
    Generate HTML report with embedded charts.

    Args:
        results: Dict of config_name -> BenchmarkResults
        output_dir: Directory to save outputs
        results_by_size: Optional scalability results

    Returns:
        Path to HTML report
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate charts
    latency_path = os.path.join(output_dir, "latency_comparison.png")
    precision_path = os.path.join(output_dir, "precision_tradeoff.png")
    query_type_path = os.path.join(output_dir, "query_type_breakdown.png")

    plot_latency_comparison(results, latency_path)
    plot_precision_latency_tradeoff(results, precision_path)
    plot_query_type_breakdown(results, query_type_path)

    if results_by_size:
        scalability_path = os.path.join(output_dir, "scalability.png")
        plot_scalability(results_by_size, scalability_path)

    # Save JSON data
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(
            {name: r.to_summary_dict() for name, r in results.items()},
            f,
            indent=2,
        )

    # Generate HTML
    html_content = _generate_html_report(results, output_dir, results_by_size)
    html_path = os.path.join(output_dir, "federation_benchmark.html")
    with open(html_path, "w") as f:
        f.write(html_content)

    return html_path


def _generate_html_report(
    results: Dict[str, BenchmarkResults],
    output_dir: str,
    results_by_size: Optional[Dict[int, Dict[str, BenchmarkResults]]] = None,
) -> str:
    """Generate HTML content for the report."""

    # Build summary table
    summary_rows = []
    for name, result in results.items():
        summary_rows.append(f"""
        <tr>
            <td>{name}</td>
            <td>{result.p50_latency_ms:.1f}</td>
            <td>{result.p99_latency_ms:.1f}</td>
            <td>{result.avg_precision:.3f}</td>
            <td>{result.avg_recall:.3f}</td>
            <td>{result.avg_nodes_queried:.1f}</td>
            <td>{result.error_rate * 100:.1f}%</td>
        </tr>
        """)

    summary_table = "\n".join(summary_rows)

    # Check which images exist
    def img_tag(filename: str, title: str) -> str:
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            return f"""
            <div class="chart">
                <h3>{title}</h3>
                <img src="{filename}" alt="{title}">
            </div>
            """
        return ""

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Federation Benchmark Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{ color: #333; border-bottom: 2px solid #4a90d9; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        h3 {{ color: #666; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #4a90d9;
            color: white;
        }}
        tr:hover {{ background: #f5f5f5; }}
        .chart {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border-radius: 4px;
        }}
        .chart img {{
            max-width: 100%;
            height: auto;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .summary-card {{
            background: white;
            padding: 15px;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .summary-card h4 {{
            margin: 0 0 10px 0;
            color: #666;
            font-size: 12px;
            text-transform: uppercase;
        }}
        .summary-card .value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
    </style>
</head>
<body>
    <h1>Federation Benchmark Report</h1>

    <h2>Summary</h2>
    <table>
        <thead>
            <tr>
                <th>Configuration</th>
                <th>P50 Latency (ms)</th>
                <th>P99 Latency (ms)</th>
                <th>Avg Precision</th>
                <th>Avg Recall</th>
                <th>Avg Nodes Queried</th>
                <th>Error Rate</th>
            </tr>
        </thead>
        <tbody>
            {summary_table}
        </tbody>
    </table>

    <h2>Visualizations</h2>

    {img_tag("latency_comparison.png", "Latency Distribution")}
    {img_tag("precision_tradeoff.png", "Precision vs Latency Tradeoff")}
    {img_tag("query_type_breakdown.png", "Metrics by Query Type")}
    {img_tag("scalability.png", "Scalability") if results_by_size else ""}

    <h2>Configuration Details</h2>
    <p>Network size: {list(results.values())[0].network_size if results else 'N/A'} nodes</p>
    <p>Total queries: {sum(len(r.queries) for r in results.values())}</p>

    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 12px;">
        Generated by Federation Benchmark Suite (Phase 6b)
    </footer>
</body>
</html>
    """

    return html
