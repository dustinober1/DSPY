"""
Visualization utilities for experiment results.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path


# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_accuracy_comparison(
    results: Dict[str, float],
    title: str = "Model Performance Comparison",
    ylabel: str = "Accuracy",
    save_path: Optional[Path] = None,
):
    """
    Plot bar chart comparing model accuracies.
    
    Args:
        results: Dict mapping model names to accuracy scores
        title: Plot title
        ylabel: Y-axis label
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(results.keys())
    scores = list(results.values())
    
    # Create bar plot
    bars = ax.bar(models, scores, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Color code bars
    colors = ['#ff6b6b' if 'zero' in m.lower() else 
              '#ffd93d' if 'few' in m.lower() else 
              '#6bcf7f' for m in models]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, max(scores) * 1.15)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig


def plot_comparison_table(
    results: Dict[str, Dict[str, Any]],
    metrics: List[str] = ['accuracy', 'avg_tokens', 'cost_estimate'],
    title: str = "Model Comparison",
    save_path: Optional[Path] = None,
):
    """
    Create a multi-metric comparison table visualization.
    
    Args:
        results: Dict mapping model names to metric dictionaries
        metrics: List of metric names to display
        title: Plot title
        save_path: Path to save figure
    """
    # Convert to DataFrame
    df = pd.DataFrame(results).T
    df = df[metrics] if all(m in df.columns for m in metrics) else df
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create table
    table_data = []
    for model in df.index:
        row = [model]
        for metric in df.columns:
            value = df.loc[model, metric]
            if isinstance(value, float):
                if 'accuracy' in metric.lower() or 'f1' in metric.lower():
                    row.append(f"{value:.1%}")
                else:
                    row.append(f"{value:.2f}")
            else:
                row.append(str(value))
        table_data.append(row)
    
    # Create table plot
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=table_data,
        colLabels=['Model'] + list(df.columns),
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(df.columns) + 1):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(df.columns) + 1):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved table to {save_path}")
    
    return fig


def plot_optimization_progress(
    scores: List[float],
    title: str = "Optimization Progress",
    xlabel: str = "Iteration",
    ylabel: str = "Score",
    save_path: Optional[Path] = None,
):
    """
    Plot optimization progress over iterations.
    
    Args:
        scores: List of scores per iteration
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = range(1, len(scores) + 1)
    
    ax.plot(iterations, scores, marker='o', linewidth=2, markersize=6)
    ax.fill_between(iterations, scores, alpha=0.3)
    
    # Mark best score
    best_idx = np.argmax(scores)
    ax.plot(best_idx + 1, scores[best_idx], 'r*', markersize=15, 
            label=f'Best: {scores[best_idx]:.1%}')
    
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig


def plot_cost_performance_tradeoff(
    results: Dict[str, Dict[str, float]],
    performance_metric: str = 'accuracy',
    cost_metric: str = 'cost_estimate',
    title: str = "Cost vs Performance",
    save_path: Optional[Path] = None,
):
    """
    Plot cost-performance tradeoff scatter plot.
    
    Args:
        results: Dict mapping model names to metrics
        performance_metric: Name of performance metric
        cost_metric: Name of cost metric
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    models = []
    performance = []
    costs = []
    
    for model, metrics in results.items():
        if performance_metric in metrics and cost_metric in metrics:
            models.append(model)
            performance.append(metrics[performance_metric])
            costs.append(metrics[cost_metric])
    
    # Create scatter plot
    scatter = ax.scatter(costs, performance, s=200, alpha=0.6, edgecolors='black', linewidth=2)
    
    # Add labels
    for i, model in enumerate(models):
        ax.annotate(model, (costs[i], performance[i]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
    
    ax.set_xlabel(cost_metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_ylabel(performance_metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig


def plot_error_analysis(
    error_categories: Dict[str, int],
    title: str = "Error Analysis",
    save_path: Optional[Path] = None,
):
    """
    Plot pie chart of error categories.
    
    Args:
        error_categories: Dict mapping error type to count
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    categories = list(error_categories.keys())
    counts = list(error_categories.values())
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=categories,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    
    # Style
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig


def create_summary_dashboard(
    gsm8k_results: Dict[str, float],
    hotpotqa_results: Dict[str, float],
    save_path: Optional[Path] = None,
):
    """
    Create a comprehensive dashboard with multiple visualizations.
    
    Args:
        gsm8k_results: GSM8K accuracy results
        hotpotqa_results: HotPotQA results
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # GSM8K comparison
    ax1 = fig.add_subplot(gs[0, 0])
    models = list(gsm8k_results.keys())
    scores = list(gsm8k_results.values())
    bars1 = ax1.bar(models, scores, alpha=0.8, edgecolor='black')
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom', fontweight='bold')
    ax1.set_title('GSM8K Results', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontweight='bold')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # HotPotQA comparison
    ax2 = fig.add_subplot(gs[0, 1])
    models = list(hotpotqa_results.keys())
    scores = list(hotpotqa_results.values())
    bars2 = ax2.bar(models, scores, alpha=0.8, edgecolor='black', color='orange')
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom', fontweight='bold')
    ax2.set_title('HotPotQA Results', fontsize=12, fontweight='bold')
    ax2.set_ylabel('F1 Score', fontweight='bold')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Overall improvement
    ax3 = fig.add_subplot(gs[1, :])
    
    # Combine results
    all_models = set(gsm8k_results.keys()) | set(hotpotqa_results.keys())
    x = np.arange(len(all_models))
    width = 0.35
    
    gsm8k_vals = [gsm8k_results.get(m, 0) for m in all_models]
    hotpot_vals = [hotpotqa_results.get(m, 0) for m in all_models]
    
    bars3 = ax3.bar(x - width/2, gsm8k_vals, width, label='GSM8K', alpha=0.8)
    bars4 = ax3.bar(x + width/2, hotpot_vals, width, label='HotPotQA', alpha=0.8)
    
    ax3.set_ylabel('Score', fontweight='bold')
    ax3.set_title('Overall Comparison', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(all_models, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    fig.suptitle('DSPy Small-to-SOTA Demo Results', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved dashboard to {save_path}")
    
    return fig


__all__ = [
    "plot_accuracy_comparison",
    "plot_comparison_table",
    "plot_optimization_progress",
    "plot_cost_performance_tradeoff",
    "plot_error_analysis",
    "create_summary_dashboard",
]
