#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
Example usage:

python /n/fs/ragr-research/users/wh8114/projects/troupe/scripts/subsampled_leaves_experiment/evaluating/plot_results.py 
    
"""

working_dir=f"/n/fs/ragr-research/users/wh8114/projects/troupe"
csv_path=f"{working_dir}/results/subsampled_leaves_4_terminals/classe/fundamental/parameter_comparison_summary.csv"


def main():
    df = pd.read_csv(csv_path)
    
    x_key = 'num_leaves' # 'num_trees'
    y_key = 'birth_kernel_rel_error' # 'birth_kernel_rel_error', 'growth_rate_error', 'growth_rate_rel_error' , 'birth_kernel_error'
    sns.scatterplot(data=df, x=x_key, y=y_key, alpha=0.75, hue='subsample')

    file_name = f"{y_key}-by-{x_key}-scatter.pdf"
    plt.legend()
    plt.xscale('log')
    plt.savefig(f"{working_dir}/results/subsampled_leaves_4_terminals/classe/fundamental/figures/{file_name}")


if __name__ == "__main__":
    main()