import glob
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

@dataclass
class ComparisonMetrics:
    """Container for comparison metrics between heatmaps, perturbations, and masks."""

    # Attack details
    location: tuple[int, int]  # Perturbation location (x, y)
    within_mask: Optional[float] # Whether perturbation is within the mask
    within_heatmap: bool # Whether perturbation is within the heatmap (heatmap > 0.5)
    heatmap_at_perturbation: float  # Mean heatmap value at perturbation location

    # Optional Relationship between Heatmap and Mask(MSPS)
    mask_heatmap_agreement: Optional[float]  # Agreement between mask and heatmap regions (heatmap > 0.5)
    mean_heatmap_mask_value: Optional[float] # Mean heatmap value within the mask
    min_heatmap_mask_value: Optional[float] # Min heatmap value within the mask
    max_heatmap_mask_value: Optional[float] # Max heatmap value within the mask

    # Mask
    pixels_mask: Optional[float] # Number of pixels in the mask


class Analysis:
    def __init__(self, json_file, results_dir=None, xai_results=None, output_dir: str = "./analysis"):
        """
        Initialize the Analysis object.

        Args:
            json_file:
                Path to the JSON file containing the AttackResults (OnePixel).
            results_dir:
                Path to the directory containing the results JSON files and adversarial examples.
            xai_results:
                Path to the directory containing the XAI results (responsibility and explanations).
            output_dir:
                Path to the directory where the analysis results will be saved.
        """
        self.results_dir = Path(results_dir) if results_dir else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load results JSON
        if self.results_dir is not None:
            assert self.results_dir.is_dir(), f"Results directory {self.results_dir} does not exist."
            results_json_files = glob.glob(str(self.results_dir / "*.json"))
            assert results_json_files, "No results JSON files found."
            results_json = results_json_files[0]
        else:
            assert json_file.endswith(".json"), "json_file must be a .json file"
            assert Path(json_file).exists(), f"JSON file {json_file} does not exist."
            results_json = json_file

        with open(results_json, 'r') as f:
            results = json.load(f)

        self.raw_results = results
        self.metadata = results.get('metadata', {})
        self.statistics = results.get('statistics', {})
        self.samples = pd.json_normalize(results.get('samples', []))

        # Load adversarial examples if results_dir provided
        if self.results_dir is not None:
            self.adv_examples_dir = self.results_dir / "adversarial_examples"
            self.adv_files = glob.glob(str(self.adv_examples_dir / "*.pt")) if self.adv_examples_dir.exists() else []
        else:
            self.adv_files = []

        # Load XAI results if provided
        self.xai_results_npy = {}
        self.xai_results_csv = {}
        if xai_results is not None:
            self.xai_results = Path(xai_results)
            assert self.xai_results.is_dir(), f"XAI results directory {self.xai_results} does not exist."
            npy_files = glob.glob(str(self.xai_results / "*.npy"))
            csv_files = glob.glob(str(self.xai_results / "*.csv"))
            self.xai_results_npy = {Path(f).stem: f for f in npy_files}
            self.xai_results_csv = {Path(f).stem: pd.read_csv(f) for f in csv_files}
            print(f"Loaded {len(npy_files)} XAI npy files and {len(csv_files)} XAI csv files from {self.xai_results}")
            print(f"XAI npy files: {list(self.xai_results_npy.keys())}")

    # ============ LOADER FUNCTIONS ============
    def load_adversarial_example(self, image_name: str) -> Dict:
        """Load adversarial example data for a specific image."""
        files = glob.glob(str(self.adv_examples_dir / f"*{image_name}*.pt"))
        if not files:
            raise FileNotFoundError(f"No adversarial example found for image {image_name}")
        return torch.load(files[0])

    def get_perturbation(self, image_name: str) -> np.ndarray:
        """Get binary perturbation mask (where attack occurred)."""
        adv_data = self.load_adversarial_example(image_name)
        pert = adv_data['perturbation']
        if isinstance(pert, torch.Tensor):
            pert = pert.cpu().numpy()
        if pert.ndim == 4:
            return pert.squeeze()[0]
        elif pert.ndim == 3:
            return pert[0]
        else:
            return pert

    def get_heatmap(self, image_name: str, heatmap_key: str) -> np.ndarray:
        """Get heatmap (XAI result like responsibility or gradient)."""
        key = f"{image_name}_{heatmap_key}"
        if key in self.xai_results_npy:
            return np.load(self.xai_results_npy[key])
        raise KeyError(f"Heatmap {key} not found in XAI results")

    def get_mask(self, image_name: str, mask_key: str) -> np.ndarray:
        """Get mask (e.g., object mask, xai mask)."""
        # Try to load from npy file
        key = f"{image_name}_{mask_key}"
        if key in self.xai_results_npy:
            mask = np.load(self.xai_results_npy[key])
            if len(mask.shape) == 4:
                return mask.squeeze()[0]
            elif len(mask.shape) == 3:
                return mask[0]
            else:
                return mask
        raise KeyError(f"Mask {key} not found")

    # ============ NORMALIZATION FUNCTIONS ============

    def normalize_to_01(self, data: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1] range."""
        data = np.asarray(data, dtype=np.float32)
        vmin, vmax = data.min(), data.max()
        if vmax == vmin:
            return np.zeros_like(data)
        return (data - vmin) / (vmax - vmin)

    def binarize(self, data: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Convert continuous values to binary using threshold."""
        return (data > threshold).astype(np.float32)

    # ============ METRIC FUNCTIONS ============
    def compute_heatmap_at_perturbation(self, heatmap: np.ndarray, perturbation: np.ndarray) -> float:
        """Mean heatmap value at perturbation locations."""
        location = np.nonzero(perturbation)
        return heatmap[location].mean() if location[0].size > 0 else 0.0

    def compute_mask_coverage(self, perturbation: np.ndarray, mask: np.ndarray) -> float:
        """Fraction of perturbation that overlaps with mask."""
        if perturbation.sum() == 0:
            return 0.0
        overlap = np.logical_and(perturbation > 0, mask > 0).sum()
        return overlap / perturbation.sum()

    def compute_mask_heatmap_agreement(self, heatmap: np.ndarray, mask: np.ndarray) -> float:
        """Agreement between high heatmap values and mask regions."""
        heatmap_bin = self.binarize(heatmap)
        mask_bin = self.binarize(mask)
        agreement = np.logical_and(heatmap_bin, mask_bin).sum() / np.logical_or(heatmap_bin, mask_bin).sum()
        return agreement

    # ============ COMPREHENSIVE COMPARISON FUNCTIONS ============

    def compare_heatmap_vs_perturbation(
        self,
        image_name: str,
        heatmap_key: str,
        heatmap_threshold: float = 0.5,
        normalize_heatmap: bool = True
    ) -> ComparisonMetrics:
        """Compute all metrics comparing a heatmap to the perturbation."""
        heatmap = self.get_heatmap(image_name, heatmap_key)
        perturbation = self.get_perturbation(image_name)

        # Normalize
        if normalize_heatmap:
            heatmap = self.normalize_to_01(heatmap)
        heatmap = np.squeeze(heatmap)

        # Binarize for some metrics
        heatmap_bin = self.binarize(heatmap, heatmap_threshold)
        perturbation_bin = perturbation
        location = np.nonzero(perturbation)

        return ComparisonMetrics(
            location=tuple(location),  # Get perturbation locations
            within_heatmap=heatmap_bin[location] == 1,  # Check if any perturbation is in the heatmap
            heatmap_at_perturbation=self.compute_heatmap_at_perturbation(heatmap, perturbation),
            within_mask=None,
            pixels_mask=None,
            mask_heatmap_agreement=None,
            mean_heatmap_mask_value=None,
            min_heatmap_mask_value=None,
            max_heatmap_mask_value=None,
        )

    def compare_with_mask(
        self,
        image_name: str,
        heatmap_key: str,
        mask_key: str,
        heatmap_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        normalize_heatmap: bool = True
    ) -> ComparisonMetrics:
        """Compute comparison metrics including mask analysis."""
        heatmap = self.get_heatmap(image_name, heatmap_key)
        perturbation = self.get_perturbation(image_name)
        mask = self.get_mask(image_name, mask_key)
        combined_mask_heatmap = np.where(mask, heatmap, 0.0)

        # Normalize
        if normalize_heatmap:
            heatmap = self.normalize_to_01(heatmap)
        # heatmap = np.squeeze(heatmap)
        # mask = np.squeeze(mask)

        # Binarize
        heatmap_bin = self.binarize(heatmap, heatmap_threshold)
        perturbation_bin = perturbation
        mask_bin = self.binarize(mask, mask_threshold)
        location = np.nonzero(perturbation)

        return ComparisonMetrics(
            location=tuple(location),  # Get perturbation locations
            within_mask=mask_bin[location] == 1,  # Check if any perturbation is within the mask
            within_heatmap=heatmap_bin[location] == 1,  # Check if any perturbation is in the heatmap
            heatmap_at_perturbation=self.compute_heatmap_at_perturbation(heatmap, perturbation),
            pixels_mask=np.count_nonzero(mask_bin),
            mask_heatmap_agreement=self.compute_mask_heatmap_agreement(heatmap, mask_bin),
            mean_heatmap_mask_value=combined_mask_heatmap.sum() / np.count_nonzero(mask_bin) if np.count_nonzero(mask_bin) > 0 else 0.0,
            min_heatmap_mask_value=combined_mask_heatmap[combined_mask_heatmap > 0].min() if combined_mask_heatmap.nonzero()[0].size > 0 else 0.0,
            max_heatmap_mask_value=combined_mask_heatmap.max(),
        )

    # ============ VISUALIZATION FUNCTIONS ============

    def plot_heatmap(self, heatmap: np.ndarray, title: str = "", cmap: str = "hot", figsize: Tuple = (6, 6)) -> plt.Figure:
        """Plot a single heatmap."""
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(heatmap, cmap=cmap)
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        return fig

    def plot_perturbation(self, perturbation: np.ndarray, title: str = "", figsize: Tuple = (6, 6)) -> plt.Figure:
        """Plot perturbation as binary mask."""
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(perturbation, cmap='Reds', alpha=0.7)
        ax.set_title(title)
        return fig

    def plot_mask(self, mask: np.ndarray, title: str = "", figsize: Tuple = (6, 6)) -> plt.Figure:
        """Plot a mask."""
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(mask, cmap='Blues', alpha=0.7)
        ax.set_title(title)
        return fig

    def plot_comparison(
        self,
        image_name: str,
        heatmap_key: str,
        mask_key: Optional[str] = None,
        heatmap_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        normalize_heatmap: bool = True,
        figsize: Tuple = (18, 6)
    ) -> plt.Figure:
        """Create side-by-side comparison plot of heatmap, perturbation, and optionally mask."""
        heatmap = self.get_heatmap(image_name, heatmap_key)
        perturbation = self.get_perturbation(image_name)

        if normalize_heatmap:
            heatmap = self.normalize_to_01(heatmap)
        heatmap = np.squeeze(heatmap)
        perturbation = np.squeeze(perturbation)

        num_plots = 3 if mask_key is None else 4
        fig, axes = plt.subplots(1, num_plots, figsize=figsize)

        # Heatmap
        im0 = axes[0].imshow(heatmap, cmap='hot')
        axes[0].set_title(f"Heatmap ({heatmap_key})")
        plt.colorbar(im0, ax=axes[0])

        # Perturbation
        axes[1].imshow(perturbation, cmap='Reds', alpha=0.7)
        axes[1].set_title("Perturbation")

        # Binarized heatmap overlay
        heatmap_bin = self.binarize(heatmap, heatmap_threshold)
        perturbation_bin = perturbation
        axes[2].imshow(heatmap_bin, cmap='Greens', alpha=0.5, label='Heatmap')
        axes[2].imshow(perturbation_bin, cmap='Reds', alpha=0.5, label='Perturbation')
        axes[2].set_title(f"Combined (Green=Heatmap, Red=Perturbation)")

        # Mask if provided
        if mask_key is not None:
            mask = self.get_mask(image_name, mask_key)
            mask = np.squeeze(mask)
            mask_bin = self.binarize(mask, mask_threshold)
            axes[3].imshow(mask_bin, cmap='Blues', alpha=0.7)
            axes[3].set_title(f"Mask ({mask_key})")

        plt.tight_layout()
        return fig

    def plot_metrics_table(self, metrics: ComparisonMetrics, title: str = "") -> plt.Figure:
        """Create a table visualization of metrics."""
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis('off')

        data = [
            ['Metric', 'Value'],
            ['Perturbation Location', str(metrics.location)],
            ['Within Mask', str(metrics.within_mask)],
            ['Within Heatmap', str(metrics.within_heatmap)],
            ['Heatmap at Perturbation', f"{metrics.heatmap_at_perturbation:.4f}"],
            ['Pixels in Mask', str(metrics.pixels_mask)],
            ['Mask-Heatmap Agreement', f"{metrics.mask_heatmap_agreement:.4f}"],
            ['Mean Heatmap in Mask', f"{metrics.mean_heatmap_mask_value:.4f}"],
            ['Min Heatmap in Mask', f"{metrics.min_heatmap_mask_value:.4f}"],
            ['Max Heatmap in Mask', f"{metrics.max_heatmap_mask_value:.4f}"],
        ]

        table = ax.table(cellText=data, cellLoc='left', loc='center', colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax.set_title(title, fontsize=12, fontweight='bold')
        return fig

    # ============ BATCH ANALYSIS FUNCTIONS ============

    def analyze_all_samples(
        self,
        heatmap_key: str,
        mask_key: Optional[str] = None,
        heatmap_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        save_results: bool = True,
        save_visualizations: bool = False
    ) -> pd.DataFrame:
        """Analyze all successful attacks and compute metrics."""
        results = []

        for idx, sample in tqdm(self.samples.iterrows(), total=len(self.samples), desc="Analyzing samples"):
            if not sample.get('attack_success', False):
                continue

            image_name = sample['image_name'].strip('.JPEG')
            class_name = sample['class_name']
            pred_clean = sample['pred_clean']
            pred_adv = sample['pred_adv']

            try:
                if mask_key:
                    metrics = self.compare_with_mask(
                        image_name, heatmap_key, mask_key,
                        heatmap_threshold, mask_threshold
                    )
                else:
                    metrics = self.compare_heatmap_vs_perturbation(
                        image_name, heatmap_key, heatmap_threshold
                    )
                if save_visualizations:
                    fig = self.plot_comparison(
                            image_name, heatmap_key, mask_key,
                            heatmap_threshold, mask_threshold
                        )
                    fig.savefig(self.output_dir / f"{image_name}_{mask_key}_{heatmap_key}_comparison.png", dpi=100)
                    plt.close(fig)
            except KeyError as e:
                print(f"Error processing {image_name}: {e}")
                continue

            result = {
                'image_name': image_name,
                'class_name': class_name,
                'pred_clean': pred_clean,
                'pred_adv': pred_adv,
                **vars(metrics)
            }
            results.append(result)


        df = pd.DataFrame(results)

        if save_results:
            output_file = self.output_dir / f"analysis_{heatmap_key}{'_' + mask_key if mask_key else ''}.csv"
            df.to_csv(output_file, index=False)
            print(f"Saved analysis to {output_file}")

        return df

    def get_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """Compute summary statistics from analysis results."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if df['within_mask'][0] != None:
            df['within_mask'] = df['within_mask'].map(lambda x: True if 'True' in x else False)
        df['within_heatmap'] = df['within_heatmap'].map(lambda x: True if 'True' in x else False)
        bool_cols = df.select_dtypes(include=[bool]).columns
        exclude_col = ['pred_clean', 'pred_adv']
        cols = [col for col in numeric_cols if col not in exclude_col] + list(bool_cols)
        return {
            'mean': df[cols].mean().to_dict(),
            'std': df[cols].std().to_dict(),
            'count': df[cols].count().to_dict(),
        }

    def get_attack_success(self):
        """Print attack success statistics."""
        attack_success = self.samples['attack_success']
        print("Attack success:", attack_success.value_counts())
        print(f"Attack success: {attack_success.sum()}/{len(attack_success)}")

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze OnePixel attack results with XAI comparisons.")
    parser.add_argument("--json_file", type=str, help="Path to results JSON file.")
    parser.add_argument("--results_dir", type=str, help="Directory containing results JSON and adversarial examples.")
    parser.add_argument("--xai_results", type=str, help="Directory containing XAI results (npy and csv files).")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save analysis results.")
    parser.add_argument("--heatmap_key", type=str, default="responsibility", help="Key for heatmap in XAI results.")
    parser.add_argument("--mask_keys", type=str, metavar='N', nargs='+', help="Key(s) for mask in XAI results (optional).")
    parser.add_argument("--heatmap_threshold", type=float, default=0.5, help="Threshold for binarizing heatmap.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.output_dir is None:
        name = args.results_dir.split('.')[0].replace("/", "_")
        args.output_dir = f"analysis_{name}_{args.heatmap_key}{'_' + args.mask_key if args.mask_key else ''}"

    analysis = Analysis(
        json_file=args.json_file,
        results_dir=args.results_dir,
        xai_results=args.xai_results,
        output_dir=args.output_dir
    )
    analysis.get_attack_success()

    if args.mask_keys == None:
        df = analysis.analyze_all_samples(
            heatmap_key=args.heatmap_key,
            heatmap_threshold=args.heatmap_threshold,
            save_results=True,
            save_visualizations=True
        )
        summary = analysis.get_summary_statistics(df)
        print("Summary statistics:")
        print(json.dumps(summary, indent=2))
    else:
        dfs = []
        for mask_key in args.mask_keys:
            df = analysis.analyze_all_samples(
                heatmap_key=args.heatmap_key,
                mask_key=mask_key,
                heatmap_threshold=args.heatmap_threshold,
                save_results=True,
                save_visualizations=True
            )
            dfs.append(df)

        # Summarise Results for each df
        for df, mask_key in zip(dfs, args.mask_keys):
            summary = analysis.get_summary_statistics(df)
            print(f"Summary statistics for mask {mask_key}:")
            # print table
            summarised = pd.DataFrame()
            summarised['mean'] = pd.DataFrame(summary['mean'], index=['mean']).T
            summarised['std'] = pd.DataFrame(summary['std'], index=['std']).T
            summarised['count'] = pd.DataFrame(summary['count'], index=['count']).T

            print(summarised)
            # save
            summarised.to_csv(analysis.output_dir / f"summary_{mask_key}_{args.heatmap_key}.csv")