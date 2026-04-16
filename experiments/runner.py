#!/usr/bin/env python
"""
Full experiment runner for adversarial attacks.

- Multiple attack types (OnePixel, Pixle etc.)
- Mask-based attacks (restrict attacks to specific image regions)
- Multiple models and datasets
- Batch processing and progress tracking
- Results saving and visualization


python runner.py --model resnet152 --attack onepixel --pixels 1 --num-samples 100
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import ImageNetDataset
from models import get_preprocessing, load_model
from torchattacks import OnePixel, Pixle, CW

def load_from_file(mask_path: str) -> torch.Tensor:
    """Load mask from file (.pt or .npy)."""
    if mask_path.endswith('.pt'):
        mask = torch.load(mask_path)
    elif mask_path.endswith('.npy'):
        mask = torch.tensor(np.load(mask_path), dtype=torch.bool)
    else:
        raise ValueError("Mask file must be .pt or .npy")

    return mask.to(torch.bool)

def get_attack(attack_name: str, model: nn.Module, **kwargs):
    """Get attack object."""
    attack_map = {
        'onepixel': OnePixel,
        'pixle': Pixle,
    }

    if attack_name not in attack_map:
        raise ValueError(f"Unknown attack: {attack_name}")
    
    logging.info(f"Creating attack: {attack_name} with params: {kwargs}")
    attack_class = attack_map[attack_name]
    attack = attack_class(model, **kwargs)
    
    return attack

class ExperimentRunner:
    def __init__(
        self,
        output_dir: str = "./results",
        device: Optional[torch.device] = None,
        verbose: bool = True
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
    
    def _setup_logging(self):
        """Setup logging to file and console."""
        log_file = self.output_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def run(
        self,
        model_name: str,
        attack_name: str,
        dataset_path: str,
        num_samples: int = 100,
        batch_size: int = 32,
        mask_type: Optional[str] = None,
        mask_file: Optional[str] = None,
        attack_params: Optional[Dict] = None,
        save_adversarial: bool = True,
    ) -> Dict:
        """
        Run experiment.
        
        Args:
            model_name: Name of model to attack
            attack_name: Name of attack to use
            dataset_path: Path to dataset (dataset.json for mini-imagenet)
            num_samples: Number of samples to attack
            batch_size: Batch size for processing
            mask_file: Path to mask file
            attack_params: Additional attack parameters
            save_adversarial: Whether to save adversarial examples
        
        Returns:
            Results dictionary
        """
        self.logger.info(f"Starting experiment: {attack_name} on {model_name}")
        self.logger.info(f"Dataset: {dataset_path}")
        self.logger.info(f"Samples: {num_samples}, Batch size: {batch_size}")
        self.logger.info(f"Mask type: {mask_type}") # future use: signal for multiple explanations or resp

        # Load dataset
        self.logger.info("Loading dataset...")
        dataset = ImageNetDataset( # assume ImageNetDataset
            Path(dataset_path),
            transform=get_preprocessing(model_name)
        )
        
        if num_samples > len(dataset):
            num_samples = len(dataset)
            self.logger.warning(f"num_samples reduced to {num_samples}")
        
        # Create a sampler for subset
        from torch.utils.data import Subset
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        subset = Subset(dataset, indices)
        dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False)
        
        # Load model
        self.logger.info("Loading model...")
        model = load_model(model_name, self.device)
        
        # Create attack
        self.logger.info("Creating attack...")
        attack_params = attack_params or {}
        attack = get_attack(attack_name, model, **attack_params)
        
        # Get image size for mask
        sample_data = dataset[0]
        img_shape = sample_data['image'].shape
        logging.info(f"Image shape: {img_shape}")
        
        # Get mask
        mask = None
        if mask_file is not None:
            self.logger.info(f"Loading mask from file: {mask_file}")
            mask = load_from_file(mask_file)

        
        # Run attacks
        self.logger.info("Running attacks...")
        results = {
            'metadata': {
                'model': model_name,
                'attack': attack_name,
                'mask_type': mask_type,
                'num_samples': num_samples,
                'timestamp': datetime.now().isoformat(),
            },
            'samples': [],
            'statistics': {}
        }
        
        attack_successes = []
        perturbations_l2 = []
        perturbations_l0 = []
        times = []
        
        # Move model to eval mode
        model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                batch_start = time.time()
                
                images = batch['image'].to(self.device)

                # Get original predictions
                logits_clean = F.softmax(model(images), dim=1)
                preds_score, pred_label = torch.topk(logits_clean, 1)
                
                # Run attack
                if mask is not None:
                    try:
                        images_adv = attack(images, pred_label, mask=mask)
                    except TypeError:
                        raise f"{attack_name} doesn't support mask parameter, running without mask"
                else:
                    images_adv = attack(images, pred_label)
                
                # Get predictions
                logits_adv = F.softmax(model(images_adv), dim=1)
                preds_adv_scores, preds_adv = torch.topk(logits_adv, 1)
                
                batch_time = time.time() - batch_start
                times.append(batch_time)
                
                # Calculate metrics
                batch_success = (pred_label != preds_adv).cpu()
                perturbation = images_adv - images
                batch_l2 = torch.norm(perturbation.view(perturbation.shape[0], -1), p=2, dim=1).cpu()
                batch_l0 = (perturbation.abs() > 1e-6).float().view(perturbation.shape[0], -1).sum(dim=1).cpu()
                
                attack_successes.extend(batch_success.tolist())
                perturbations_l2.extend(batch_l2.tolist())
                perturbations_l0.extend(batch_l0.tolist())
                
                # Save sample results
                for i in range(images.shape[0]):
                    sample_result = {
                        'image_path': batch['image_path'][i],
                        'image_name': batch['image_name'][i],
                        'image_id': batch['image_id'][i].item(),
                        'class_folder': batch['class_folder'][i],
                        'class_name': batch['class_name'][i],
                        'class_id': batch['class_id'][i].item(),
                        'pred_clean': pred_label[i].item(),
                        'pred_adv': preds_adv[i].item(),
                        'attack_success': bool(batch_success[i].item()),
                        'perturbation_l2': float(batch_l2[i].item()),
                        'perturbation_l0': float(batch_l0[i].item()),
                    }
                    results['samples'].append(sample_result)
                
                # Save adversarial examples
                if save_adversarial:
                    adv_dir = self.output_dir / 'adversarial_examples'
                    adv_dir.mkdir(exist_ok=True)
                    for i in range(images.shape[0]):
                        adv_path = adv_dir / f"batch{batch_idx:04d}_sample{i:04d}_{batch['image_name'][i]}.pt"
                        torch.save({
                            'original': images[i].cpu(),
                            'adversarial': images_adv[i].cpu(),
                            'perturbation': perturbation[i].cpu(),
                            'metadata': results['samples'][batch_idx * batch_size + i - 1],
                        }, adv_path)
                
                self.logger.info(
                    f"Batch {batch_idx+1}/{len(dataloader)}: "
                    f"Success={batch_success.float().mean().item():.1%}, "
                    f"L2={batch_l2.mean().item():.4f}, "
                    f"Time={batch_time:.2f}s"
                )

                if batch_idx % 10 == 0:
                    results_path = self.output_dir / f"results_{attack_name}_{model_name}.json"
                    with open(results_path, 'w') as f:
                        json.dump(results, f, indent=2)
        # Calculate statistics
        attack_success_rate = np.mean(attack_successes) if attack_successes else 0.0
        results['statistics'] = {
            'attack_success_rate': float(attack_success_rate),
            'mean_perturbation_l2': float(np.mean(perturbations_l2)) if perturbations_l2 else 0.0,
            'mean_perturbation_l0': float(np.mean(perturbations_l0)) if perturbations_l0 else 0.0,
            'median_perturbation_l2': float(np.median(perturbations_l2)) if perturbations_l2 else 0.0,
            'median_perturbation_l0': float(np.median(perturbations_l0)) if perturbations_l0 else 0.0,
            'total_time': float(sum(times)),
            'average_time_per_sample': float(np.mean(times)) if times else 0.0,
        }
        
        # Save results
        results_path = self.output_dir / f"results_{attack_name}_{model_name}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Experiment completed!")
        self.logger.info(f"Attack Success Rate: {attack_success_rate:.1%}")
        self.logger.info(f"Mean L2 Perturbation: {results['statistics']['mean_perturbation_l2']:.4f}")
        self.logger.info(f"Total Time: {results['statistics']['total_time']:.1f}s")
        self.logger.info(f"Results saved to: {results_path}")
        self.logger.info("=" * 80)
        return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run adversarial attack experiments for L0."
    )
    
    # Dataset
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='./data/miniimagenet/dataset.json',
        help='Path to dataset.json file'
    )
    
    # Model
    parser.add_argument(
        '--model',
        type=str,
        default='resnet152',
        choices=['resnet18', 'resnet50', 'resnet101', 'resnet152',
                 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                 'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large'],
        help='Model architecture'
    )
    
    # Attack
    parser.add_argument(
        '--attack',
        type=str,
        default='onepixel',
        choices=['onepixel', 'pixle'],
        help='Attack type'
    )
    
    # Attack parameters (generic, can be extended)
    parser.add_argument('--pixels', type=int, default=1, help='OnePixel: number of pixels')
    parser.add_argument('--steps', type=int, default=10, help='OnePixel: optimization steps')
    parser.add_argument('--popsize', type=int, default=10, help='OnePixel: population size')

    parser.add_argument(
        '--mask-file',
        type=str,
        default=None,
        help='Path to custom mask file (.pt or .npy)'
    )
    
    # Experiment
    parser.add_argument('--num-samples', type=int, default=100, help='Number of samples to attack')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--output-dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--no-save-adversarial', action='store_true', help='Do not save adversarial examples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create runner
    runner = ExperimentRunner(
        output_dir=args.output_dir,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )

    # Build attack parameters
    attack_params = {}
    if args.attack == 'onepixel':
        attack_params = {
            'pixels': args.pixels,
            'steps': args.steps,
            'popsize': args.popsize,
        }
    elif args.attack == 'pixle':
        attack_params = {
            'restarts': 10, # should add more params...
        }

    # Run experiment
    results = runner.run(
        model_name=args.model,
        attack_name=args.attack,
        dataset_path=args.dataset_path,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        mask_file=args.mask_file,
        attack_params=attack_params,
        save_adversarial=not args.no_save_adversarial,
    )
