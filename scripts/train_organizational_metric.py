#!/usr/bin/env python3
"""
Train Organizational Metric

Learns a metric space where distance reflects organizational proximity
in Pearltrees hierarchy, using path-weighted MSE loss.

Usage:
    python scripts/train_organizational_metric.py --epochs 50
"""

import argparse
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import random
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Hierarchy Parsing
# =============================================================================

def parse_hierarchy_path(target_text: str) -> List[str]:
    """
    Parse target_text to extract hierarchy path.

    Example target_text:
    "/10311468/11820376/.../2492215
    - s243a
      - root node
        - Society
          - groups
            - Hacktivism"

    Returns: ['s243a', 'root node', 'Society', 'groups', 'Hacktivism']
    """
    lines = target_text.strip().split('\n')
    path = []

    for line in lines[1:]:  # Skip first line (IDs)
        # Count leading spaces to determine depth
        stripped = line.lstrip('- ')
        if stripped:
            path.append(stripped.strip())

    return path


def compute_tree_distance(path_a: List[str], path_b: List[str]) -> int:
    """
    Compute tree distance (hops) between two items.

    Distance = steps to common ancestor + steps from ancestor to other node
    """
    # Find common prefix length
    common_len = 0
    for i in range(min(len(path_a), len(path_b))):
        if path_a[i] == path_b[i]:
            common_len = i + 1
        else:
            break

    # Distance = (depth_a - common) + (depth_b - common)
    dist = (len(path_a) - common_len) + (len(path_b) - common_len)
    return dist


def load_hierarchy_data(
    jsonl_path: str,
    root_account: str = 's243a'
) -> Dict[str, List[str]]:
    """
    Load hierarchy paths from JSONL, filtering to items traceable to root.

    Returns: {tree_id: path_list}
    """
    id_to_path = {}
    skipped = 0

    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)

            # Filter to specified account
            if rec.get('account') != root_account:
                continue

            tree_id = rec.get('tree_id')
            target_text = rec.get('target_text', '')

            if not tree_id or not target_text:
                skipped += 1
                continue

            path = parse_hierarchy_path(target_text)

            # Verify path starts with root account
            if not path or path[0] != root_account:
                skipped += 1
                continue

            id_to_path[tree_id] = path

    print(f"Loaded {len(id_to_path)} items with valid paths (skipped {skipped})")
    return id_to_path


# =============================================================================
# Dataset
# =============================================================================

class OrganizationalPairDataset(Dataset):
    """
    Dataset of item pairs with tree distances.
    """

    def __init__(
        self,
        input_embeddings: np.ndarray,
        output_embeddings: np.ndarray,
        weights: np.ndarray,
        item_ids: List[str],
        id_to_path: Dict[str, List[str]],
        num_pairs: int = 100000,
        max_distance: int = 20
    ):
        self.input_emb = torch.from_numpy(input_embeddings).float()
        self.output_emb = torch.from_numpy(output_embeddings).float()
        self.weights = torch.from_numpy(weights).float()

        # Build index mapping
        self.id_to_idx = {id_: i for i, id_ in enumerate(item_ids)}

        # Filter to items with hierarchy paths
        self.valid_ids = [id_ for id_ in item_ids if id_ in id_to_path]
        self.id_to_path = id_to_path

        print(f"Valid items with paths: {len(self.valid_ids)}/{len(item_ids)}")

        # Pre-generate pairs
        self.pairs = self._generate_pairs(num_pairs, max_distance)
        print(f"Generated {len(self.pairs)} training pairs")

    def _generate_pairs(
        self,
        num_pairs: int,
        max_distance: int
    ) -> List[Tuple[int, int, int]]:
        """Generate (idx_a, idx_b, distance) pairs."""
        pairs = []

        # Sample pairs with various distances
        for _ in range(num_pairs):
            # Random sample two items
            id_a, id_b = random.sample(self.valid_ids, 2)

            path_a = self.id_to_path[id_a]
            path_b = self.id_to_path[id_b]

            dist = compute_tree_distance(path_a, path_b)
            dist = min(dist, max_distance)  # Cap distance

            idx_a = self.id_to_idx[id_a]
            idx_b = self.id_to_idx[id_b]

            pairs.append((idx_a, idx_b, dist))

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        idx_a, idx_b, dist = self.pairs[idx]

        return {
            'input_a': self.input_emb[idx_a],
            'output_a': self.output_emb[idx_a],
            'weights_a': self.weights[idx_a],
            'input_b': self.input_emb[idx_b],
            'output_b': self.output_emb[idx_b],
            'weights_b': self.weights[idx_b],
            'distance': torch.tensor(dist, dtype=torch.float32)
        }


# =============================================================================
# Model
# =============================================================================

class OrganizationalMetric(nn.Module):
    """
    Learns a metric space where Euclidean distance = organizational proximity.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        weight_dim: int = 64,
        hidden_dim: int = 256,
        output_dim: int = 64,
        weight_power: float = 2.0  # Square weights to sharpen
    ):
        super().__init__()

        self.weight_power = weight_power

        # Input: input_emb + output_emb + weights + entropy = 768 + 768 + 64 + 1
        input_dim = embed_dim * 2 + weight_dim + 1

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def compute_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute entropy of probability distribution."""
        # probs: (batch, n_basis) - already softmax'd
        p = probs + 1e-8
        entropy = -(p * torch.log(p)).sum(dim=-1, keepdim=True)
        return entropy

    def encode(
        self,
        input_emb: torch.Tensor,
        output_emb: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode item features into organizational metric space.

        Returns: (batch, output_dim)
        """
        # Convert raw weights to probabilities via softmax
        probs = torch.softmax(weights, dim=-1)

        # Sharpen probabilities
        w = probs ** self.weight_power
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)

        # Compute entropy (on original probs, not sharpened)
        entropy = self.compute_entropy(probs)

        # Concatenate features
        x = torch.cat([input_emb, output_emb, w, entropy], dim=-1)

        return self.encoder(x)

    def forward(
        self,
        input_a: torch.Tensor,
        output_a: torch.Tensor,
        weights_a: torch.Tensor,
        input_b: torch.Tensor,
        output_b: torch.Tensor,
        weights_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute predicted distance between two items.

        Returns: (batch,) predicted distances
        """
        enc_a = self.encode(input_a, output_a, weights_a)
        enc_b = self.encode(input_b, output_b, weights_b)

        # Euclidean distance
        dist = torch.norm(enc_a - enc_b, dim=-1)

        return dist


# =============================================================================
# Training
# =============================================================================

def path_weighted_mse(
    predicted: torch.Tensor,
    actual: torch.Tensor,
    alpha: float = 1.0
) -> torch.Tensor:
    """
    Path-weighted MSE loss.

    Shorter paths (more certain) are weighted more heavily.

    weight = 1 / (1 + alpha * actual_distance)
    """
    weight = 1.0 / (1.0 + alpha * actual)
    loss = weight * (predicted - actual) ** 2
    return loss.mean()


def train_epoch(
    model: OrganizationalMetric,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alpha: float = 1.0
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        # Move to device
        input_a = batch['input_a'].to(device)
        output_a = batch['output_a'].to(device)
        weights_a = batch['weights_a'].to(device)
        input_b = batch['input_b'].to(device)
        output_b = batch['output_b'].to(device)
        weights_b = batch['weights_b'].to(device)
        actual_dist = batch['distance'].to(device)

        # Forward
        predicted = model(input_a, output_a, weights_a, input_b, output_b, weights_b)

        # Loss
        loss = path_weighted_mse(predicted, actual_dist, alpha)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(
    model: OrganizationalMetric,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()

    all_predicted = []
    all_actual = []

    with torch.no_grad():
        for batch in dataloader:
            input_a = batch['input_a'].to(device)
            output_a = batch['output_a'].to(device)
            weights_a = batch['weights_a'].to(device)
            input_b = batch['input_b'].to(device)
            output_b = batch['output_b'].to(device)
            weights_b = batch['weights_b'].to(device)
            actual_dist = batch['distance'].to(device)

            predicted = model(input_a, output_a, weights_a, input_b, output_b, weights_b)

            all_predicted.extend(predicted.cpu().numpy())
            all_actual.extend(actual_dist.cpu().numpy())

    predicted = np.array(all_predicted)
    actual = np.array(all_actual)

    # Metrics
    mse = ((predicted - actual) ** 2).mean()
    mae = np.abs(predicted - actual).mean()

    # Correlation
    corr = np.corrcoef(predicted, actual)[0, 1]

    # Per-distance bucket accuracy
    buckets = {}
    for d in range(0, 11):
        mask = (actual >= d) & (actual < d + 1)
        if mask.sum() > 0:
            bucket_mae = np.abs(predicted[mask] - actual[mask]).mean()
            buckets[f'mae_dist_{d}'] = bucket_mae

    return {
        'mse': mse,
        'mae': mae,
        'correlation': corr,
        **buckets
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Organizational Metric')
    parser.add_argument('--jsonl', default='reports/pearltrees_targets_full_multi_account.jsonl',
                        help='Path to JSONL with hierarchy data')
    parser.add_argument('--model-dir', default='models/pearltrees_federated_nomic',
                        help='Path to federated model directory')
    parser.add_argument('--transformer', default='models/bivector_paired.pt',
                        help='Path to transformer model for weights')
    parser.add_argument('--account', default='s243a',
                        help='Root account to filter to')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-pairs', type=int, default=100000)
    parser.add_argument('--output', default='models/organizational_metric.pt')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load hierarchy data
    print("\nLoading hierarchy data...")
    id_to_path = load_hierarchy_data(args.jsonl, args.account)

    # Load federated model for IDs
    print("\nLoading federated model...")
    with open(f'{args.model_dir}.pkl', 'rb') as f:
        fed_model = pickle.load(f)

    item_ids = [str(id_) for id_ in fed_model['global_target_ids']]
    print(f"Total items: {len(item_ids)}")

    # Load routing data
    print("\nLoading routing data...")
    routing = np.load(f'{args.model_dir}/routing_data.npz', allow_pickle=True)
    input_embeddings = routing['query_embeddings']
    output_embeddings = routing['target_embeddings']

    # Load transformer and compute weights
    print("\nLoading transformer and computing weights...")
    from scripts.train_orthogonal_codebook import load_composed_bivector_transformer

    transformer = load_composed_bivector_transformer(args.transformer)
    transformer.eval_mode()

    # Compute weights for all items
    batch_size = 256
    all_weights = []

    with torch.no_grad():
        for i in range(0, len(input_embeddings), batch_size):
            batch = input_embeddings[i:i+batch_size]
            batch_tensor = torch.from_numpy(batch.astype(np.float32)).to(device)
            _, weights, _, _ = transformer.forward(batch_tensor)
            all_weights.append(weights.cpu().numpy())

    weights = np.concatenate(all_weights, axis=0)
    print(f"Computed weights: {weights.shape}")

    # Create dataset
    print("\nCreating dataset...")
    dataset = OrganizationalPairDataset(
        input_embeddings=input_embeddings,
        output_embeddings=output_embeddings,
        weights=weights,
        item_ids=item_ids,
        id_to_path=id_to_path,
        num_pairs=args.num_pairs
    )

    # Split train/test
    n_test = min(10000, len(dataset) // 10)
    n_train = len(dataset) - n_test
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_test]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Create model
    print("\nCreating model...")
    model = OrganizationalMetric(
        embed_dim=768,
        weight_dim=weights.shape[1],
        hidden_dim=256,
        output_dim=64
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print("\nTraining...")
    best_loss = float('inf')

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            metrics = evaluate(model, test_loader, device)
            print(f"Epoch {epoch+1}/{args.epochs}: loss={train_loss:.4f}, "
                  f"test_mae={metrics['mae']:.3f}, corr={metrics['correlation']:.3f}")

            if train_loss < best_loss:
                best_loss = train_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'embed_dim': 768,
                    'weight_dim': weights.shape[1],
                    'hidden_dim': 256,
                    'output_dim': 64,
                    'metrics': metrics
                }, args.output)

    # Final evaluation
    print("\nFinal evaluation...")
    metrics = evaluate(model, test_loader, device)
    print(f"Test MSE: {metrics['mse']:.4f}")
    print(f"Test MAE: {metrics['mae']:.4f}")
    print(f"Correlation: {metrics['correlation']:.4f}")

    print("\nPer-distance MAE:")
    for k, v in sorted(metrics.items()):
        if k.startswith('mae_dist'):
            print(f"  {k}: {v:.3f}")

    print(f"\nSaved model to {args.output}")


if __name__ == '__main__':
    main()
