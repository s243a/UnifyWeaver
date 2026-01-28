#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
"""
Model Registry - Discovers and manages semantic search models.

Usage:
    from unifyweaver.config.model_registry import ModelRegistry

    registry = ModelRegistry()
    model = registry.get_model('pearltrees_federated_nomic')

    # Or auto-select for task
    models = registry.get_for_task('bookmark_filing')
    print(models['projection'].name)

    # Load a model for inference
    engine = registry.load_model('pearltrees_federated_nomic')

    # Load best available model for project knowledge
    engine = registry.load_knowledge_model(prefer_newer=True)

    # Show training command
    cmd = registry.get_training_command('pearltrees_federated_nomic')
    print(cmd)
"""

import os
import sys
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
import logging

logger = logging.getLogger(__name__)


def _load_yaml():
    """Lazy import yaml to avoid dependency at import time."""
    try:
        import yaml
        return yaml
    except ImportError:
        logger.warning("PyYAML not installed. Install with: pip install pyyaml")
        return None


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""
    name: str
    type: str  # embedding, federated, orthogonal_codebook
    description: str = ""
    dimensions: int = 768
    embedding_model: Optional[str] = None
    features: List[str] = field(default_factory=list)
    prefixes: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    source: Dict[str, str] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    recommended_for: List[str] = field(default_factory=list)
    accounts: List[str] = field(default_factory=list)
    notes: str = ""
    # Filtering attributes
    scope: str = ""  # multi_account, single_account, domain, holdout
    tags: List[str] = field(default_factory=list)  # alternative, experimental, etc.
    domain: str = ""  # physics, etc. for domain-specific models

    def get_path(self, search_paths: List[Path], project_root: Path) -> Optional[Path]:
        """Resolve model path from search paths."""
        local = self.source.get('local')
        if not local:
            return None

        # Try each search path
        for base in search_paths:
            if not base.is_absolute():
                base = project_root / base
            path = base / local if not local.startswith(str(base)) else Path(local)

            # Handle relative paths
            if not path.is_absolute():
                path = project_root / path

            if path.exists():
                return path

        # Try local path directly relative to project
        direct_path = project_root / local
        if direct_path.exists():
            return direct_path

        # Try private fallback
        fallback = self.source.get('private_fallback')
        if fallback:
            fallback_expanded = os.path.expandvars(os.path.expanduser(fallback))
            fallback_path = Path(fallback_expanded)
            if fallback_path.exists():
                return fallback_path

        return None

    def get_huggingface_id(self) -> Optional[str]:
        """Get HuggingFace model ID if available."""
        return self.source.get('huggingface')


class ModelRegistry:
    """Central registry for semantic search models."""

    # User overrides (loaded on top of defaults)
    USER_CONFIG_ORDER = [
        '.UnifyWeaver-models.yaml',      # Root, user private
        '.local/models.yaml',             # .local directory
        'config/model_registry.yaml',     # User's registry (gitignored)
    ]

    # Project defaults (fallback)
    PROJECT_DEFAULTS = [
        'config/model_registry_defaults.yaml',  # Public model definitions
        'config/model_defaults.yaml',            # Task defaults
    ]

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or self._find_project_root()
        self._registry: Dict[str, ModelMetadata] = {}
        self._defaults: Dict[str, Dict] = {}
        self._search_paths: List[Path] = []
        self._inference_settings: Dict[str, Any] = {}

        self._load_project_defaults()
        self._load_user_configs()
        self._discover_local_models()

    @staticmethod
    def _find_project_root() -> Path:
        """Find project root by looking for markers."""
        markers = ['.git', 'src', 'scripts', 'config']
        current = Path.cwd()

        for parent in [current] + list(current.parents):
            if all((parent / m).exists() for m in ['src', 'scripts']):
                return parent
            if (parent / '.git').exists():
                return parent

        # Fallback: check if we're in the project
        if (current / 'src' / 'unifyweaver').exists():
            return current

        return current

    def _load_yaml_file(self, path: Path) -> Optional[Dict]:
        """Load a YAML file if it exists."""
        yaml = _load_yaml()
        if not yaml:
            return None

        if not path.exists():
            return None

        try:
            with open(path) as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            return None

    def _load_project_defaults(self):
        """Load project default configurations."""
        for config_file in self.PROJECT_DEFAULTS:
            path = self.project_root / config_file
            data = self._load_yaml_file(path)
            if data:
                self._parse_config(data, is_defaults=True)

    def _load_user_configs(self):
        """Load user configurations (override defaults)."""
        for config_file in self.USER_CONFIG_ORDER:
            path = self.project_root / config_file
            data = self._load_yaml_file(path)
            if data:
                self._parse_config(data, is_defaults=False)

    def _parse_config(self, data: Dict, is_defaults: bool = True):
        """Parse configuration data into registry."""
        # Parse model definitions
        for category in ['embedding_models', 'projection_models', 'transformer_models']:
            models = data.get(category, {})
            for name, info in models.items():
                if isinstance(info, dict):
                    self._registry[name] = ModelMetadata(
                        name=name,
                        type=info.get('type', 'unknown'),
                        description=info.get('description', ''),
                        dimensions=info.get('dimensions', 768),
                        embedding_model=info.get('embedding_model'),
                        features=info.get('features', []),
                        prefixes=info.get('prefixes', {}),
                        metrics=info.get('metrics', {}),
                        source=info.get('source', {}),
                        training=info.get('training', {}),
                        recommended_for=info.get('recommended_for', []),
                        accounts=info.get('accounts', []),
                        notes=info.get('notes', ''),
                        scope=info.get('scope', ''),
                        tags=info.get('tags', []),
                        domain=info.get('domain', ''),
                    )

        # Parse task defaults
        defaults = data.get('defaults', {})
        for task, config in defaults.items():
            if task not in self._defaults:
                self._defaults[task] = {}
            self._defaults[task].update(config)

        # Parse overrides (user config)
        overrides = data.get('overrides', {})
        for task, config in overrides.items():
            if task not in self._defaults:
                self._defaults[task] = {}
            self._defaults[task].update(config)

        # Parse search paths
        search_paths = data.get('search_paths', [])
        for path_str in search_paths:
            expanded = os.path.expandvars(os.path.expanduser(path_str))
            path = Path(expanded)
            if path not in self._search_paths:
                self._search_paths.append(path)

        # Parse inference settings
        inference = data.get('inference', {})
        self._inference_settings.update(inference)

        # Parse use modes (top-level in config)
        use_modes = data.get('use_modes', {})
        if use_modes:
            if 'use_modes' not in self._inference_settings:
                self._inference_settings['use_modes'] = {}
            self._inference_settings['use_modes'].update(use_modes)

        # Parse task modes (top-level in config)
        task_modes = data.get('task_modes', {})
        if task_modes:
            if 'task_modes' not in self._inference_settings:
                self._inference_settings['task_modes'] = {}
            self._inference_settings['task_modes'].update(task_modes)

        # Parse knowledge models (top-level in config)
        knowledge_models = data.get('knowledge_models', {})
        if knowledge_models:
            self._inference_settings['knowledge_models'] = knowledge_models

        # Parse environments
        environments = data.get('environments', {})
        if environments:
            if 'environments' not in self._inference_settings:
                self._inference_settings['environments'] = {}
            self._inference_settings['environments'].update(environments)

        # Parse environment selection rules
        env_selection = data.get('environment_selection', {})
        if env_selection:
            self._inference_settings['environment_selection'] = env_selection

        # Parse inference permissions
        permissions = data.get('inference_permissions', {})
        if permissions:
            self._inference_settings['inference_permissions'] = permissions

    def _discover_local_models(self):
        """Auto-discover models not in registry."""
        models_dir = self.project_root / 'models'
        if not models_dir.exists():
            return

        for path in models_dir.glob('*.pkl'):
            name = path.stem
            if name not in self._registry:
                # Infer type from naming convention
                if 'federated' in name:
                    model_type = 'federated'
                elif 'transformer' in name:
                    model_type = 'orthogonal_codebook'
                else:
                    model_type = 'projection'

                self._registry[name] = ModelMetadata(
                    name=name,
                    type=model_type,
                    source={'local': f'models/{path.name}'},
                    description=f'Auto-discovered: {path.name}'
                )

        # Also discover .pt files (transformers)
        for path in models_dir.glob('*.pt'):
            name = path.stem
            if name not in self._registry:
                self._registry[name] = ModelMetadata(
                    name=name,
                    type='orthogonal_codebook',
                    source={'local': f'models/{path.name}'},
                    description=f'Auto-discovered: {path.name}'
                )

    def get_model(self, name: str) -> Optional[ModelMetadata]:
        """Get model metadata by name."""
        return self._registry.get(name)

    def get_for_task(self, task: str) -> Dict[str, ModelMetadata]:
        """Get recommended models for a task."""
        task_config = self._defaults.get(task, {})
        result = {}

        for role in ['projection', 'embedding', 'transformer']:
            model_name = task_config.get(role)
            if model_name and model_name in self._registry:
                result[role] = self._registry[model_name]

        return result

    def get_model_path(self, name: str) -> Optional[Path]:
        """Get resolved path to a model file."""
        model = self.get_model(name)
        if not model:
            return None
        return model.get_path(self._search_paths, self.project_root)

    def list_models(
        self,
        model_type: Optional[str] = None,
        embedding_model: Optional[str] = None,
        scope: Optional[str] = None,
        domain: Optional[str] = None,
        tags: Optional[List[str]] = None,
        exclude_tags: Optional[List[str]] = None,
        available_only: bool = False,
    ) -> List[ModelMetadata]:
        """
        List registered models with optional filtering.

        Args:
            model_type: Filter by type (embedding, federated, orthogonal_codebook)
            embedding_model: Filter by underlying embedding model name
            scope: Filter by scope (multi_account, single_account, domain, holdout)
            domain: Filter by domain (physics, etc.)
            tags: Include only models with ALL of these tags
            exclude_tags: Exclude models with ANY of these tags
            available_only: Only return models that are available (file exists)

        Returns:
            List of matching ModelMetadata objects
        """
        models = list(self._registry.values())

        if model_type:
            models = [m for m in models if m.type == model_type]

        if embedding_model:
            # Match partial names (e.g., "minilm" matches "all-MiniLM-L6-v2")
            embedding_lower = embedding_model.lower()
            models = [m for m in models if m.embedding_model and
                      embedding_lower in m.embedding_model.lower()]

        if scope:
            models = [m for m in models if m.scope == scope]

        if domain:
            models = [m for m in models if m.domain == domain]

        if tags:
            # Model must have ALL specified tags
            models = [m for m in models if all(t in m.tags for t in tags)]

        if exclude_tags:
            # Exclude models with ANY of these tags
            models = [m for m in models if not any(t in m.tags for t in exclude_tags)]

        if available_only:
            models = [m for m in models if self.check_model_available(m.name)]

        return models

    def get_training_command(self, name: str) -> Optional[str]:
        """Get command to regenerate a model."""
        model = self.get_model(name)
        if not model or not model.training:
            return None

        training = model.training
        script = training.get('script')
        data = training.get('data')
        args = training.get('args', {})
        federated_models = training.get('federated_models', [])

        if not script:
            return None

        cmd_parts = [f'python3 {script}']

        # Add data path if not multi-source
        if data and not federated_models:
            cmd_parts.append(data)
            # Add output path
            local_path = model.source.get('local', f'models/{name}.pkl')
            cmd_parts.append(local_path)

        # Add federated models for multi-source training
        if federated_models:
            cmd_parts.append('--federated-models')
            for fm in federated_models:
                fm_model = self.get_model(fm)
                if fm_model:
                    fm_path = fm_model.source.get('local', f'models/{fm}.pkl')
                    cmd_parts.append(fm_path)

        # Add other arguments
        for key, val in args.items():
            flag = f'--{key.replace("_", "-")}'
            if isinstance(val, bool):
                if val:
                    cmd_parts.append(flag)
            else:
                cmd_parts.append(f'{flag} {val}')

        return ' \\\n  '.join(cmd_parts)

    def get_inference_setting(self, key: str, default: Any = None) -> Any:
        """Get an inference setting."""
        return self._inference_settings.get(key, default)

    def get_search_paths(self) -> List[Path]:
        """Get model search paths."""
        return self._search_paths.copy()

    def check_model_available(self, name: str) -> bool:
        """Check if a model is available (file exists or cached)."""
        status = self.get_model_availability(name)
        return status.get('available', False)

    def get_model_availability(self, name: str) -> Dict[str, Any]:
        """
        Get detailed availability info for a model.

        Returns dict with:
            available: bool - whether model can be loaded
            local_path: Path or None - path if found locally
            huggingface_cached: bool - if HF model is cached
            huggingface_id: str or None - HF model ID if applicable
            generated_path: Path or None - if found in search paths by name
        """
        result = {
            'available': False,
            'local_path': None,
            'huggingface_cached': False,
            'huggingface_id': None,
            'generated_path': None,
        }

        model = self.get_model(name)

        # Check configured local path
        path = self.get_model_path(name)
        if path is not None and path.exists():
            result['available'] = True
            result['local_path'] = path

        # Check search paths for model by name (even if not explicitly configured)
        for search_path in self._search_paths:
            if not search_path.is_absolute():
                search_path = self.project_root / search_path
            if not search_path.exists():
                continue

            # Check common extensions
            for ext in ['.pkl', '.pt', '.bin', '']:
                candidate = search_path / f'{name}{ext}'
                if candidate.exists():
                    result['generated_path'] = candidate
                    result['available'] = True
                    break

            # Also check without hyphens/dots (nomic-embed-text-v1.5 -> nomic_embed_text_v1_5)
            safe_name = name.replace('-', '_').replace('.', '_')
            for ext in ['.pkl', '.pt', '.bin']:
                candidate = search_path / f'{safe_name}{ext}'
                if candidate.exists():
                    result['generated_path'] = candidate
                    result['available'] = True
                    break

        # Check HuggingFace cache for embedding models
        if model and model.type == 'embedding':
            hf_id = model.get_huggingface_id()
            if hf_id:
                result['huggingface_id'] = hf_id
                if self._check_huggingface_cached(hf_id):
                    result['huggingface_cached'] = True
                    result['available'] = True

        return result

    def _check_huggingface_cached(self, model_id: str) -> bool:
        """Check if a HuggingFace model is cached locally."""
        # HuggingFace cache locations
        cache_dirs = [
            Path.home() / '.cache' / 'huggingface' / 'hub',
            Path.home() / '.cache' / 'torch' / 'sentence_transformers',
        ]

        # Model ID formats: "org/model" -> "models--org--model" or "org_model"
        org_model = model_id.replace('/', '--')
        org_model_underscore = model_id.replace('/', '_')

        for cache_dir in cache_dirs:
            if not cache_dir.exists():
                continue

            # Check for HuggingFace hub format (models--org--model)
            hf_path = cache_dir / f'models--{org_model}'
            if hf_path.exists():
                return True

            # Check for sentence-transformers format (org_model)
            st_path = cache_dir / org_model_underscore
            if st_path.exists():
                return True

        return False

    def get_model_timestamp(self, name: str) -> Optional[datetime]:
        """Get model file modification timestamp."""
        path = self.get_model_path(name)
        if path and path.exists():
            return datetime.fromtimestamp(path.stat().st_mtime)
        return None

    def get_model_age_days(self, name: str) -> Optional[float]:
        """Get model age in days."""
        ts = self.get_model_timestamp(name)
        if ts:
            return (datetime.now() - ts).total_seconds() / 86400
        return None

    def load_model(self, name: str) -> Any:
        """
        Load a model by name.

        Returns the appropriate inference engine based on model type:
        - federated: FederatedInferenceEngine
        - orthogonal_codebook: FastOrthogonalTransformer
        - embedding: SentenceTransformer

        Raises FileNotFoundError if model not available.
        """
        model_meta = self.get_model(name)
        if not model_meta:
            raise ValueError(f"Unknown model: {name}")

        path = self.get_model_path(name)
        if not path or not path.exists():
            # Check if it's a HuggingFace model
            hf_id = model_meta.get_huggingface_id()
            if hf_id:
                return self._load_huggingface_model(hf_id)
            raise FileNotFoundError(f"Model not found: {name}")

        if model_meta.type == 'federated':
            return self._load_federated_model(path)
        elif model_meta.type == 'orthogonal_codebook':
            return self._load_transformer_model(path)
        elif model_meta.type == 'embedding':
            hf_id = model_meta.get_huggingface_id()
            if hf_id:
                return self._load_huggingface_model(hf_id)
            raise ValueError(f"Embedding model {name} requires HuggingFace ID")
        else:
            # Generic pickle load
            with open(path, 'rb') as f:
                return pickle.load(f)

    def _load_federated_model(self, path: Path) -> Any:
        """Load a federated inference engine."""
        # Lazy import to avoid circular dependencies
        try:
            sys.path.insert(0, str(self.project_root / 'scripts'))
            from infer_pearltrees_federated import FederatedInferenceEngine
            return FederatedInferenceEngine(str(path))
        except ImportError:
            logger.warning("FederatedInferenceEngine not available, loading raw pickle")
            with open(path, 'rb') as f:
                return pickle.load(f)

    def _load_transformer_model(self, path: Path) -> Any:
        """Load an orthogonal transformer model."""
        try:
            sys.path.insert(0, str(self.project_root / 'scripts'))
            from train_orthogonal_codebook import load_orthogonal_transformer
            return load_orthogonal_transformer(str(path))
        except ImportError:
            logger.warning("load_orthogonal_transformer not available")
            # Try torch load as fallback
            try:
                import torch
                return torch.load(path, map_location='cpu')
            except ImportError:
                raise ImportError("PyTorch required to load transformer models")

    def _load_huggingface_model(self, model_id: str) -> Any:
        """Load a HuggingFace embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(model_id, trust_remote_code=True)
        except ImportError:
            raise ImportError("sentence-transformers required: pip install sentence-transformers")

    def load_knowledge_model(
        self,
        prefer_newer: bool = None,
        prefer_federated: bool = None,
        prefer_transformer: bool = None,
        task: str = 'skills_qa',
    ) -> Tuple[Any, str]:
        """
        Load the best available model for project knowledge.

        Considers multiple models (skills_qa, books_qa, combined) and selects
        based on preferences and availability.

        Args:
            prefer_newer: Prefer more recently trained models (overrides config)
            prefer_federated: Prefer federated over transformer (overrides config)
            prefer_transformer: Prefer transformer over federated (overrides config)
            task: Task to get models for (default: skills_qa)

        Returns:
            Tuple of (loaded_model, model_name)
        """
        # Get preference settings from config, allow overrides
        prefs = self._inference_settings.get('preferences', {})
        if prefer_newer is None:
            prefer_newer = prefs.get('prefer_newer', False)
        if prefer_federated is None:
            prefer_federated = prefs.get('prefer_federated', False)
        if prefer_transformer is None:
            prefer_transformer = prefs.get('prefer_transformer', False)

        # Knowledge model candidates in priority order
        knowledge_models = [
            'orthogonal_transformer_multisource',  # Combined skills+books transformer
            'skills_qa_federated',                  # Skills-only federated
            'books_qa_federated_nomic',             # Books-only federated
        ]

        # Filter to available models
        available = []
        for name in knowledge_models:
            if self.check_model_available(name):
                meta = self.get_model(name)
                ts = self.get_model_timestamp(name)
                available.append((name, meta, ts))

        if not available:
            # Try task-based fallback
            task_models = self.get_for_task(task)
            if 'projection' in task_models:
                name = task_models['projection'].name
                if self.check_model_available(name):
                    return self.load_model(name), name
            raise FileNotFoundError(f"No knowledge models available for task: {task}")

        # Sort by preferences
        def sort_key(item):
            name, meta, ts = item
            score = 0

            # Type preference
            if prefer_transformer and meta.type == 'orthogonal_codebook':
                score += 100
            elif prefer_federated and meta.type == 'federated':
                score += 100

            # Newer preference
            if prefer_newer and ts:
                # More recent = higher score (days ago as negative)
                age_days = (datetime.now() - ts).total_seconds() / 86400
                score -= age_days

            # Multi-source models get bonus (more knowledge)
            if 'multisource' in name or 'combined' in name:
                score += 50

            return -score  # Negative for descending sort

        available.sort(key=sort_key)
        best_name = available[0][0]

        logger.info(f"Selected knowledge model: {best_name}")
        return self.load_model(best_name), best_name

    def get_missing_models(self, task: str) -> List[str]:
        """Get list of missing models required for a task."""
        models = self.get_for_task(task)
        missing = []
        for role, model in models.items():
            if not self.check_model_available(model.name):
                missing.append(model.name)
        return missing

    # =========================================================================
    # Use Modes - Agent, Loop, Inference
    # =========================================================================

    def get_use_modes(self) -> Dict[str, Dict]:
        """Get available use modes."""
        return self._inference_settings.get('use_modes', {})

    def get_mode_for_task(self, task: str, variant: str = 'recommended') -> Optional[str]:
        """
        Get recommended mode for a task.

        Args:
            task: Task name (e.g., 'bookmark_filing', 'skills_qa')
            variant: Which variant ('recommended', 'interactive', 'batch', 'fast')

        Returns:
            Mode name ('agent', 'loop', 'inference') or None
        """
        task_modes = self._inference_settings.get('task_modes', {})
        task_config = task_modes.get(task, {})
        return task_config.get(variant) or task_config.get('recommended')

    def get_mode_script(self, mode: str) -> Optional[Path]:
        """Get script path for a use mode."""
        modes = self.get_use_modes()
        mode_config = modes.get(mode, {})
        script = mode_config.get('script') or mode_config.get('launcher')
        if script:
            return self.project_root / script
        return None

    def get_mode_command(
        self,
        mode: str,
        task: str = None,
        model: str = None,
        query: str = None,
        **kwargs
    ) -> List[str]:
        """
        Build command to run a use mode.

        Args:
            mode: 'agent', 'loop', or 'inference'
            task: Task name for model selection
            model: Override model name
            query: Query text (for inference/loop modes)
            **kwargs: Additional arguments

        Returns:
            Command as list of strings
        """
        script = self.get_mode_script(mode)
        if not script:
            raise ValueError(f"Unknown mode: {mode}")

        cmd = [sys.executable, str(script)]

        # Get model path
        if model:
            model_path = self.get_model_path(model)
        elif task:
            task_models = self.get_for_task(task)
            if 'projection' in task_models:
                model_path = self.get_model_path(task_models['projection'].name)
            else:
                model_path = None
        else:
            model_path = None

        if model_path:
            cmd.extend(['--model', str(model_path)])

        if query and mode != 'agent':
            cmd.extend(['--query', query])

        # Add extra arguments
        for key, val in kwargs.items():
            flag = f'--{key.replace("_", "-")}'
            if isinstance(val, bool):
                if val:
                    cmd.append(flag)
            elif val is not None:
                cmd.extend([flag, str(val)])

        return cmd

    def run_mode(
        self,
        mode: str,
        task: str = None,
        model: str = None,
        query: str = None,
        interactive: bool = False,
        **kwargs
    ) -> Union[str, int]:
        """
        Run a use mode.

        Args:
            mode: 'agent', 'loop', or 'inference'
            task: Task name for model selection
            model: Override model name
            query: Query text
            interactive: If True, run interactively (for agent mode)
            **kwargs: Additional arguments

        Returns:
            Output string (for inference/loop) or exit code (for agent)
        """
        import subprocess

        cmd = self.get_mode_command(mode, task, model, query, **kwargs)

        if mode == 'agent' or interactive:
            # Run interactively
            result = subprocess.run(cmd)
            return result.returncode
        else:
            # Capture output
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Mode {mode} failed: {result.stderr}")
            return result.stdout

    def create_inference_loop(
        self,
        task: str = None,
        model: str = None,
        embedder: str = None,
    ):
        """
        Create a raw inference loop for programmatic use.

        Returns an inference engine that can be called in a loop.

        Args:
            task: Task name for model selection
            model: Override model name
            embedder: Override embedding model

        Returns:
            FederatedInferenceEngine or similar

        Example:
            engine = registry.create_inference_loop(task='bookmark_filing')
            for bookmark in bookmarks:
                results = engine.infer(bookmark.title, top_k=10)
                # process results
        """
        if model:
            model_name = model
        elif task:
            task_models = self.get_for_task(task)
            if 'projection' in task_models:
                model_name = task_models['projection'].name
            else:
                raise ValueError(f"No projection model for task: {task}")
        else:
            # Use first available knowledge model
            for candidate in ['pearltrees_federated_nomic', 'skills_qa_federated']:
                if self.check_model_available(candidate):
                    model_name = candidate
                    break
            else:
                raise FileNotFoundError("No projection model available")

        return self.load_model(model_name)

    # =========================================================================
    # Environment Management
    # =========================================================================

    def get_environments(self) -> Dict[str, Dict]:
        """Get all configured environments."""
        return self._inference_settings.get('environments', {})

    def get_environment(self, name: str) -> Optional[Dict]:
        """Get configuration for a specific environment."""
        return self.get_environments().get(name)

    def check_environment_available(self, name: str) -> Dict[str, Any]:
        """
        Check if an environment exists and get its status.

        Returns dict with:
            available: bool
            path: resolved path (if venv)
            python_path: path to python executable
            type: 'venv', 'conda', or 'system'
            reason: explanation if not available
        """
        env = self.get_environment(name)
        if not env:
            return {'available': False, 'reason': f'Environment {name} not configured'}

        env_type = env.get('type', 'venv')
        result = {
            'available': False,
            'type': env_type,
            'config': env,
        }

        if env_type == 'system':
            # System Python is always "available"
            import sys
            result['available'] = True
            result['python_path'] = sys.executable
            result['python_version'] = f"{sys.version_info.major}.{sys.version_info.minor}"
            return result

        elif env_type == 'venv':
            path = env.get('path', '')
            if not path:
                result['reason'] = 'No path specified'
                return result

            # Resolve path
            venv_path = Path(os.path.expanduser(os.path.expandvars(path)))
            if not venv_path.is_absolute():
                venv_path = self.project_root / venv_path

            result['path'] = venv_path

            # Check for python executable
            python_path = venv_path / 'bin' / 'python'
            if not python_path.exists():
                python_path = venv_path / 'Scripts' / 'python.exe'  # Windows

            if python_path.exists():
                result['available'] = True
                result['python_path'] = python_path
            else:
                result['reason'] = f'Python not found at {venv_path}'

            return result

        elif env_type == 'conda':
            conda_name = env.get('name', '')
            if not conda_name:
                result['reason'] = 'No conda environment name specified'
                return result

            # Check if conda env exists
            import subprocess
            try:
                output = subprocess.run(
                    ['conda', 'env', 'list', '--json'],
                    capture_output=True, text=True, timeout=10
                )
                if output.returncode == 0:
                    import json
                    envs = json.loads(output.stdout)
                    for env_path in envs.get('envs', []):
                        if Path(env_path).name == conda_name or env_path.endswith(f'/{conda_name}'):
                            result['available'] = True
                            result['path'] = Path(env_path)
                            result['python_path'] = Path(env_path) / 'bin' / 'python'
                            return result
                result['reason'] = f'Conda environment {conda_name} not found'
            except (subprocess.TimeoutExpired, FileNotFoundError):
                result['reason'] = 'Conda not available'

            return result

        result['reason'] = f'Unknown environment type: {env_type}'
        return result

    def get_environment_for_model(self, model_name: str) -> Optional[str]:
        """
        Get the recommended environment for a model.

        Returns environment name or None if any environment works.
        """
        env_selection = self._inference_settings.get('environment_selection', {})
        model_reqs = env_selection.get('model_requirements', {})

        if model_name in model_reqs:
            return model_reqs[model_name].get('preferred_env')

        # Check preference order for first available
        pref_order = env_selection.get('preference_order', ['default', 'system'])
        for env_name in pref_order:
            status = self.check_environment_available(env_name)
            if status.get('available'):
                return env_name

        return None

    def get_activation_command(self, env_name: str) -> Optional[str]:
        """
        Get the shell command to activate an environment.

        Returns:
            Activation command string, or None if not applicable.
        """
        status = self.check_environment_available(env_name)
        if not status.get('available'):
            return None

        env_type = status.get('type')
        env_config = status.get('config', {})

        if env_type == 'system':
            return None  # No activation needed

        elif env_type == 'venv':
            venv_path = status.get('path')
            activate = venv_path / 'bin' / 'activate'
            if not activate.exists():
                activate = venv_path / 'Scripts' / 'activate'  # Windows
            return f'source {activate}'

        elif env_type == 'conda':
            conda_name = env_config.get('name')
            return f'conda activate {conda_name}'

        return None

    def get_python_for_environment(self, env_name: str) -> Optional[Path]:
        """Get the Python executable path for an environment."""
        status = self.check_environment_available(env_name)
        if status.get('available'):
            return status.get('python_path')
        return None

    def list_available_environments(self) -> List[Dict]:
        """List all environments with their availability status."""
        results = []
        for name in self.get_environments():
            status = self.check_environment_available(name)
            status['name'] = name
            results.append(status)
        return results

    def can_install_packages(self, env_name: str) -> bool:
        """Check if we're allowed to install packages in an environment."""
        env = self.get_environment(env_name)
        if env:
            return env.get('allow_install', False)
        return False

    def get_inference_permission(self, permission: str) -> bool:
        """Check if a specific inference permission is granted."""
        permissions = self._inference_settings.get('inference_permissions', {})
        return permissions.get(permission, False)

    def discover_environments(self, force: bool = False) -> List[Dict]:
        """
        Auto-discover virtual environments if permitted.

        Args:
            force: If True, discover even if permission is False

        Returns:
            List of discovered environment info dicts
        """
        if not force and not self.get_inference_permission('discover_environments'):
            logger.info("Environment discovery not permitted (set inference_permissions.discover_environments: true)")
            return []

        discovered = []

        # Common venv locations to check
        venv_candidates = [
            '.venv',
            'venv',
            'env',
            '.env',
            '.venv38',
            '.venv39',
            '.venv310',
            '.venv311',
            '.venv312',
        ]

        for venv_name in venv_candidates:
            venv_path = self.project_root / venv_name
            python_path = venv_path / 'bin' / 'python'

            if not python_path.exists():
                python_path = venv_path / 'Scripts' / 'python.exe'  # Windows

            if python_path.exists():
                # Get Python version
                import subprocess
                try:
                    result = subprocess.run(
                        [str(python_path), '--version'],
                        capture_output=True, text=True, timeout=5
                    )
                    version = result.stdout.strip().replace('Python ', '')
                except Exception:
                    version = 'unknown'

                # Check numpy version if available
                numpy_version = None
                try:
                    result = subprocess.run(
                        [str(python_path), '-c', 'import numpy; print(numpy.__version__)'],
                        capture_output=True, text=True, timeout=10
                    )
                    if result.returncode == 0:
                        numpy_version = result.stdout.strip()
                except Exception:
                    pass

                discovered.append({
                    'name': venv_name,
                    'type': 'venv',
                    'path': venv_path,
                    'python_path': python_path,
                    'python_version': version,
                    'numpy_version': numpy_version,
                    'discovered': True,
                })

        # Check for conda environments if conda is available
        import json as json_module
        try:
            import subprocess
            result = subprocess.run(
                ['conda', 'env', 'list', '--json'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                conda_data = json_module.loads(result.stdout)
                for env_path in conda_data.get('envs', []):
                    env_path = Path(env_path)
                    if env_path.name and env_path.name != 'base':
                        python_path = env_path / 'bin' / 'python'
                        if python_path.exists():
                            discovered.append({
                                'name': f'conda:{env_path.name}',
                                'type': 'conda',
                                'path': env_path,
                                'python_path': python_path,
                                'conda_name': env_path.name,
                                'discovered': True,
                            })
        except (subprocess.TimeoutExpired, FileNotFoundError, json_module.JSONDecodeError):
            pass  # Conda not available

        return discovered

    def add_discovered_environment(self, env_info: Dict) -> None:
        """Add a discovered environment to the runtime config (not persisted)."""
        name = env_info.get('name')
        if not name:
            return

        if 'environments' not in self._inference_settings:
            self._inference_settings['environments'] = {}

        self._inference_settings['environments'][name] = {
            'type': env_info.get('type', 'venv'),
            'path': str(env_info.get('path', '')),
            'python_version': env_info.get('python_version', 'unknown'),
            'allow_install': False,  # Discovered envs are read-only by default
            'discovered': True,
            'packages': {
                'numpy': env_info.get('numpy_version', 'unknown'),
            } if env_info.get('numpy_version') else {},
        }

        if env_info.get('conda_name'):
            self._inference_settings['environments'][name]['name'] = env_info['conda_name']

    def find_compatible_environment(self, model_name: str = None, requires_numpy2: bool = True) -> Optional[str]:
        """
        Find an environment compatible with a model's requirements.

        Args:
            model_name: Model to find environment for (uses model_requirements config)
            requires_numpy2: If True, require numpy >= 2.0

        Returns:
            Environment name or None
        """
        # Check model-specific requirements first
        if model_name:
            env_name = self.get_environment_for_model(model_name)
            if env_name:
                status = self.check_environment_available(env_name)
                if status.get('available'):
                    return env_name

        # Search for compatible environment
        env_selection = self._inference_settings.get('environment_selection', {})
        pref_order = env_selection.get('preference_order', [])

        # Add all configured environments
        all_envs = list(pref_order) + [
            e for e in self.get_environments().keys() if e not in pref_order
        ]

        for env_name in all_envs:
            status = self.check_environment_available(env_name)
            if not status.get('available'):
                continue

            env_config = status.get('config', {})
            packages = env_config.get('packages', {})

            # Check numpy requirement
            if requires_numpy2:
                numpy_req = packages.get('numpy', '')
                if '>=2' in numpy_req or '>2' in numpy_req or '2.' in numpy_req:
                    return env_name

            # If no numpy requirement, any available env works
            if not requires_numpy2:
                return env_name

        return None

    def smart_find_environment(
        self,
        requires: List[str] = None,
        min_python: str = "3.9",
        allow_discovery: bool = True,
        prefer_writable: bool = True,
    ) -> List[Dict]:
        """
        Smart environment detection with scoring to help user decide.

        Args:
            requires: List of required packages (e.g., ['numpy>=2.0', 'torch'])
            min_python: Minimum Python version (default 3.9 for numpy 2.x)
            allow_discovery: If True, discover environments not in config
            prefer_writable: Prefer environments with allow_install=True

        Returns:
            List of candidate environments sorted by score (best first), each with:
                name: environment name
                score: compatibility score
                available: bool
                has_packages: list of required packages already present
                missing_packages: list that need installing
                python_ok: bool if Python version is sufficient
                reasons: list explaining the score
        """
        requires = requires or ['numpy>=2.0']
        candidates = []

        # Parse min_python
        try:
            min_major, min_minor = map(int, min_python.split('.')[:2])
        except ValueError:
            min_major, min_minor = 3, 9

        # Get configured environments
        configured = set(self.get_environments().keys())

        # Discover additional environments
        discovered = []
        if allow_discovery:
            discovered = self._quick_discover_local_venvs()

        # Add discovered to runtime config
        for env_info in discovered:
            if env_info['name'] not in configured:
                self.add_discovered_environment(env_info)

        # Score all environments
        all_envs = list(self.get_environments().keys())

        for env_name in all_envs:
            status = self.check_environment_available(env_name)
            env_config = self.get_environment(env_name) or {}

            candidate = {
                'name': env_name,
                'score': 0,
                'available': status.get('available', False),
                'has_packages': [],
                'missing_packages': list(requires),
                'python_ok': False,
                'python_version': None,
                'reasons': [],
                'python_path': status.get('python_path'),
                'allow_install': env_config.get('allow_install', False),
                'type': env_config.get('type', 'unknown'),
                'path': env_config.get('path', ''),
            }

            if not status.get('available'):
                candidate['reasons'].append('not available')
                candidates.append(candidate)
                continue

            # Score: available
            candidate['score'] += 1
            candidate['reasons'].append('+1 available')

            # Check Python version (critical for numpy 2.x)
            # Priority: actual version > config > extracted from name
            actual_py_ver = self._get_actual_python_version(status.get('python_path'))
            if actual_py_ver:
                py_ver = actual_py_ver
            else:
                py_ver = self._resolve_env_python_version(env_name, env_config) or ''
            candidate['python_version'] = py_ver

            if py_ver:
                try:
                    major, minor = map(int, py_ver.split('.')[:2])
                    if major > min_major or (major == min_major and minor >= min_minor):
                        candidate['python_ok'] = True
                        candidate['score'] += 5
                        candidate['reasons'].append(f'+5 Python {py_ver} >= {min_python}')
                    else:
                        candidate['score'] -= 10
                        candidate['reasons'].append(f'-10 Python {py_ver} < {min_python}')
                except ValueError:
                    candidate['reasons'].append(f'? Python version unknown: {py_ver}')

            # Score: has required packages
            packages = env_config.get('packages', {})
            has_pkgs = []
            missing_pkgs = []

            for req in requires:
                pkg_name = req.split('>=')[0].split('<=')[0].split('==')[0].split('>')[0].split('<')[0]
                pkg_constraint = req[len(pkg_name):] if len(req) > len(pkg_name) else ''

                if pkg_name in packages:
                    pkg_ver = packages[pkg_name]
                    if not pkg_constraint or pkg_constraint in pkg_ver or pkg_ver == '*':
                        has_pkgs.append(req)
                        candidate['score'] += 10
                        candidate['reasons'].append(f'+10 has {pkg_name}')
                    else:
                        missing_pkgs.append(req)
                else:
                    missing_pkgs.append(req)

            candidate['has_packages'] = has_pkgs
            candidate['missing_packages'] = missing_pkgs

            # Score: allow_install (important if packages missing)
            if env_config.get('allow_install', False):
                if prefer_writable:
                    candidate['score'] += 5
                    candidate['reasons'].append('+5 writable')
            elif missing_pkgs:
                candidate['score'] -= 5
                candidate['reasons'].append('-5 read-only with missing packages')

            # Score: project-local venvs preferred
            path = env_config.get('path', '')
            if path in ['.venv', 'venv', '.env', 'env'] or path.startswith('./'):
                candidate['score'] += 3
                candidate['reasons'].append('+3 project-local')

            # Score: is 'default' (slight preference)
            if env_name == 'default':
                candidate['score'] += 1
                candidate['reasons'].append('+1 is default')

            # Penalty: system Python
            if env_config.get('type') == 'system':
                candidate['score'] -= 20
                candidate['reasons'].append('-20 system Python (avoid)')

            candidates.append(candidate)

        # Sort by score descending
        candidates.sort(key=lambda x: (-x['score'], x['name']))
        return candidates

    def _get_actual_python_version(self, python_path) -> Optional[str]:
        """Get actual Python version from executable."""
        if not python_path or not Path(python_path).exists():
            return None
        try:
            import subprocess
            result = subprocess.run(
                [str(python_path), '--version'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                # "Python 3.9.5" -> "3.9.5"
                return result.stdout.strip().replace('Python ', '')
        except Exception:
            pass
        return None

    @staticmethod
    def _extract_python_version_from_name(env_name: str) -> Optional[str]:
        """
        Extract Python version from environment name patterns.

        Supports patterns like:
            venv-3.9      -> 3.9
            venv-3.11     -> 3.11
            default39     -> 3.9
            venv311       -> 3.11
            .venv312      -> 3.12
            py39-env      -> 3.9
            env-py311     -> 3.11

        Returns:
            Python version string (e.g., "3.9") or None if not found
        """
        import re

        # Pattern: explicit version with dot (e.g., venv-3.9, .venv-3.11)
        match = re.search(r'[._-]?(\d+\.\d+)', env_name)
        if match:
            return match.group(1)

        # Pattern: version without dot at end (e.g., default39, venv311, .venv312)
        match = re.search(r'(\d)(\d{1,2})$', env_name)
        if match:
            major, minor = match.groups()
            return f"{major}.{minor}"

        # Pattern: py followed by version (e.g., py39-env, env-py311)
        match = re.search(r'py(\d)(\d{1,2})', env_name, re.IGNORECASE)
        if match:
            major, minor = match.groups()
            return f"{major}.{minor}"

        return None

    def _resolve_env_python_version(self, env_name: str, env_config: Dict) -> Optional[str]:
        """
        Resolve Python version for an environment.

        Checks in order:
        1. Explicit python_version in config
        2. Extract from environment name pattern
        3. Return None if unknown

        Args:
            env_name: Environment name
            env_config: Environment configuration dict

        Returns:
            Python version string or None
        """
        # Check explicit config
        py_version = env_config.get('python_version')
        if py_version and py_version != 'unknown':
            return py_version

        # Try extracting from name
        extracted = self._extract_python_version_from_name(env_name)
        if extracted:
            return extracted

        return None

    def get_default_env_name(self, python_version: str = None) -> str:
        """
        Get the default environment name, optionally for a specific Python version.

        Uses the pattern from config if specified (e.g., "venv-*" where * is version).

        Args:
            python_version: Python version (e.g., "3.11")

        Returns:
            Environment name (e.g., "venv-3.11" or "default")
        """
        env_config = self._inference_settings.get('environment_selection', {})
        pattern = env_config.get('default_env_pattern', 'default')

        if '*' in pattern and python_version:
            return pattern.replace('*', python_version)
        elif '*' in pattern:
            # Pattern specified but no version - use current Python
            import sys
            current_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
            return pattern.replace('*', current_ver)
        else:
            return pattern

    def _quick_discover_local_venvs(self) -> List[Dict]:
        """Quick discovery of local venvs without full subprocess checks."""
        discovered = []
        venv_names = ['.venv', 'venv', 'env', '.env']

        for name in venv_names:
            venv_path = self.project_root / name
            python_path = venv_path / 'bin' / 'python'

            if not python_path.exists():
                python_path = venv_path / 'Scripts' / 'python.exe'

            if python_path.exists():
                discovered.append({
                    'name': name,
                    'type': 'venv',
                    'path': str(venv_path),
                    'python_path': python_path,
                    'python_version': 'unknown',
                    'discovered': True,
                    'allow_install': True,  # Local venvs are writable
                })

        return discovered

    def select_install_target(
        self,
        requires: List[str] = None,
        auto: bool = False,
    ) -> Union[Dict, List[Dict], None]:
        """
        Select the best environment for package installation.

        Args:
            requires: Required packages
            auto: If True, auto-select best; if False, return ranked list

        Returns:
            If auto: best environment dict or None
            If not auto: list of candidates sorted by score
        """
        candidates = self.smart_find_environment(
            requires=requires,
            allow_discovery=True,
            prefer_writable=True,
        )

        # Filter to available environments with good Python
        viable = [c for c in candidates if c['available'] and c['python_ok']]

        if auto:
            # Return best writable, or best overall
            writable = [c for c in viable if c['allow_install']]
            return writable[0] if writable else (viable[0] if viable else None)

        return candidates  # Return full list for user to review

    def create_environment(
        self,
        name: str,
        python_executable: str = None,
        packages: List[str] = None,
        dry_run: bool = False,
    ) -> bool:
        """
        Create a new virtual environment.

        Args:
            name: Environment name (must be configured)
            python_executable: Python to use (default from config or python3.9)
            packages: Packages to install after creation
            dry_run: If True, just show what would be done

        Returns:
            True if successful
        """
        import subprocess

        env_config = self.get_environment(name)
        if not env_config:
            logger.error(f"Environment '{name}' not configured")
            return False

        if env_config.get('type') != 'venv':
            logger.error(f"Can only create venv type, not {env_config.get('type')}")
            return False

        path = env_config.get('path', '')
        if not path:
            logger.error(f"No path configured for '{name}'")
            return False

        venv_path = Path(os.path.expanduser(os.path.expandvars(path)))
        if not venv_path.is_absolute():
            venv_path = self.project_root / venv_path

        # Determine Python executable
        if not python_executable:
            python_executable = env_config.get('python_executable')
        if not python_executable:
            py_ver = env_config.get('python_version', '3.9')
            python_executable = f'python{py_ver}'

        if dry_run:
            print(f"Would create venv at: {venv_path}")
            print(f"Using Python: {python_executable}")
            if packages:
                print(f"Would install: {', '.join(packages)}")
            return True

        if venv_path.exists():
            logger.warning(f"Already exists: {venv_path}")
            return False

        logger.info(f"Creating venv at {venv_path}...")
        try:
            result = subprocess.run(
                [python_executable, '-m', 'venv', str(venv_path)],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                logger.error(f"Failed: {result.stderr}")
                return False
        except FileNotFoundError:
            logger.error(f"Python not found: {python_executable}")
            return False

        if packages:
            return self.install_packages(name, packages)

        return True

    def install_packages(
        self,
        env_name: str,
        packages: List[str],
        dry_run: bool = False,
    ) -> bool:
        """
        Install packages into an environment.

        Args:
            env_name: Target environment
            packages: Packages to install
            dry_run: If True, just show what would be done

        Returns:
            True if successful
        """
        import subprocess

        env_config = self.get_environment(env_name)
        if not env_config:
            logger.error(f"Environment '{env_name}' not configured")
            return False

        if not env_config.get('allow_install', False):
            logger.error(f"Installation not allowed for '{env_name}'")
            return False

        if env_config.get('type') == 'system':
            logger.error("Cannot install to system Python")
            return False

        status = self.check_environment_available(env_name)
        if not status.get('available'):
            logger.error(f"Environment '{env_name}' not available")
            return False

        python_path = status.get('python_path')

        if dry_run:
            print(f"Would install to: {env_name}")
            print(f"Python: {python_path}")
            print(f"Packages: {', '.join(packages)}")
            return True

        logger.info(f"Installing to {env_name}: {', '.join(packages)}")
        try:
            result = subprocess.run(
                [str(python_path), '-m', 'pip', 'install'] + packages,
                capture_output=True, text=True
            )
            if result.returncode != 0:
                logger.error(f"pip failed: {result.stderr}")
                return False
            logger.info("Installation successful")
            return True
        except Exception as e:
            logger.error(f"Failed: {e}")
            return False

    def setup_environment(
        self,
        env_name: str = 'default',
        packages: List[str] = None,
        dry_run: bool = False,
    ) -> bool:
        """
        Set up environment: create if needed, then install packages.

        Args:
            env_name: Environment name
            packages: Packages (default: inference requirements)
            dry_run: If True, just show what would be done

        Returns:
            True if successful
        """
        if packages is None:
            packages = ['numpy>=2.0', 'torch>=2.0', 'sentence-transformers>=2.2', 'pyyaml']

        status = self.check_environment_available(env_name)

        if not status.get('available'):
            if dry_run:
                print(f"Would create environment: {env_name}")
            if not self.create_environment(env_name, dry_run=dry_run):
                if not dry_run:
                    return False

        return self.install_packages(env_name, packages, dry_run=dry_run)

    def run_with_environment(
        self,
        command: List[str],
        model_name: str = None,
        env_name: str = None,
        interactive: bool = False,
    ) -> Union[int, str]:
        """
        Run a command using the appropriate environment for a model.

        Args:
            command: Command to run (script path and args, without python)
            model_name: Model name to determine environment
            env_name: Override environment name
            interactive: If True, run interactively

        Returns:
            Exit code (if interactive) or stdout (if not interactive)
        """
        import subprocess

        # Determine environment
        if not env_name:
            env_name = self.find_compatible_environment(model_name, requires_numpy2=True)

        # Get Python executable
        if env_name:
            python_path = self.get_python_for_environment(env_name)
            if python_path:
                logger.info(f"Using environment '{env_name}': {python_path}")
            else:
                logger.warning(f"Environment '{env_name}' not available, falling back to system Python")
                python_path = sys.executable
        else:
            logger.warning("No compatible environment found, using system Python")
            python_path = sys.executable

        # Build full command
        full_cmd = [str(python_path)] + command

        if interactive:
            result = subprocess.run(full_cmd)
            return result.returncode
        else:
            result = subprocess.run(full_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Command failed: {result.stderr}")
            return result.stdout

    def run_inference_script(
        self,
        model_name: str,
        query: str = None,
        interactive: bool = False,
        top_k: int = 10,
        **kwargs
    ) -> Union[int, str]:
        """
        Run the inference script with the correct environment.

        Args:
            model_name: Model to use
            query: Query text (if not interactive)
            interactive: Run in interactive mode
            top_k: Number of results
            **kwargs: Additional arguments

        Returns:
            Exit code (if interactive) or stdout
        """
        model_path = self.get_model_path(model_name)
        if not model_path:
            raise FileNotFoundError(f"Model not found: {model_name}")

        script = self.project_root / 'scripts' / 'infer_pearltrees_federated.py'
        if not script.exists():
            raise FileNotFoundError(f"Inference script not found: {script}")

        cmd = [str(script), '--model', str(model_path), '--top-k', str(top_k)]

        if interactive:
            cmd.append('--interactive')
        elif query:
            cmd.extend(['--query', query])

        for key, val in kwargs.items():
            if val is not None:
                cmd.extend([f'--{key.replace("_", "-")}', str(val)])

        return self.run_with_environment(
            cmd,
            model_name=model_name,
            interactive=interactive
        )


def main():
    """CLI interface for model registry."""
    import argparse

    parser = argparse.ArgumentParser(description='UnifyWeaver Model Registry')
    parser.add_argument('--list', '-l', action='store_true', help='List all models')
    parser.add_argument('--type', '-t', help='Filter by model type (embedding, federated, orthogonal_codebook)')
    parser.add_argument('--embedding', '-e', help='Filter by embedding model (e.g., nomic, minilm, bge)')
    parser.add_argument('--scope', help='Filter by scope (multi_account, single_account, domain, holdout)')
    parser.add_argument('--domain', help='Filter by domain (e.g., physics)')
    parser.add_argument('--tag', action='append', dest='tags', help='Include models with this tag (can repeat)')
    parser.add_argument('--exclude-tag', action='append', dest='exclude_tags',
                        help='Exclude models with this tag (can repeat)')
    parser.add_argument('--available', action='store_true', help='Only show available models')
    parser.add_argument('--all', action='store_true',
                        help='Show all models including holdout/experimental (default excludes these)')
    parser.add_argument('--task', help='Show models for a task')
    parser.add_argument('--training', '-T', help='Show training command for model')
    parser.add_argument('--path', '-p', help='Show resolved path for model')
    parser.add_argument('--check', '-c', help='Check if model is available')
    parser.add_argument('--missing', '-m', help='Show missing models for task')
    parser.add_argument('--age', '-a', help='Show model age in days')
    parser.add_argument('--knowledge', '-k', action='store_true',
                        help='Show best knowledge model')
    parser.add_argument('--prefer-newer', action='store_true',
                        help='Prefer newer models (with --knowledge)')
    parser.add_argument('--prefer-federated', action='store_true',
                        help='Prefer federated over transformer (with --knowledge)')
    parser.add_argument('--prefer-transformer', action='store_true',
                        help='Prefer transformer over federated (with --knowledge)')
    parser.add_argument('--load', help='Load and show model info')
    parser.add_argument('--modes', action='store_true', help='List available use modes')
    parser.add_argument('--mode-for', help='Show recommended mode for a task')
    parser.add_argument('--run', help='Run a mode (agent, loop, inference)')
    parser.add_argument('--query', '-q', help='Query text for --run')
    # Environment commands
    parser.add_argument('--envs', action='store_true', help='List configured environments')
    parser.add_argument('--env', help='Check specific environment availability')
    parser.add_argument('--env-for', help='Show recommended environment for a model')
    parser.add_argument('--activate', help='Show activation command for environment')
    parser.add_argument('--discover-envs', action='store_true',
                        help='Discover available environments (requires permission or --force)')
    parser.add_argument('--force', action='store_true',
                        help='Force discovery even if not permitted in config')
    parser.add_argument('--infer', metavar='MODEL',
                        help='Run inference with auto-environment detection')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Run in interactive mode (with --infer)')
    parser.add_argument('--top-k', type=int, default=10,
                        help='Number of results (with --infer)')
    # Environment setup commands
    parser.add_argument('--smart-envs', action='store_true',
                        help='Smart environment detection with scoring')
    parser.add_argument('--create-env', metavar='NAME',
                        help='Create a virtual environment')
    parser.add_argument('--setup-env', metavar='NAME', nargs='?', const='default',
                        help='Set up environment (create + install packages)')
    parser.add_argument('--install', metavar='ENV',
                        help='Install inference packages to environment')
    parser.add_argument('--packages', nargs='+',
                        help='Packages to install (with --install or --setup-env)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without doing it')
    parser.add_argument('--auto', action='store_true',
                        help='Auto-select best environment without prompting')

    args = parser.parse_args()
    registry = ModelRegistry()

    if args.list:
        # Build exclude_tags list - default excludes holdout/experimental unless --all
        exclude_tags = args.exclude_tags or []
        if not args.all:
            exclude_tags.extend(['holdout', 'experimental', 'validation'])

        models = registry.list_models(
            model_type=args.type,
            embedding_model=args.embedding,
            scope=args.scope,
            domain=args.domain,
            tags=args.tags,
            exclude_tags=exclude_tags if exclude_tags else None,
            available_only=args.available,
        )

        if not models:
            print("No models match the specified filters.")
            sys.exit(0)

        # Sort: Nomic models first, then by type, then by name
        def sort_key(m):
            # Nomic preferred (0), others later (1)
            nomic_score = 0 if m.embedding_model and 'nomic' in m.embedding_model.lower() else 1
            # Alternative/fallback models last
            alt_score = 1 if 'alternative' in m.tags else 0
            # Type order: embedding, federated, transformer
            type_order = {'embedding': 0, 'federated': 1, 'orthogonal_codebook': 2}.get(m.type, 3)
            return (nomic_score, alt_score, type_order, m.name)

        models.sort(key=sort_key)

        for m in models:
            if registry.check_model_available(m.name):
                available = "+"
            elif m.get_huggingface_id():
                available = "H"  # HuggingFace - downloadable
            else:
                available = "-"
            age = registry.get_model_age_days(m.name)
            age_str = f"{age:.1f}d" if age else "n/a"

            # Build info string
            info_parts = [m.type]
            if m.embedding_model:
                # Shorten embedding model name
                emb_short = m.embedding_model.split('/')[-1][:12]
                info_parts.append(emb_short)
            if m.scope:
                info_parts.append(m.scope)
            if m.tags:
                info_parts.append(','.join(m.tags[:2]))

            info_str = ', '.join(info_parts)
            desc = m.description[:35] if m.description else ""
            print(f"[{available}] {m.name}")
            print(f"      ({info_str}, {age_str})")
            if desc:
                print(f"      {desc}...")

    elif args.task:
        models = registry.get_for_task(args.task)
        print(f"Models for task '{args.task}':")
        for role, model in models.items():
            avail = registry.get_model_availability(model.name)
            age = registry.get_model_age_days(model.name)
            age_str = f", {age:.1f} days old" if age else ""

            # Build status string
            sources = []
            if avail['local_path']:
                sources.append('local')
            if avail['generated_path'] and avail['generated_path'] != avail['local_path']:
                sources.append(f"generated: {avail['generated_path'].name}")
            if avail['huggingface_cached']:
                sources.append('HF cached')
            elif avail['huggingface_id'] and not avail['available']:
                sources.append('HF (not cached)')

            if sources:
                status = ', '.join(sources)
            elif avail['huggingface_id']:
                status = "HuggingFace (download on use)"
            else:
                status = "MISSING"

            print(f"  {role}: {model.name} [{status}{age_str}]")

    elif args.training:
        cmd = registry.get_training_command(args.training)
        if cmd:
            print(cmd)
        else:
            print(f"No training info for '{args.training}'")

    elif args.path:
        path = registry.get_model_path(args.path)
        if path:
            print(path)
        else:
            print(f"Model '{args.path}' not found")

    elif args.check:
        avail = registry.get_model_availability(args.check)
        print(f"Model: {args.check}")
        print(f"  Available: {'Yes' if avail['available'] else 'No'}")
        if avail['local_path']:
            print(f"  Local path: {avail['local_path']}")
        if avail['generated_path']:
            print(f"  Generated path: {avail['generated_path']}")
        if avail['huggingface_id']:
            cached = "Yes" if avail['huggingface_cached'] else "No (will download on use)"
            print(f"  HuggingFace: {avail['huggingface_id']}")
            print(f"  HF Cached: {cached}")

    elif args.age:
        age = registry.get_model_age_days(args.age)
        if age is not None:
            ts = registry.get_model_timestamp(args.age)
            print(f"{args.age}: {age:.1f} days old (modified: {ts})")
        else:
            print(f"Model '{args.age}' not found")

    elif args.knowledge:
        try:
            _, name = registry.load_knowledge_model(
                prefer_newer=args.prefer_newer,
                prefer_federated=args.prefer_federated,
                prefer_transformer=args.prefer_transformer,
            )
            meta = registry.get_model(name)
            age = registry.get_model_age_days(name)
            print(f"Best knowledge model: {name}")
            print(f"  Type: {meta.type}")
            print(f"  Age: {age:.1f} days" if age else "  Age: unknown")
            if meta.metrics:
                print(f"  Metrics: {meta.metrics}")
            print(f"\nTo load in Python:")
            print(f"  registry.load_model('{name}')")
        except FileNotFoundError as e:
            print(f"Error: {e}")

    elif args.load:
        try:
            model = registry.load_model(args.load)
            print(f"Loaded: {args.load}")
            print(f"  Type: {type(model).__name__}")
            if hasattr(model, 'meta'):
                print(f"  Clusters: {model.meta.get('num_clusters', 'n/a')}")
        except Exception as e:
            print(f"Error loading '{args.load}': {e}")

    elif args.missing:
        missing = registry.get_missing_models(args.missing)
        if missing:
            print(f"Missing models for '{args.missing}':")
            for m in missing:
                cmd = registry.get_training_command(m)
                print(f"  - {m}")
                if cmd:
                    print(f"    Train with: {cmd.split(chr(10))[0]}...")
        else:
            print(f"All models available for '{args.missing}'")

    elif args.modes:
        modes = registry.get_use_modes()
        print("Available use modes:")
        for name, config in modes.items():
            desc = config.get('description', '')
            script = config.get('script') or config.get('launcher', '')
            requires_llm = config.get('requires_llm', False)
            llm_str = " (requires LLM)" if requires_llm else ""
            print(f"\n  {name}{llm_str}")
            print(f"    {desc}")
            print(f"    Script: {script}")
            best_for = config.get('best_for', [])
            if best_for:
                print(f"    Best for: {', '.join(best_for)}")

    elif args.mode_for:
        mode = registry.get_mode_for_task(args.mode_for)
        if mode:
            print(f"Recommended mode for '{args.mode_for}': {mode}")
            # Show all variants
            task_modes = registry._inference_settings.get('task_modes', {})
            task_config = task_modes.get(args.mode_for, {})
            if len(task_config) > 1:
                print("  Variants:")
                for variant, m in task_config.items():
                    print(f"    {variant}: {m}")
        else:
            print(f"No mode recommendation for task: {args.mode_for}")

    elif args.run:
        mode = args.run
        query = args.query
        task = args.task

        if mode == 'agent':
            print(f"Launching {mode} mode interactively...")
            exit_code = registry.run_mode(mode, task=task, interactive=True)
            sys.exit(exit_code)
        elif query:
            print(f"Running {mode} mode with query: {query[:50]}...")
            output = registry.run_mode(mode, task=task, query=query)
            print(output)
        else:
            # Show command that would be run
            try:
                cmd = registry.get_mode_command(mode, task=task)
                print(f"Mode: {mode}")
                print(f"Command: {' '.join(cmd)}")
                print("\nAdd --query 'text' to run, or use --run agent for interactive")
            except ValueError as e:
                print(f"Error: {e}")

    elif args.envs:
        print("Configured environments:")
        for env_info in registry.list_available_environments():
            name = env_info['name']
            available = "+" if env_info.get('available') else "-"
            env_type = env_info.get('type', 'unknown')
            config = env_info.get('config', {})
            python_ver = config.get('python_version', '?')
            allow_install = "writable" if config.get('allow_install') else "read-only"

            print(f"\n[{available}] {name} ({env_type}, Python {python_ver}, {allow_install})")

            if env_info.get('available'):
                if env_info.get('path'):
                    print(f"    Path: {env_info['path']}")
                if env_info.get('python_path'):
                    print(f"    Python: {env_info['python_path']}")
            else:
                reason = env_info.get('reason', 'Unknown')
                print(f"    Not available: {reason}")

            # Show package requirements
            packages = config.get('packages', {})
            if packages:
                pkg_str = ', '.join(f"{k}{v}" for k, v in packages.items())
                print(f"    Packages: {pkg_str}")

    elif args.env:
        status = registry.check_environment_available(args.env)
        print(f"Environment: {args.env}")
        print(f"  Available: {'Yes' if status.get('available') else 'No'}")
        print(f"  Type: {status.get('type', 'unknown')}")

        if status.get('available'):
            if status.get('path'):
                print(f"  Path: {status['path']}")
            if status.get('python_path'):
                print(f"  Python: {status['python_path']}")
            activation = registry.get_activation_command(args.env)
            if activation:
                print(f"  Activate: {activation}")
            can_install = registry.can_install_packages(args.env)
            print(f"  Can install packages: {'Yes' if can_install else 'No'}")
        else:
            print(f"  Reason: {status.get('reason', 'Unknown')}")

    elif args.env_for:
        env_name = registry.get_environment_for_model(args.env_for)
        if env_name:
            print(f"Recommended environment for '{args.env_for}': {env_name}")
            activation = registry.get_activation_command(env_name)
            if activation:
                print(f"  Activate: {activation}")
        else:
            print(f"No specific environment required for '{args.env_for}'")

    elif args.activate:
        activation = registry.get_activation_command(args.activate)
        if activation:
            print(activation)
        else:
            status = registry.check_environment_available(args.activate)
            if not status.get('available'):
                print(f"Environment '{args.activate}' not available: {status.get('reason')}")
            else:
                print(f"No activation needed for '{args.activate}'")

    elif args.infer:
        model_name = args.infer
        query = args.query
        interactive = args.interactive

        # Find compatible environment
        env_name = registry.find_compatible_environment(model_name, requires_numpy2=True)
        if env_name:
            status = registry.check_environment_available(env_name)
            print(f"Using environment: {env_name}")
            print(f"  Python: {status.get('python_path')}")
        else:
            print("Warning: No compatible environment found, using system Python")
            print("  Models saved with numpy 2.x may fail to load")

        try:
            if interactive:
                print(f"\nLaunching interactive inference for '{model_name}'...\n")
                exit_code = registry.run_inference_script(model_name, interactive=True, top_k=args.top_k)
                sys.exit(exit_code)
            elif query:
                print(f"\nRunning inference for '{model_name}' with query: {query[:50]}...\n")
                output = registry.run_inference_script(model_name, query=query, top_k=args.top_k)
                print(output)
            else:
                print(f"\nModel: {model_name}")
                print(f"Use --query 'text' for single query, or --interactive for REPL")
                model_path = registry.get_model_path(model_name)
                if model_path:
                    print(f"\nDirect command:")
                    python_path = registry.get_python_for_environment(env_name) if env_name else sys.executable
                    print(f"  {python_path} scripts/infer_pearltrees_federated.py --model {model_path} --interactive")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.discover_envs:
        permitted = registry.get_inference_permission('discover_environments')
        if not permitted and not args.force:
            print("Environment discovery not permitted.")
            print("Set 'inference_permissions.discover_environments: true' in config")
            print("Or use --force to override")
            sys.exit(1)

        print("Discovering environments..." + (" (forced)" if args.force and not permitted else ""))
        discovered = registry.discover_environments(force=args.force)

        if not discovered:
            print("No environments discovered.")
        else:
            print(f"\nFound {len(discovered)} environment(s):\n")
            for env in discovered:
                env_type = env.get('type', 'venv')
                name = env.get('name')
                py_ver = env.get('python_version', '?')
                numpy_ver = env.get('numpy_version')

                print(f"[+] {name} ({env_type}, Python {py_ver})")
                print(f"    Path: {env.get('path')}")
                if numpy_ver:
                    print(f"    numpy: {numpy_ver}")

                # Show activation command
                registry.add_discovered_environment(env)
                activation = registry.get_activation_command(name)
                if activation:
                    print(f"    Activate: {activation}")
                print()

    elif args.smart_envs:
        requires = args.packages or ['numpy>=2.0']
        print(f"Smart environment detection (requires: {', '.join(requires)})")
        print()

        candidates = registry.smart_find_environment(
            requires=requires,
            allow_discovery=True,
            prefer_writable=True,
        )

        if not candidates:
            print("No environments found.")
            sys.exit(1)

        print(f"{'Rank':<5} {'Score':<6} {'Env Name':<20} {'Python':<10} {'Status'}")
        print("-" * 70)

        for i, c in enumerate(candidates, 1):
            status_parts = []
            if not c['available']:
                status_parts.append('NOT AVAILABLE')
            else:
                if c['python_ok']:
                    status_parts.append('Python OK')
                else:
                    status_parts.append('Python too old')
                if c['has_packages']:
                    status_parts.append(f"has: {','.join(c['has_packages'][:2])}")
                if c['missing_packages']:
                    status_parts.append(f"needs: {','.join(c['missing_packages'][:2])}")
                if c['allow_install']:
                    status_parts.append('writable')

            py_ver = c['python_version'] or '?'
            status = ', '.join(status_parts)
            print(f"{i:<5} {c['score']:<6} {c['name']:<20} {py_ver:<10} {status}")

        # Show recommendation
        print()
        viable = [c for c in candidates if c['available'] and c['python_ok']]
        if viable:
            best = viable[0]
            writable_best = [c for c in viable if c['allow_install']]
            if writable_best:
                print(f"Recommended for installation: {writable_best[0]['name']}")
            else:
                print(f"Best available (read-only): {best['name']}")
        else:
            print("No viable environments found. Consider creating one:")
            print(f"  python3 -m unifyweaver.config.model_registry --create-env default")

    elif args.create_env:
        env_name = args.create_env
        packages = args.packages
        dry_run = args.dry_run

        print(f"Creating environment: {env_name}")
        if registry.create_environment(env_name, packages=packages, dry_run=dry_run):
            if not dry_run:
                print(f"Environment '{env_name}' created successfully.")
        else:
            if not dry_run:
                print(f"Failed to create environment '{env_name}'.")
                sys.exit(1)

    elif args.setup_env is not None:
        env_name = args.setup_env
        packages = args.packages
        dry_run = args.dry_run

        print(f"Setting up environment: {env_name}")
        if registry.setup_environment(env_name, packages=packages, dry_run=dry_run):
            if not dry_run:
                print(f"Environment '{env_name}' is ready.")
        else:
            if not dry_run:
                print(f"Failed to set up environment '{env_name}'.")
                sys.exit(1)

    elif args.install:
        env_name = args.install
        packages = args.packages or ['numpy>=2.0', 'torch>=2.0', 'sentence-transformers>=2.2']
        dry_run = args.dry_run

        if registry.install_packages(env_name, packages, dry_run=dry_run):
            if not dry_run:
                print("Installation complete.")
        else:
            if not dry_run:
                print("Installation failed.")
                sys.exit(1)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
