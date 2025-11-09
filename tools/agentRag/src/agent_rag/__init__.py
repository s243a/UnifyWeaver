"""
Agent-Based RAG System
A revolutionary approach to RAG using LLMs as intelligent retrievers
"""

__version__ = "1.0.0"
__author__ = "Agent RAG Team"

from pathlib import Path

# Get the package root directory
PACKAGE_ROOT = Path(__file__).parent  # This is src/agent_rag/
PROJECT_ROOT = PACKAGE_ROOT.parent.parent  # This goes up to project root
CONFIG_DIR = PACKAGE_ROOT / "config"
CORE_DIR = PACKAGE_ROOT / "core"
TEST_DIR = PACKAGE_ROOT / "tests"