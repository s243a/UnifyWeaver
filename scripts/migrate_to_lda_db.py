#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# Migration script: Import qa_pairs_v1.json into LDA database

"""
Migrate existing Q-A pairs to SQLite database.

This script:
1. Creates the database schema
2. Imports answers and questions from qa_pairs_v1.json
3. Creates clusters linking them
4. Embeds everything with sentence-transformers
5. Trains and stores the projection matrix
"""

import sys
import json
from pathlib import Path
import argparse

import numpy as np

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src" / "unifyweaver" / "targets" / "python_runtime"))

from lda_database import LDAProjectionDB
from projection import compute_W


def load_qa_pairs(path: str) -> dict:
    """Load Q-A pairs from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def import_to_database(db: LDAProjectionDB, qa_data: dict, model_name: str = "all-MiniLM-L6-v2"):
    """Import Q-A pairs into database.

    Returns:
        List of (cluster_id, answer_ids, question_ids) tuples
    """
    clusters_info = []

    for cluster_data in qa_data['clusters']:
        cluster_id_name = cluster_data['id']
        source_file = cluster_data.get('answer_source', 'unknown')

        print(f"\n  Importing cluster: {cluster_id_name}")

        # Add answers (one per model variant)
        answer_ids = []
        answers_dict = cluster_data['answers']

        # First add the default answer
        default_text = answers_dict.get('default', '')
        if default_text:
            aid = db.add_answer(
                source_file=source_file,
                text=default_text,
                record_id=cluster_id_name,
                text_variant='default'
            )
            answer_ids.append(aid)
            print(f"    Added default answer: id={aid}")

        # Add model-specific variants as related answers
        for model, text in answers_dict.items():
            if model != 'default' and model == model_name:
                # This is the primary answer for our target model
                aid = db.add_answer(
                    source_file=source_file,
                    text=text,
                    record_id=cluster_id_name,
                    text_variant=model
                )
                # Link as variant of default
                if answer_ids:
                    db.add_relation(aid, answer_ids[0], 'variant_of')
                answer_ids.append(aid)
                print(f"    Added {model} variant: id={aid}")

        # Add questions
        question_ids = []
        queries = cluster_data['queries']

        for length_type, query_list in queries.items():
            for query_text in query_list:
                qid = db.add_question(
                    text=query_text,
                    length_type=length_type
                )
                question_ids.append(qid)

        print(f"    Added {len(question_ids)} questions")

        # Create cluster
        cluster_id = db.create_cluster(
            name=cluster_id_name,
            answer_ids=answer_ids,
            question_ids=question_ids,
            description=f"Q-A cluster for {cluster_id_name}"
        )

        clusters_info.append((cluster_id, answer_ids, question_ids))
        print(f"    Created cluster: id={cluster_id}")

    return clusters_info


def embed_all(db: LDAProjectionDB, model_name: str, clusters_info: list):
    """Embed all answers and questions."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed")
        print("  pip install sentence-transformers")
        return None

    print(f"\nLoading embedding model: {model_name}")
    embedder = SentenceTransformer(model_name)
    dimension = embedder.get_sentence_embedding_dimension()

    # Register model
    model_id = db.add_model(
        name=model_name,
        dimension=dimension,
        backend='python',
        max_tokens=256,
        notes='sentence-transformers'
    )
    print(f"  Registered model: id={model_id}, dim={dimension}")

    # Embed answers
    print("\nEmbedding answers...")
    all_answers = set()
    for cluster_id, answer_ids, question_ids in clusters_info:
        all_answers.update(answer_ids)

    for aid in all_answers:
        answer = db.get_answer(aid)
        if answer:
            emb = embedder.encode(answer['text'], convert_to_numpy=True)
            db.store_embedding(model_id, 'answer', aid, emb)
    print(f"  Embedded {len(all_answers)} answers")

    # Embed questions
    print("Embedding questions...")
    all_questions = set()
    for cluster_id, answer_ids, question_ids in clusters_info:
        all_questions.update(question_ids)

    for qid in all_questions:
        question = db.get_question(qid)
        if question:
            emb = embedder.encode(question['text'], convert_to_numpy=True)
            db.store_embedding(model_id, 'question', qid, emb)
    print(f"  Embedded {len(all_questions)} questions")

    return model_id, embedder


def train_projection(db: LDAProjectionDB, model_id: int, clusters_info: list, lambda_reg: float = 1.0, ridge: float = 1e-6):
    """Train and store projection matrix."""
    print("\nTraining projection matrix...")

    # Build clusters in format expected by compute_W
    # Format: List of (answer_embedding, question_embeddings_matrix)
    training_clusters = []
    cluster_ids = []

    for cluster_id, answer_ids, question_ids in clusters_info:
        # Get answer embeddings (use first one for the cluster)
        answer_emb = None
        for aid in answer_ids:
            emb = db.get_embedding(model_id, 'answer', aid)
            if emb is not None:
                answer_emb = emb
                break

        if answer_emb is None:
            print(f"  Warning: No answer embedding for cluster {cluster_id}")
            continue

        # Get question embeddings
        q_embs = []
        for qid in question_ids:
            emb = db.get_embedding(model_id, 'question', qid)
            if emb is not None:
                q_embs.append(emb)

        if not q_embs:
            print(f"  Warning: No question embeddings for cluster {cluster_id}")
            continue

        training_clusters.append((answer_emb, np.array(q_embs)))
        cluster_ids.append(cluster_id)

    if not training_clusters:
        print("  ERROR: No valid training clusters")
        return None

    print(f"  Training with {len(training_clusters)} clusters")

    # Compute W matrix
    W = compute_W(training_clusters, lambda_reg=lambda_reg, ridge=ridge)
    print(f"  W shape: {W.shape}")

    # Calculate metrics
    total_queries = sum(len(q_embs) for _, q_embs in training_clusters)
    correct = 0
    ranks = []

    answer_embs = np.array([a for a, _ in training_clusters])
    answer_norms = np.linalg.norm(answer_embs, axis=1, keepdims=True)
    answer_embs_normed = answer_embs / answer_norms

    for i, (_, q_embs) in enumerate(training_clusters):
        for q in q_embs:
            projected = W @ q
            proj_norm = np.linalg.norm(projected)
            if proj_norm > 0:
                projected = projected / proj_norm

            sims = answer_embs_normed @ projected
            ranking = np.argsort(-sims)
            rank = np.where(ranking == i)[0][0] + 1

            if rank == 1:
                correct += 1
            ranks.append(1.0 / rank)

    recall_at_1 = correct / total_queries
    mrr = np.mean(ranks)

    print(f"  Recall@1: {recall_at_1:.2%}")
    print(f"  MRR: {mrr:.4f}")

    # Store projection (symmetric: same model for input and output)
    projection_id = db.store_projection(
        input_model_id=model_id,
        output_model_id=model_id,
        W=W,
        name="v1_from_migration",
        cluster_ids=cluster_ids,
        lambda_reg=lambda_reg,
        ridge=ridge,
        metrics={
            'num_clusters': len(training_clusters),
            'total_queries': total_queries,
            'recall_at_1': recall_at_1,
            'mrr': mrr
        }
    )

    print(f"  Stored projection: id={projection_id}")
    return projection_id


def test_search(db: LDAProjectionDB, projection_id: int, embedder):
    """Test the search API with a novel query."""
    print("\nTesting search API...")

    test_queries = [
        "How do I load data from a CSV file?",
        "mutual recursion predicates",
        "component registration system"
    ]

    for query in test_queries:
        print(f"\n  Query: \"{query}\"")
        results = db.search_with_embedder(
            query_text=query,
            projection_id=projection_id,
            embedder=embedder,
            top_k=3,
            log=True
        )

        for i, r in enumerate(results):
            print(f"    {i+1}. [{r['score']:.3f}] {r['record_id']}: {r['text'][:60]}...")


def process_batch(db: LDAProjectionDB, batch_id: int, source_file: str, model_name: str,
                  lambda_reg: float, ridge: float, skip_train: bool, skip_test: bool):
    """Process a single batch file through the full pipeline."""
    embedder = None
    projection_id = None

    try:
        # Load Q-A pairs
        db.update_batch_status(batch_id, 'importing', f'Loading {source_file}')
        qa_data = load_qa_pairs(source_file)
        num_clusters = len(qa_data['clusters'])
        print(f"  Found {num_clusters} clusters")

        # Import data
        clusters_info = import_to_database(db, qa_data, model_name)
        num_questions = sum(len(qids) for _, _, qids in clusters_info)
        db.update_batch_status(batch_id, 'importing',
                               f'Imported {num_clusters} clusters, {num_questions} questions',
                               num_clusters=num_clusters, num_questions=num_questions)

        # Embed
        db.update_batch_status(batch_id, 'embedding', 'Starting embedding')
        result = embed_all(db, model_name, clusters_info)
        if result is None:
            db.update_batch_status(batch_id, 'failed', error_message='Embedding failed: sentence-transformers not installed')
            return None, None
        model_id, embedder = result
        db.update_batch_status(batch_id, 'embedding', 'Embedding complete')

        # Train projection
        if not skip_train:
            db.update_batch_status(batch_id, 'training', 'Training W matrix')
            projection_id = train_projection(
                db, model_id, clusters_info,
                lambda_reg=lambda_reg,
                ridge=ridge
            )
            if projection_id:
                db.update_batch_status(batch_id, 'completed',
                                       f'Training complete, projection_id={projection_id}',
                                       projection_id=projection_id)
            else:
                db.update_batch_status(batch_id, 'failed', error_message='Training failed: no valid clusters')
                return embedder, None
        else:
            db.update_batch_status(batch_id, 'completed', 'Skipped training')

        # Test search
        if not skip_test and projection_id is not None:
            test_search(db, projection_id, embedder)

        return embedder, projection_id

    except Exception as e:
        db.update_batch_status(batch_id, 'failed', error_message=str(e))
        raise


def main():
    parser = argparse.ArgumentParser(description="Migrate Q-A pairs to LDA database")
    parser.add_argument('--input', default='playbooks/lda-training-data/raw/qa_pairs_v1.json',
                        help='Path to Q-A pairs JSON file (or directory with --scan)')
    parser.add_argument('--db', default='playbooks/lda-training-data/lda.db',
                        help='Output database path')
    parser.add_argument('--model', default='all-MiniLM-L6-v2',
                        help='Embedding model name')
    parser.add_argument('--lambda-reg', type=float, default=1.0,
                        help='Lambda regularization')
    parser.add_argument('--ridge', type=float, default=1e-6,
                        help='Ridge regularization')
    parser.add_argument('--skip-train', action='store_true',
                        help='Skip projection training')
    parser.add_argument('--skip-test', action='store_true',
                        help='Skip search test')
    parser.add_argument('--scan', action='store_true',
                        help='Scan directory for new/modified JSON files')
    parser.add_argument('--process-pending', action='store_true',
                        help='Process all pending batches')
    parser.add_argument('--retry-failed', action='store_true',
                        help='Retry failed batches')
    parser.add_argument('--list-batches', action='store_true',
                        help='List all batches and exit')
    args = parser.parse_args()

    print("=" * 60)
    print("LDA Database Migration")
    print("=" * 60)

    # Create database
    print(f"\nDatabase: {args.db}")
    db = LDAProjectionDB(args.db)

    # List batches mode
    if args.list_batches:
        batches = db.list_batches()
        if not batches:
            print("\nNo batches found.")
        else:
            print(f"\n{'ID':<4} {'Status':<12} {'Clusters':<8} {'File':<50}")
            print("-" * 80)
            for b in batches:
                clusters = b['num_clusters'] or '-'
                fname = Path(b['source_file']).name[:48]
                print(f"{b['batch_id']:<4} {b['status']:<12} {str(clusters):<8} {fname}")
        db.close()
        return 0

    # Scan mode: find new/modified files
    if args.scan:
        print(f"\nScanning directory: {args.input}")
        results = db.scan_for_new_batches(args.input, args.model)
        if not results:
            print("  No new or modified files found.")
        else:
            for batch_id, status in results:
                batch = db.get_batch(batch_id)
                print(f"  [{status}] {Path(batch['source_file']).name} -> batch_id={batch_id}")
        db.close()
        return 0

    # Get batches to process
    batches_to_process = []

    if args.process_pending:
        batches_to_process = db.get_pending_batches()
        print(f"\nFound {len(batches_to_process)} pending batches")

    elif args.retry_failed:
        batches_to_process = db.get_failed_batches()
        print(f"\nFound {len(batches_to_process)} failed batches to retry")
        # Reset status to pending for retry
        for b in batches_to_process:
            db.update_batch_status(b['batch_id'], 'pending', 'Retrying failed batch')

    else:
        # Single file mode
        print(f"\nRegistering batch: {args.input}")
        batch_id, status = db.register_batch(args.input, args.model)

        if status == 'unchanged':
            batch = db.get_batch(batch_id)
            if batch['status'] == 'completed':
                print(f"  File already processed (batch_id={batch_id}, status=completed)")
                print("  Use --scan or modify the file to reprocess")
                db.close()
                return 0
            else:
                print(f"  Batch exists but not completed (status={batch['status']})")
        else:
            print(f"  Registered as {status} batch: id={batch_id}")

        batches_to_process = [db.get_batch(batch_id)]

    # Process batches
    for batch in batches_to_process:
        batch_id = batch['batch_id']
        source_file = batch['source_file']
        print(f"\n{'='*60}")
        print(f"Processing batch {batch_id}: {Path(source_file).name}")
        print("=" * 60)

        embedder, projection_id = process_batch(
            db, batch_id, source_file, args.model,
            args.lambda_reg, args.ridge,
            args.skip_train, args.skip_test
        )

    # Print stats
    print("\n" + "=" * 60)
    print("Database Statistics")
    print("=" * 60)
    stats = db.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Print recent batches
    print("\nRecent Batches:")
    for b in db.list_batches()[:5]:
        print(f"  [{b['batch_id']}] {b['status']}: {Path(b['source_file']).name}")

    db.close()
    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
