"""
Web UI for Agent-Based RAG System
Streamlit-based interface for testing and demonstration
"""


import plotly.express as px
import plotly.graph_objects as go

import os
import sys
from pathlib import Path

if __name__ == "__main__":
    current_dir = Path(__file__).parent
    src_dir = current_dir.parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

import streamlit as st
import requests
import json
import sqlite3
import pandas as pd
from datetime import datetime

# Get configuration
try:
    from agent_rag.config import get_config
    config = get_config()
    ORCHESTRATOR_URL = "http://localhost:8001"
    GEMINI_URL = "http://localhost:8000"
    EMBEDDING_URL = "http://localhost:8002"
    DB_PATH = config["db_path"]
except ImportError:
    ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8001")
    GEMINI_URL = os.getenv("GEMINI_URL", "http://localhost:8000")
    EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://localhost:8002")
    DB_PATH = os.getenv("DB_PATH", "rag_index.db")

# Page configuration
st.set_page_config(
    page_title="Agent RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stAlert {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Configuration
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8001")
GEMINI_URL = os.getenv("GEMINI_URL", "http://localhost:8000")
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://localhost:8002")
DB_PATH = os.getenv("DB_PATH", "rag_index.db")

# Initialize session state
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'current_result' not in st.session_state:
    st.session_state.current_result = None
if 'shards' not in st.session_state:
    st.session_state.shards = []

def check_service_health(url, name):
    """Check if a service is healthy"""
    try:
        response = requests.get(f"{url}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def process_query(query, shards=None):
    """Send query to orchestrator"""
    try:
        payload = {
            "query": query,
            "shards": shards or []
        }
        response = requests.post(
            f"{ORCHESTRATOR_URL}/query",
            json=payload,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Server returned {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def get_database_stats():
    """Get statistics from the database"""
    if not Path(DB_PATH).exists():
        return None
    
    conn = sqlite3.connect(DB_PATH)
    
    # Get chunk counts
    chunk_stats = pd.read_sql_query("""
        SELECT 
            chunk_type,
            COUNT(*) as count,
            AVG(token_count) as avg_tokens
        FROM chunks
        GROUP BY chunk_type
    """, conn)
    
    # Get recent queries
    recent_queries = pd.read_sql_query("""
        SELECT 
            timestamp,
            query,
            model
        FROM answers
        ORDER BY timestamp DESC
        LIMIT 10
    """, conn)
    
    conn.close()
    
    return {
        "chunks": chunk_stats,
        "recent": recent_queries
    }

def semantic_search(query, top_k=5):
    """Perform semantic search using embedding service"""
    try:
        response = requests.post(
            f"{EMBEDDING_URL}/search",
            json={"query": query, "top_k": top_k}
        )
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

# Sidebar
with st.sidebar:
    st.title("🤖 Agent RAG System")
    
    # Service Status
    st.subheader("Service Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if check_service_health(ORCHESTRATOR_URL, "Orchestrator"):
            st.success("✅ Orchestrator")
        else:
            st.error("❌ Orchestrator")
    
    with col2:
        if check_service_health(GEMINI_URL, "Gemini"):
            st.success("✅ Gemini")
        else:
            st.error("❌ Gemini")
    
    with col3:
        if check_service_health(EMBEDDING_URL, "Embeddings"):
            st.success("✅ Embeddings")
        else:
            st.warning("⚠️ Embeddings")
    
    # Configuration
    st.subheader("Configuration")
    
    use_shards = st.checkbox("Use Custom Shards", value=False)
    
    if use_shards:
        num_shards = st.number_input("Number of Shards", min_value=1, max_value=5, value=2)
        
        shards = []
        for i in range(num_shards):
            with st.expander(f"Shard {i+1}"):
                shard_name = st.text_input(f"Name", key=f"shard_name_{i}", value=f"shard_{i+1}")
                shard_docs = st.text_area(
                    f"Documents (one per line)", 
                    key=f"shard_docs_{i}",
                    height=100
                ).split('\n')
                
                if shard_name and shard_docs:
                    shards.append({
                        "shard_name": shard_name,
                        "shard_docs": [doc.strip() for doc in shard_docs if doc.strip()]
                    })
        
        st.session_state.shards = shards
    
    # Query Settings
    st.subheader("Query Settings")
    max_tokens = st.slider("Max Response Tokens", 500, 4000, 2000)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.5)
    
    # Database Stats
    st.subheader("Database Statistics")
    stats = get_database_stats()
    if stats:
        if not stats["chunks"].empty:
            for _, row in stats["chunks"].iterrows():
                st.metric(
                    f"{row['chunk_type'].title()} Chunks",
                    int(row['count']),
                    f"~{int(row['avg_tokens'])} tokens avg"
                )

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["💬 Query", "📊 Analytics", "🔍 Search Test", "📚 Documentation"])

with tab1:
    st.header("Query Interface")
    
    # Query input
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_area(
            "Enter your question:",
            placeholder="How does the authentication system work?",
            height=100
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")
        if st.button("🚀 Submit", type="primary", use_container_width=True):
            if query:
                with st.spinner("Processing query..."):
                    result = process_query(
                        query, 
                        st.session_state.shards if use_shards else None
                    )
                    st.session_state.current_result = result
                    st.session_state.query_history.append({
                        "timestamp": datetime.now(),
                        "query": query,
                        "result": result
                    })
    
    # Display result
    if st.session_state.current_result:
        result = st.session_state.current_result
        
        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            # Answer section
            st.subheader("Answer")
            answer_container = st.container()
            with answer_container:
                st.markdown(result.get("answer", "No answer provided"))
            
            # Metadata
            with st.expander("📋 Metadata"):
                metadata = result.get("metadata", {})
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Model", metadata.get("model", "Unknown"))
                with col2:
                    st.metric("Answer ID", metadata.get("answer_id", "N/A")[:8])
                with col3:
                    st.metric("Timestamp", metadata.get("timestamp", "N/A")[:19])
            
            # Master Prompt
            with st.expander("🔧 Master Prompt (Synthesized Context)"):
                st.text_area(
                    "Master Prompt",
                    value=result.get("master_prompt", "Not available"),
                    height=200,
                    disabled=True
                )
            
            # Download results
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📥 Download Answer",
                    data=result.get("answer", ""),
                    file_name=f"answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
            with col2:
                st.download_button(
                    "📥 Download Full Result",
                    data=json.dumps(result, indent=2),
                    file_name=f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

with tab2:
    st.header("Analytics Dashboard")
    
    if st.session_state.query_history:
        # Query timeline
        st.subheader("Query Timeline")
        df = pd.DataFrame([
            {
                "timestamp": q["timestamp"],
                "query": q["query"][:50] + "...",
                "has_error": "error" in q["result"]
            }
            for q in st.session_state.query_history
        ])
        
        fig = px.scatter(
            df,
            x="timestamp",
            y="query",
            color="has_error",
            title="Query History",
            labels={"has_error": "Error Status"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Cost estimation
        st.subheader("Cost Analysis")
        
        # Estimate costs (simplified)
        total_queries = len(st.session_state.query_history)
        estimated_cost = total_queries * 0.08  # Average cost per query
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Queries", total_queries)
        with col2:
            st.metric("Estimated Cost", f"${estimated_cost:.2f}")
        with col3:
            st.metric("Avg Cost/Query", f"${estimated_cost/max(total_queries, 1):.3f}")
        
        # Token usage chart (mock data for demonstration)
        token_data = {
            "Stage": ["Retrieval", "Synthesis", "Reasoning"],
            "Tokens": [3000, 1500, 500]
        }
        
        fig = go.Figure(data=[
            go.Bar(x=token_data["Stage"], y=token_data["Tokens"])
        ])
        fig.update_layout(title="Token Usage by Stage")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No queries processed yet. Submit a query to see analytics.")

with tab3:
    st.header("Semantic Search Test")
    
    search_query = st.text_input("Search Query", placeholder="authentication flow")
    search_button = st.button("🔍 Search")
    
    if search_button and search_query:
        with st.spinner("Searching..."):
            results = semantic_search(search_query)
        
        if results:
            st.success(f"Found {results['count']} results")
            
            for i, result in enumerate(results.get("results", []), 1):
                with st.expander(f"Result {i} - Score: {result['similarity_score']:.3f}"):
                    st.write(f"**Chunk ID:** {result['chunk_id']}")
                    st.write(f"**Type:** {result['chunk_type']}")
                    st.write(f"**Source:** {result['source_file']}")
                    st.write(f"**Text:**")
                    st.text(result['text'][:500] + "..." if len(result['text']) > 500 else result['text'])
        else:
            st.warning("Semantic search service not available or no results found")

with tab4:
    st.header("Documentation")
    
    st.markdown("""
    ## Quick Start Guide
    
    ### 1. Basic Query
    Simply enter your question in the Query tab and click Submit. The system will:
    - Use Perplexity for global search if no shards are configured
    - Synthesize results using GPT
    - Generate a final answer using Claude
    
    ### 2. Using Custom Shards
    Enable "Use Custom Shards" in the sidebar to manually specify document chunks:
    - Add shard names and documents
    - The system will use Gemini Flash to search within these shards
    - Results are synthesized before final reasoning
    
    ### 3. Understanding the Pipeline
    ```
    Query → Shard Retrieval → Global Search → Synthesis → Reasoning → Answer
    ```
    
    ### Cost Breakdown
    - **Gemini Flash**: ~$0.01 per query (retrieval)
    - **Perplexity**: ~$0.02 per query (synthesis)
    - **Claude**: ~$0.05 per query (reasoning)
    - **Total**: ~$0.08 per complex query
    
    ### API Endpoints
    - **Orchestrator**: `POST /query` - Main query endpoint
    - **Gemini**: `POST /gemini-flash-retrieve` - Shard retrieval
    - **Embeddings**: `POST /search` - Semantic search
    
    ### Tips for Better Results
    1. Be specific in your queries
    2. Use custom shards for domain-specific content
    3. Check service status before querying
    4. Monitor costs in the Analytics tab
    
    ### Troubleshooting
    - **Red service indicators**: Check that services are running
    - **Slow responses**: Reduce shard count or chunk sizes
    - **No results**: Verify documents are properly indexed
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Agent-Based RAG System | Built with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)