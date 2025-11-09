#!/usr/bin/env python3
"""
Test Automation Scripts for Agent-Based RAG System
Place in: src/agent_rag/tests/progressive_tests.py
"""

import os
import sys
import json
import time
import asyncio
import argparse
import subprocess
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

@dataclass
class TestResult:
    """Store test execution results"""
    test_name: str
    phase: str
    status: str  # PASS, FAIL, SKIP
    duration: float
    cost: float
    error: Optional[str] = None
    metrics: Optional[Dict] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class TestRunner:
    """Progressive test runner for RAG system"""
    
    def __init__(self, config_path: str = ".env"):
        self.config_path = config_path
        self.results: List[TestResult] = []
        self.original_env = os.environ.copy()
        
    def set_env(self, **kwargs):
        """Set environment variables for test"""
        for key, value in kwargs.items():
            os.environ[key] = str(value)
    
    def reset_env(self):
        """Reset to original environment"""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def run_command(self, cmd: str, timeout: int = 30) -> Tuple[bool, str, float]:
        """Execute command and return success, output, duration"""
        start = time.time()
        try:
            result = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=timeout
            )
            duration = time.time() - start
            success = result.returncode == 0
            output = result.stdout if success else result.stderr
            return success, output, duration
        except subprocess.TimeoutExpired:
            return False, "Command timed out", timeout
        except Exception as e:
            return False, str(e), time.time() - start
    
    async def test_local_model(self) -> TestResult:
        """Test 1.1: Phi-3 Mini Standalone"""
        print("\n🧪 Testing Phi-3 Mini Standalone...")
        
        # Check if ollama is installed
        success, output, duration = self.run_command("ollama list | grep phi3")
        
        if not success:
            print("  ⚠️  Phi-3 not found. Pulling model...")
            success, output, duration = self.run_command("ollama pull phi3", timeout=300)
            
        if success:
            # Test inference
            success, output, duration = self.run_command(
                'ollama run phi3 "Explain JWT in one sentence"'
            )
            
        return TestResult(
            test_name="Phi-3 Mini Standalone",
            phase="Phase 1",
            status="PASS" if success else "FAIL",
            duration=duration,
            cost=0.0,
            error=None if success else output,
            metrics={"model": "phi3", "provider": "ollama"}
        )
    
    async def test_local_retrieval(self) -> TestResult:
        """Test 1.2: Local Retrieval Pipeline"""
        print("\n🧪 Testing Local Retrieval Pipeline...")
        
        self.set_env(
            RETRIEVER_PROVIDER="local",
            LOCAL_RETRIEVAL_MODEL="phi3",
            USE_LOCAL_MODELS="true",
            LOCAL_MODEL_PROVIDER="ollama",
            ENABLE_CODE_STORAGE="false"
        )
        
        test_script = """
import sys
sys.path.append('src/agent_rag/core')
from local_model_integration import LocalModelManager

manager = LocalModelManager({"local_model_provider": "ollama"})
result = manager.test_retrieval("How does auth work?", ["Sample code chunk"])
print(f"Success: {result is not None}")
"""
        
        with open("/tmp/test_local_retrieval.py", "w") as f:
            f.write(test_script)
        
        success, output, duration = self.run_command(
            "python /tmp/test_local_retrieval.py"
        )
        
        self.reset_env()
        
        return TestResult(
            test_name="Local Retrieval Pipeline",
            phase="Phase 1",
            status="PASS" if success else "FAIL",
            duration=duration,
            cost=0.0,
            error=None if success else output
        )
    
    async def test_gemini_retrieval(self, api_key: str) -> TestResult:
        """Test 2.1: Gemini Retrieval Only"""
        print("\n🧪 Testing Gemini Retrieval...")
        
        self.set_env(
            RETRIEVER_PROVIDER="gemini",
            RETRIEVER_API_KEY=api_key,
            RETRIEVER_MODEL="gemini-2.0-flash-exp",
            SYNTH_PROVIDER="local",
            REASON_PROVIDER="local"
        )
        
        test_script = f"""
import requests
import json

# Test Gemini retriever endpoint
response = requests.post(
    "http://localhost:8000/retrieve",
    json={{
        "query": "What is authentication?",
        "shard_index": 0,
        "shard_content": "Authentication is the process of verifying identity."
    }},
    headers={{"X-API-Key": "{api_key}"}}
)

print(f"Status: {{response.status_code}}")
print(f"Response: {{response.json() if response.ok else response.text}}")
"""
        
        with open("/tmp/test_gemini.py", "w") as f:
            f.write(test_script)
        
        # Start Gemini retriever service
        proc = subprocess.Popen(
            ["python", "src/agent_rag/core/gemini_retriever.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        time.sleep(5)  # Wait for service to start
        
        success, output, duration = self.run_command("python /tmp/test_gemini.py")
        
        proc.terminate()
        self.reset_env()
        
        # Estimate cost (Gemini Flash: ~$0.00001 per query)
        cost = 0.00001 if success else 0.0
        
        return TestResult(
            test_name="Gemini Retrieval",
            phase="Phase 2",
            status="PASS" if success else "FAIL",
            duration=duration,
            cost=cost,
            error=None if success else output,
            metrics={"provider": "gemini", "model": "gemini-2.0-flash-exp"}
        )
    
    async def test_full_pipeline(self, config: Dict) -> TestResult:
        """Test complete pipeline with specified configuration"""
        print(f"\n🧪 Testing Full Pipeline: {config['name']}...")
        
        for key, value in config['env'].items():
            self.set_env(**{key: value})
        
        # Build test query
        test_query = {
            "query": "Explain how JWT authentication works with code examples",
            "config": {
                "retriever": config['env'].get('RETRIEVER_PROVIDER'),
                "synthesizer": config['env'].get('SYNTH_PROVIDER'),
                "reasoner": config['env'].get('REASON_PROVIDER')
            }
        }
        
        test_script = f"""
import requests
import json
import time

start = time.time()

response = requests.post(
    "http://localhost:8001/rag_query",
    json={json.dumps(test_query)},
    timeout=60
)

duration = time.time() - start

result = {{
    "success": response.ok,
    "status_code": response.status_code,
    "duration": duration,
    "response_size": len(response.text) if response.ok else 0
}}

print(json.dumps(result))
"""
        
        with open("/tmp/test_pipeline.py", "w") as f:
            f.write(test_script)
        
        success, output, duration = self.run_command(
            "python /tmp/test_pipeline.py", 
            timeout=90
        )
        
        self.reset_env()
        
        # Parse results
        try:
            if success:
                result_data = json.loads(output)
                success = result_data.get('success', False)
                duration = result_data.get('duration', duration)
        except:
            pass
        
        return TestResult(
            test_name=config['name'],
            phase=config['phase'],
            status="PASS" if success else "FAIL",
            duration=duration,
            cost=config.get('estimated_cost', 0.0),
            error=None if success else output,
            metrics=config['env']
        )
    
    async def test_ui(self) -> TestResult:
        """Test 5.1: Web UI"""
        print("\n🧪 Testing Web UI...")
        
        # Start Streamlit
        proc = subprocess.Popen(
            ["streamlit", "run", "src/agent_rag/core/web_ui.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        time.sleep(10)  # Wait for Streamlit to start
        
        # Test if UI is accessible
        success, output, duration = self.run_command(
            "curl -s -o /dev/null -w '%{http_code}' http://localhost:8501"
        )
        
        proc.terminate()
        
        http_code = output.strip()
        success = http_code == "200"
        
        return TestResult(
            test_name="Web UI",
            phase="Phase 5",
            status="PASS" if success else "FAIL",
            duration=duration,
            cost=0.0,
            error=None if success else f"HTTP {http_code}",
            metrics={"http_code": http_code}
        )
    
    async def test_code_storage(self) -> TestResult:
        """Test 5.3: Code Storage Pipeline"""
        print("\n🧪 Testing Code Storage Pipeline...")
        
        self.set_env(
            ENABLE_CODE_STORAGE="true",
            CODE_STORAGE_DB="test_code_storage.db"
        )
        
        test_script = """
import sys
import os
sys.path.append('src/agent_rag/core')
from code_storage import CodeRetrievalPipeline

# Test storage pipeline
config = {
    "code_storage_db": "test_code_storage.db",
    "dedup_threshold": 0.95,
    "max_chunk_tokens": 4000,
    "chunk_overlap": 0.1
}

pipeline = CodeRetrievalPipeline(config)

# Test content hashing
hash1 = pipeline.compute_content_hash("def test(): pass")
hash2 = pipeline.compute_content_hash("def test():  pass")  # Extra space
hash3 = pipeline.compute_content_hash("def test2(): pass")

# Test deduplication
success = hash1 == hash2 and hash1 != hash3

# Cleanup
if os.path.exists("test_code_storage.db"):
    os.remove("test_code_storage.db")

print(f"Success: {success}")
"""
        
        with open("/tmp/test_storage.py", "w") as f:
            f.write(test_script)
        
        success, output, duration = self.run_command("python /tmp/test_storage.py")
        
        self.reset_env()
        
        return TestResult(
            test_name="Code Storage Pipeline",
            phase="Phase 5",
            status="PASS" if success and "Success: True" in output else "FAIL",
            duration=duration,
            cost=0.0,
            error=None if success else output
        )
    
    async def run_phase_1(self):
        """Run all Phase 1 tests (Local Only)"""
        print("\n" + "="*50)
        print("PHASE 1: LOCAL-ONLY TESTING")
        print("="*50)
        
        tests = [
            self.test_local_model(),
            self.test_local_retrieval()
        ]
        
        results = await asyncio.gather(*tests)
        self.results.extend(results)
        
        return results
    
    async def run_phase_2(self, gemini_key: str):
        """Run all Phase 2 tests (Gemini)"""
        print("\n" + "="*50)
        print("PHASE 2: GEMINI TESTING")
        print("="*50)
        
        results = [
            await self.test_gemini_retrieval(gemini_key),
            await self.test_full_pipeline({
                "name": "Full Gemini Pipeline",
                "phase": "Phase 2",
                "estimated_cost": 0.01,
                "env": {
                    "RETRIEVER_PROVIDER": "gemini",
                    "SYNTH_PROVIDER": "gemini",
                    "REASON_PROVIDER": "gemini",
                    "RETRIEVER_API_KEY": gemini_key,
                    "SYNTH_API_KEY": gemini_key,
                    "REASON_API_KEY": gemini_key
                }
            })
        ]
        
        self.results.extend(results)
        return results
    
    def generate_report(self):
        """Generate test execution report"""
        print("\n" + "="*50)
        print("TEST EXECUTION REPORT")
        print("="*50)
        
        # Summary by phase
        phases = {}
        for result in self.results:
            if result.phase not in phases:
                phases[result.phase] = {"pass": 0, "fail": 0, "skip": 0, "cost": 0.0}
            
            phases[result.phase][result.status.lower()] += 1
            phases[result.phase]["cost"] += result.cost
        
        print("\n📊 Summary by Phase:")
        for phase, stats in sorted(phases.items()):
            total = stats["pass"] + stats["fail"] + stats["skip"]
            pass_rate = (stats["pass"] / total * 100) if total > 0 else 0
            print(f"  {phase}:")
            print(f"    ✅ Pass: {stats['pass']}/{total} ({pass_rate:.1f}%)")
            print(f"    💰 Cost: ${stats['cost']:.4f}")
        
        # Detailed results
        print("\n📋 Detailed Results:")
        for result in self.results:
            icon = "✅" if result.status == "PASS" else "❌"
            print(f"  {icon} {result.test_name}")
            print(f"     Duration: {result.duration:.2f}s")
            print(f"     Cost: ${result.cost:.4f}")
            if result.error:
                print(f"     Error: {result.error[:100]}...")
        
        # Save to JSON
        report_path = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        
        print(f"\n💾 Report saved to: {report_path}")
        
        # Overall status
        total_pass = sum(1 for r in self.results if r.status == "PASS")
        total_tests = len(self.results)
        overall_pass_rate = (total_pass / total_tests * 100) if total_tests > 0 else 0
        
        print("\n" + "="*50)
        if overall_pass_rate >= 80:
            print(f"✅ OVERALL: PASS ({overall_pass_rate:.1f}%)")
        else:
            print(f"❌ OVERALL: FAIL ({overall_pass_rate:.1f}%)")
        print("="*50)


async def main():
    """Main test execution"""
    parser = argparse.ArgumentParser(description="Progressive RAG System Testing")
    parser.add_argument("--phase", type=int, help="Run specific phase (1-6)")
    parser.add_argument("--gemini-key", help="Gemini API key for Phase 2+")
    parser.add_argument("--perplexity-key", help="Perplexity API key for Phase 3+")
    parser.add_argument("--anthropic-key", help="Anthropic API key for Phase 4+")
    parser.add_argument("--all", action="store_true", help="Run all phases")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    try:
        if args.phase == 1 or args.all:
            await runner.run_phase_1()
        
        if (args.phase == 2 or args.all) and args.gemini_key:
            await runner.run_phase_2(args.gemini_key)
        
        # Add more phases as needed...
        
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
    finally:
        runner.generate_report()


if __name__ == "__main__":
    asyncio.run(main())