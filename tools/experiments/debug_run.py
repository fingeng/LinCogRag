import sys
sys.path.insert(0, '.')

print("=== Debug Mode ===")
print("Step 1: Importing modules...")

from sentence_transformers import SentenceTransformer
from src.config import LinearRAGConfig
from src.LinearRAG import LinearRAG
from src.llm import LLM
import os

print("Step 2: Setting up environment...")
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'test-key')
os.environ['OPENAI_BASE_URL'] = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')

print("Step 3: Loading embedding model...")
embedding_model = SentenceTransformer("model/all-mpnet-base-v2", device="cuda")

print("Step 4: Creating test data...")
# 创建少量测试数据
passages = [
    "0:This is a test passage about medicine.",
    "1:Another medical text about treatment.",
    "2:Research on drug efficacy."
]

print(f"Created {len(passages)} test passages")

print("Step 5: Initializing LLM...")
llm_model = LLM("gpt-4o-mini")

print("Step 6: Creating LinearRAG config...")
config = LinearRAGConfig(
    dataset_name="debug_test",
    embedding_model=embedding_model,
    spacy_model="en_core_sci_scibert",
    max_workers=2,
    llm_model=llm_model
)

print("Step 7: Initializing LinearRAG...")
rag_model = LinearRAG(global_config=config)

print("Step 8: Building index (this is where it might hang)...")
print("   - Starting index build...")
try:
    rag_model.index(passages)
    print("   ✅ Index built successfully!")
except Exception as e:
    print(f"   ❌ Error during index build: {e}")
    import traceback
    traceback.print_exc()

print("=== Debug Complete ===")
