"""
Multi-GPU encoding with manual distribution
"""
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer

def encode_on_gpu(gpu_id, texts, model_path):
    """Encode texts on specific GPU"""
    torch.cuda.set_device(gpu_id)
    model = SentenceTransformer(model_path, device=f'cuda:{gpu_id}')
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings

def multi_gpu_encode(texts, model_path, batch_size=64, num_gpus=None):
    """Encode texts using multiple GPUs"""
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    if num_gpus <= 1:
        # Fallback to single GPU
        model = SentenceTransformer(model_path, device='cuda:0')
        return model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
    
    print(f"🚀 Using {num_gpus} GPUs for encoding")
    
    # Split texts across GPUs
    chunk_size = len(texts) // num_gpus
    text_chunks = [texts[i*chunk_size:(i+1)*chunk_size] for i in range(num_gpus)]
    # Handle remainder
    if len(texts) % num_gpus != 0:
        text_chunks[-1].extend(texts[num_gpus*chunk_size:])
    
    # Encode in parallel
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = [
            executor.submit(encode_on_gpu, gpu_id, chunk, model_path)
            for gpu_id, chunk in enumerate(text_chunks)
        ]
        
        embeddings_list = [f.result() for f in futures]
    
    # Concatenate results
    all_embeddings = np.vstack(embeddings_list)
    return all_embeddings
