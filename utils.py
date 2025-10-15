from rankify.utils.pre_defind_models import HF_PRE_DEFIND_MODELS
from rankify.models.reranking import Reranking
from rankify.dataset.dataset import Question, Context, Document
import torch
import gc
import os
import subprocess
import psutil

def clear_gpu_memory():
    """
    Comprehensive GPU memory clearing function
    """
    print("🧹 Clearing GPU memory...")
    
    # 1. Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✅ PyTorch CUDA cache cleared")
        
        # 2. Reset CUDA memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        print("✅ CUDA memory stats reset")
        
        # 3. Show current GPU memory usage
        show_gpu_memory()
    
    # 4. Force Python garbage collection
    gc.collect()
    print("✅ Python garbage collection completed")
def show_gpu_memory():
    """
    Display current GPU memory usage
    """
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        cached = torch.cuda.memory_reserved(device) / 1024**3      # GB
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
        
        print(f"📊 GPU Memory Status:")
        print(f"   Device: {torch.cuda.get_device_name(device)}")
        print(f"   Allocated: {allocated:.2f} GB")
        print(f"   Cached: {cached:.2f} GB") 
        print(f"   Total: {total:.2f} GB")
        print(f"   Free: {total - cached:.2f} GB")
    else:
        print("❌ CUDA not available")

def get_rankify_methods():
    return list(HF_PRE_DEFIND_MODELS.keys())

def get_models_for_method(method):
    return list(HF_PRE_DEFIND_MODELS.get(method, {}).keys())

def run_rankify_reranker(query, doc_texts, method, model_name):
    clear_gpu_memory()
    print(model_name)
    model_name= HF_PRE_DEFIND_MODELS.get(method, {}).get(model_name)
    print(model_name)
    #aaaaaaaaaaaaaaaaa
    question = Question(query)
    contexts = [Context(text=text, id= str(i), score=1) for i, text in enumerate(doc_texts) if text.strip()]
    document = Document(question=question, contexts=contexts, answers=[])
    if not document.contexts:
        return ["⚠️ No valid documents provided."]
    reranker = Reranking(method=method, model_name=model_name)
    results = reranker.rank([document])
    
    #print(results)
    return [ctx.text for ctx in results[0].reorder_contexts]

def run_rankify_reranker_from_retriever(document, method, model_name):
    if not document.contexts:
        return ["⚠️ No valid documents provided."]
    reranker = Reranking(method=method, model_name=model_name)
    results = reranker.rank([document])
    return  results[0].reorder_contexts