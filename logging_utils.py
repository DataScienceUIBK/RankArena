# logging_utils.py
import json
import datetime
import os
import uuid
import hashlib
from pathlib import Path
from openai import AzureOpenAI

# Global storage for user sessions
user_sessions = {}
current_interactions = {}

# LLM Judge configuration
class LLMJudge:
    def __init__(self):
        self.endpoint = ""
        self.api_key = ""
        self.deployment = "gpt-4o-2"
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version="2024-05-01-preview",
        )
    
    def evaluate_reranking(self, query, model_a_info, model_b_info, reranked_a, reranked_b):
        """Evaluate which reranking is better using LLM judge"""
        
        # Format the reranked results for comparison
        def format_results(results, model_name):
            formatted = f"\n=== {model_name} Results ===\n"
            for i, ctx in enumerate(results, 1):
                formatted += f"{i}. {getattr(ctx, 'text', str(ctx))}\n\n"
            return formatted
        
        model_a_results = format_results(reranked_a, f"{model_a_info['method']}::{model_a_info['model']}")
        model_b_results = format_results(reranked_b, f"{model_b_info['method']}::{model_b_info['model']}")
        
        prompt = f"""You are an expert evaluator for document reranking systems. Given a query and two different reranking results, you need to determine which reranking is better. The documents are the same on both rerankers, but the order is important, so your mission is to evaluate based on the order of the rerank documents. 

Query: "{query}"

{model_a_results}

{model_b_results}

Please evaluate both reranking results based on:
1. Relevance to the query
2. Quality of ranking order
3. Overall usefulness for answering the query
4. You should evaluate the order of the reranker documents in each reranker.

Respond with ONLY ONE of these options:
- "Model A" if the first reranking is better
- "Model B" if the second reranking is better
- "Tie" if both are equally good

Then provide a brief explanation (2-3 sentences) of your reasoning.

Format your response as:
WINNER: [Model A/Model B/Tie]
REASONING: [Your explanation]

Please Answer as format, as we provide before dont generate outside this format
"""

        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator for information retrieval and document reranking systems."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse the response
            lines = result.split('\n')
            winner = "Unknown"
            reasoning = result
            
            for line in lines:
                if line.startswith("WINNER:"):
                    winner = line.replace("WINNER:", "").strip()
                elif line.startswith("REASONING:"):
                    reasoning = line.replace("REASONING:", "").strip()
            
            return {
                "winner": winner,
                "reasoning": reasoning,
                "full_response": result
            }
            
        except Exception as e:
            return {
                "winner": "Error",
                "reasoning": f"Failed to evaluate: {str(e)}",
                "full_response": f"Error: {str(e)}"
            }

# Initialize LLM Judge
llm_judge = LLMJudge()

def generate_user_id(request_info=None):
    """Generate a unique user ID based on session or IP"""
    # In a real application, you might use session ID, IP address, or other identifiers
    # For now, we'll generate a random UUID that persists during the session
    import gradio as gr
    
    # Try to get user info from Gradio request
    try:
        # This is a simple approach - in production you'd want more sophisticated user tracking
        user_key = str(uuid.uuid4())[:8]  # Short unique ID
        return user_key
    except:
        return str(uuid.uuid4())[:8]

def get_or_create_user_session():
    """Get or create a user session ID"""
    # In a real app, you'd get this from the request context
    # For now, we'll create a simple session management
    session_id = str(uuid.uuid4())[:8]
    return session_id

def ensure_user_folders(user_id):
    """Create folder structure for a user"""
    base_path = Path("user_data") / user_id
    folders = {
        'retrieve': base_path / "retrieve",
        'reranker': base_path / "reranker", 
        'votes': base_path / "votes",
        'votellm': base_path / "votellm",
        'rag': base_path / "rag",  # RAG folder for LLM answers
        'annotate': base_path / "annotate"  # 🆕 NEW: Annotation folder for manual rankings
    }
    
    for folder_path in folders.values():
        folder_path.mkdir(parents=True, exist_ok=True)
    
    return folders

def save_retrieval_interaction(user_id, query, retriever_config, retrieved_documents):
    """Save retrieval interaction for a specific user"""
    folders = ensure_user_folders(user_id)
    
    timestamp = datetime.datetime.now()
    interaction_id = hashlib.md5(f"{user_id}_{timestamp.isoformat()}_{query}".encode()).hexdigest()[:12]
    
    retrieval_data = {
        "interaction_id": interaction_id,
        "user_id": user_id,
        "timestamp": timestamp.isoformat(),
        "query": query,
        "retriever_config": retriever_config,
        "retrieved_documents": [
            {
                "id": getattr(ctx, 'id', f"doc_{i}"),
                "title": getattr(ctx, 'title', f"Document {i+1}"),
                "text": getattr(ctx, 'text', str(ctx)),
                "score": float(getattr(ctx, 'score', 0.0))
            }
            for i, ctx in enumerate(retrieved_documents)
        ],
        "num_documents": len(retrieved_documents)
    }
    
    # Save to retrieve folder
    filename = f"retrieve_{interaction_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    file_path = folders['retrieve'] / filename
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(retrieval_data, f, indent=2, ensure_ascii=False)
    
    # Store for reranking reference
    current_interactions[user_id] = {
        'interaction_id': interaction_id,
        'retrieval_data': retrieval_data,
        'timestamp': timestamp
    }
    
    print(f"✅ Saved retrieval for user {user_id}: {file_path}")
    return interaction_id

def save_reranking_interaction(user_id, query, method_a, model_a, method_b, model_b, 
                              reranked_contexts_a, reranked_contexts_b, original_contexts=None):
    """Save reranking interaction for a specific user"""
    folders = ensure_user_folders(user_id)
    
    timestamp = datetime.datetime.now()
    
    # Get the original retrieval interaction if available
    retrieval_ref = current_interactions.get(user_id, {})
    base_interaction_id = retrieval_ref.get('interaction_id', 'unknown')
    
    reranking_id = f"{base_interaction_id}_rerank_{hashlib.md5(f'{method_a}_{model_a}_{method_b}_{model_b}'.encode()).hexdigest()[:8]}"
    
    reranking_data = {
        "reranking_id": reranking_id,
        "base_interaction_id": base_interaction_id,
        "user_id": user_id,
        "timestamp": timestamp.isoformat(),
        "query": query,
        "models": {
            "model_a": {
                "method": method_a,
                "model": model_a,
                "full_key": f"{method_a}::{model_a}"
            },
            "model_b": {
                "method": method_b,
                "model": model_b,
                "full_key": f"{method_b}::{model_b}"
            }
        },
        "original_contexts": [
            {
                "id": getattr(ctx, 'id', f"orig_{i}"),
                "title": getattr(ctx, 'title', f"Original {i+1}"),
                "text": getattr(ctx, 'text', str(ctx)),
                "score": float(getattr(ctx, 'score', 0.0)),
                "original_rank": i + 1
            }
            for i, ctx in enumerate(original_contexts or [])
        ] if original_contexts else [],
        "reranked_results": {
            "model_a": [
                {
                    "id": getattr(ctx, 'id', f"a_{i}"),
                    "title": getattr(ctx, 'title', f"Model A {i+1}"),
                    "text": getattr(ctx, 'text', str(ctx)),
                    "score": float(getattr(ctx, 'score', 0.0)),
                    "new_rank": i + 1
                }
                for i, ctx in enumerate(reranked_contexts_a)
            ],
            "model_b": [
                {
                    "id": getattr(ctx, 'id', f"b_{i}"),
                    "title": getattr(ctx, 'title', f"Model B {i+1}"),
                    "text": getattr(ctx, 'text', str(ctx)),
                    "score": float(getattr(ctx, 'score', 0.0)),
                    "new_rank": i + 1
                }
                for i, ctx in enumerate(reranked_contexts_b)
            ]
        }
    }
    
    # Save to reranker folder
    filename = f"rerank_{reranking_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    file_path = folders['reranker'] / filename
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(reranking_data, f, indent=2, ensure_ascii=False)
    
    # Update current interaction for voting reference
    if user_id in current_interactions:
        current_interactions[user_id].update({
            'reranking_id': reranking_id,
            'reranking_data': reranking_data
        })
    else:
        current_interactions[user_id] = {
            'reranking_id': reranking_id,
            'reranking_data': reranking_data
        }
    
    print(f"✅ Saved reranking for user {user_id}: {file_path}")
    return reranking_id

def save_rag_interaction(user_id, query, reranking_id, method_a, model_a, method_b, model_b, 
                        llm_result_a, llm_result_b, reranked_contexts_a=None, reranked_contexts_b=None):
    """
    🆕 Save RAG interaction with LLM answers from both rerankers to user-specific folder
    
    Args:
        user_id: User session ID
        query: The original query
        reranking_id: ID from the reranking interaction
        method_a, model_a: First reranker configuration
        method_b, model_b: Second reranker configuration  
        llm_result_a: LLM answer result from first reranker (dict with answer/id/document)
        llm_result_b: LLM answer result from second reranker (dict with answer/id/document)
        reranked_contexts_a: Optional - reranked contexts from model A
        reranked_contexts_b: Optional - reranked contexts from model B
    
    Returns:
        str: Unique RAG interaction ID
    """
    
    folders = ensure_user_folders(user_id)
    
    timestamp = datetime.datetime.now()
    
    # Generate RAG interaction ID
    rag_interaction_id = f"{reranking_id}_rag_{hashlib.md5(f'{user_id}_{timestamp.isoformat()}'.encode()).hexdigest()[:8]}"
    
    # Calculate answer metrics
    def calculate_answer_metrics(llm_result):
        if llm_result.get("error"):
            return {
                "has_error": True,
                "error_message": llm_result["error"],
                "answer_length": 0,
                "source_document_id": None,
                "source_document_length": 0,
                "answer_word_count": 0
            }
        
        answer = llm_result.get("answer", "")
        document = llm_result.get("document", "")
        
        return {
            "has_error": False,
            "error_message": None,
            "answer_length": len(answer),
            "source_document_id": llm_result.get("id", "unknown"),
            "source_document_length": len(document),
            "answer_word_count": len(answer.split()) if answer else 0
        }
    
    metrics_a = calculate_answer_metrics(llm_result_a)
    metrics_b = calculate_answer_metrics(llm_result_b)
    
    # Prepare contexts data (convert Context objects to serializable format)
    def contexts_to_dict(contexts):
        if not contexts:
            return []
        
        contexts_data = []
        for ctx in contexts:
            contexts_data.append({
                "id": getattr(ctx, 'id', 'unknown'),
                "title": getattr(ctx, 'title', ''),
                "text": getattr(ctx, 'text', '')[:500],  # Truncate for storage
                "score": float(getattr(ctx, 'score', 0.0))
            })
        return contexts_data
    
    # Get current interaction data for reference
    current_interaction = current_interactions.get(user_id, {})
    
    # Create the RAG interaction record
    rag_interaction = {
        "rag_interaction_id": rag_interaction_id,
        "timestamp": timestamp.isoformat(),
        "user_id": user_id,
        "query": query,
        "reranking_id": reranking_id,
        "base_interaction_id": current_interaction.get('interaction_id', 'unknown'),
        
        # Reranker configurations
        "reranker_a": {
            "method": method_a,
            "model": model_a,
            "full_key": f"{method_a}::{model_a}"
        },
        "reranker_b": {
            "method": method_b,
            "model": model_b,
            "full_key": f"{method_b}::{model_b}"
        },
        
        # LLM Results (Full JSON responses)
        "llm_result_a": {
            "answer": llm_result_a.get("answer", ""),
            "source_document_id": llm_result_a.get("id", ""),
            "source_document_text": llm_result_a.get("document", ""),
            "has_error": bool(llm_result_a.get("error")),
            "error_message": llm_result_a.get("error", None)
        },
        "llm_result_b": {
            "answer": llm_result_b.get("answer", ""),
            "source_document_id": llm_result_b.get("id", ""),
            "source_document_text": llm_result_b.get("document", ""),
            "has_error": bool(llm_result_b.get("error")),
            "error_message": llm_result_b.get("error", None)
        },
        
        # Answer metrics and comparison
        "answer_metrics": {
            "model_a": metrics_a,
            "model_b": metrics_b,
            "comparison": {
                "both_successful": not metrics_a["has_error"] and not metrics_b["has_error"],
                "answer_length_diff": abs(metrics_a["answer_length"] - metrics_b["answer_length"]),
                "same_source_document": (
                    metrics_a["source_document_id"] == metrics_b["source_document_id"] 
                    if not metrics_a["has_error"] and not metrics_b["has_error"] else False
                ),
                "word_count_diff": abs(metrics_a.get("answer_word_count", 0) - metrics_b.get("answer_word_count", 0))
            }
        },
        
        # Optional: Include reranked contexts summary
        "reranked_contexts": {
            "model_a_contexts": contexts_to_dict(reranked_contexts_a),
            "model_b_contexts": contexts_to_dict(reranked_contexts_b),
            "context_counts": {
                "model_a": len(reranked_contexts_a) if reranked_contexts_a else 0,
                "model_b": len(reranked_contexts_b) if reranked_contexts_b else 0
            }
        },
        
        # LLM configuration used
        "llm_config": {
            "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            "provider": "Together AI",
            "temperature": 0.1,
            "max_tokens": 1000
        }
    }
    
    # Save to RAG folder
    filename = f"rag_{rag_interaction_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    file_path = folders['rag'] / filename
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(rag_interaction, f, indent=2, ensure_ascii=False)
        
        print(f"✅ RAG interaction saved for user {user_id}: {file_path}")
        print(f"📊 Answer lengths - Model A: {metrics_a['answer_length']}, Model B: {metrics_b['answer_length']}")
        
        # Print quick summary
        if metrics_a["has_error"] or metrics_b["has_error"]:
            print(f"⚠️ Errors detected - Model A: {metrics_a['has_error']}, Model B: {metrics_b['has_error']}")
        else:
            same_source = rag_interaction["answer_metrics"]["comparison"]["same_source_document"]
            print(f"📄 Same source document used: {same_source}")
        
        # Update current interaction for RAG reference
        if user_id in current_interactions:
            current_interactions[user_id].update({
                'rag_interaction_id': rag_interaction_id,
                'rag_data': rag_interaction
            })
            
    except Exception as e:
        print(f"❌ Error saving RAG interaction: {str(e)}")
        # Fallback: save with timestamp to avoid conflicts
        fallback_filename = f"rag_backup_{timestamp.strftime('%Y%m%d_%H%M%S')}_{rag_interaction_id}.json"
        fallback_path = folders['rag'] / fallback_filename
        try:
            with open(fallback_path, 'w', encoding='utf-8') as f:
                json.dump(rag_interaction, f, indent=2, ensure_ascii=False)
            print(f"💾 RAG interaction saved to backup file: {fallback_path}")
        except Exception as e2:
            print(f"❌ Critical error: Could not save RAG interaction: {str(e2)}")
    
    return rag_interaction_id

def save_llm_vote_interaction(user_id, llm_evaluation, user_vote=None):
    """Save LLM judge evaluation for a specific user"""
    folders = ensure_user_folders(user_id)
    
    timestamp = datetime.datetime.now()
    
    # Get the current interaction data
    current_interaction = current_interactions.get(user_id, {})
    
    llm_vote_data = {
        "llm_vote_id": f"llmvote_{hashlib.md5(f'{user_id}_{timestamp.isoformat()}'.encode()).hexdigest()[:12]}",
        "user_id": user_id,
        "timestamp": timestamp.isoformat(),
        "llm_evaluation": llm_evaluation,
        "user_vote": user_vote,
        "agreement": llm_evaluation.get("winner", "").lower() == user_vote.lower() if user_vote else None,
        "interaction_references": {
            "retrieval_id": current_interaction.get('interaction_id'),
            "reranking_id": current_interaction.get('reranking_id'),
            "rag_id": current_interaction.get('rag_interaction_id')  # 🆕 NEW: Include RAG reference
        }
    }
    
    # Include full context if available
    if 'retrieval_data' in current_interaction:
        llm_vote_data['retrieval_context'] = current_interaction['retrieval_data']
    
    if 'reranking_data' in current_interaction:
        llm_vote_data['reranking_context'] = current_interaction['reranking_data']
    
    if 'rag_data' in current_interaction:  # 🆕 NEW: Include RAG context
        llm_vote_data['rag_context'] = current_interaction['rag_data']
    
    # Save to votellm folder
    filename = f"llmvote_{llm_vote_data['llm_vote_id']}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    file_path = folders['votellm'] / filename
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(llm_vote_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved LLM vote for user {user_id}: {file_path}")
    return llm_vote_data['llm_vote_id']

def save_vote_interaction(user_id, winner, additional_feedback=None):
    """Save voting interaction for a specific user and trigger LLM evaluation"""
    folders = ensure_user_folders(user_id)
    
    timestamp = datetime.datetime.now()
    
    # Get the current interaction data
    current_interaction = current_interactions.get(user_id, {})
    
    vote_data = {
        "vote_id": f"vote_{hashlib.md5(f'{user_id}_{timestamp.isoformat()}'.encode()).hexdigest()[:12]}",
        "user_id": user_id,
        "timestamp": timestamp.isoformat(),
        "winner": winner,
        "additional_feedback": additional_feedback,
        "interaction_references": {
            "retrieval_id": current_interaction.get('interaction_id'),
            "reranking_id": current_interaction.get('reranking_id'),
            "rag_id": current_interaction.get('rag_interaction_id')  # 🆕 NEW: Include RAG reference
        }
    }
    
    # Include full context if available
    if 'retrieval_data' in current_interaction:
        vote_data['retrieval_context'] = current_interaction['retrieval_data']
    
    if 'reranking_data' in current_interaction:
        vote_data['reranking_context'] = current_interaction['reranking_data']
    
    if 'rag_data' in current_interaction:  # 🆕 NEW: Include RAG context
        vote_data['rag_context'] = current_interaction['rag_data']
    
    # Save to votes folder
    filename = f"vote_{vote_data['vote_id']}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    file_path = folders['votes'] / filename
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(vote_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved vote for user {user_id}: {file_path}")
    
    # Trigger LLM evaluation if we have reranking data
    llm_result = None
    if 'reranking_data' in current_interaction:
        try:
            reranking_data = current_interaction['reranking_data']
            query = reranking_data.get('query', '')
            
            # Get model info
            model_a_info = reranking_data['models']['model_a']
            model_b_info = reranking_data['models']['model_b']
            
            # Get reranked results
            reranked_a = reranking_data['reranked_results']['model_a']
            reranked_b = reranking_data['reranked_results']['model_b']
            
            # Convert to context-like objects for LLM evaluation
            class ContextForLLM:
                def __init__(self, data):
                    self.id = data.get('id', '')
                    self.title = data.get('title', '')
                    self.text = data.get('text', '')
                    self.score = data.get('score', 0.0)
            
            contexts_a = [ContextForLLM(ctx) for ctx in reranked_a]
            contexts_b = [ContextForLLM(ctx) for ctx in reranked_b]
            
            # Get LLM evaluation
            llm_evaluation = llm_judge.evaluate_reranking(
                query, model_a_info, model_b_info, contexts_a, contexts_b
            )
            
            # Save LLM evaluation
            llm_vote_id = save_llm_vote_interaction(user_id, llm_evaluation, winner)
            llm_result = llm_evaluation
            
            print(f"✅ LLM Judge evaluation completed: {llm_evaluation['winner']}")
            
        except Exception as e:
            print(f"❌ Error during LLM evaluation: {str(e)}")
            llm_result = {
                "winner": "Error",
                "reasoning": f"Failed to evaluate: {str(e)}",
                "full_response": f"Error: {str(e)}"
            }
    
    # Clean up current interaction after vote
    if user_id in current_interactions:
        del current_interactions[user_id]
    
    return vote_data['vote_id'], llm_result

def get_user_stats(user_id):
    """Get statistics for a specific user"""
    folders = ensure_user_folders(user_id)
    
    stats = {
        "user_id": user_id,
        "retrieval_count": len(list(folders['retrieve'].glob("*.json"))),
        "reranking_count": len(list(folders['reranker'].glob("*.json"))),
        "vote_count": len(list(folders['votes'].glob("*.json"))),
        "llm_vote_count": len(list(folders['votellm'].glob("*.json"))),
        "rag_count": len(list(folders['rag'].glob("*.json"))),  # 🆕 NEW: RAG count
        "folders": {
            "retrieve": str(folders['retrieve']),
            "reranker": str(folders['reranker']),
            "votes": str(folders['votes']),
            "votellm": str(folders['votellm']),
            "rag": str(folders['rag'])  # 🆕 NEW: RAG folder
        }
    }
    
    return stats

def get_rag_stats(user_id=None):
    """
    🆕 Get RAG interaction statistics for a specific user or globally
    
    Args:
        user_id: Optional - get stats for specific user, or None for global stats
        
    Returns:
        dict: Statistics about RAG interactions
    """
    
    stats = {
        "total_rag_interactions": 0,
        "successful_interactions": 0,
        "failed_interactions": 0,
        "total_answers_generated": 0,
        "average_answer_length": 0,
        "most_used_rerankers": {},
        "error_rate": 0
    }
    
    try:
        if user_id:
            # Get stats for specific user
            folders = ensure_user_folders(user_id)
            rag_files = list(folders['rag'].glob("*.json"))
        else:
            # Get global stats across all users
            rag_files = []
            base_path = Path("user_data")
            if base_path.exists():
                for user_folder in base_path.iterdir():
                    if user_folder.is_dir():
                        rag_folder = user_folder / "rag"
                        if rag_folder.exists():
                            rag_files.extend(rag_folder.glob("*.json"))
        
        all_interactions = []
        for rag_file in rag_files:
            try:
                with open(rag_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_interactions.append(data)
            except Exception as e:
                print(f"⚠️ Error reading {rag_file}: {str(e)}")
                continue
        
        stats["total_rag_interactions"] = len(all_interactions)
        
        if all_interactions:
            answer_lengths = []
            reranker_usage = {}
            successful_count = 0
            total_answers = 0
            
            for interaction in all_interactions:
                # Count successful interactions
                metrics = interaction.get("answer_metrics", {})
                comparison = metrics.get("comparison", {})
                if comparison.get("both_successful", False):
                    successful_count += 1
                
                # Track answer lengths
                model_a_metrics = metrics.get("model_a", {})
                model_b_metrics = metrics.get("model_b", {})
                
                if not model_a_metrics.get("has_error", True):
                    answer_lengths.append(model_a_metrics.get("answer_length", 0))
                    total_answers += 1
                    
                if not model_b_metrics.get("has_error", True):
                    answer_lengths.append(model_b_metrics.get("answer_length", 0))
                    total_answers += 1
                
                # Track reranker usage
                reranker_a = interaction.get("reranker_a", {})
                reranker_b = interaction.get("reranker_b", {})
                
                for reranker in [reranker_a, reranker_b]:
                    method = reranker.get("method", "unknown")
                    model = reranker.get("model", "unknown")
                    key = f"{method}::{model}"
                    reranker_usage[key] = reranker_usage.get(key, 0) + 1
            
            stats["successful_interactions"] = successful_count
            stats["failed_interactions"] = stats["total_rag_interactions"] - successful_count
            stats["total_answers_generated"] = total_answers
            stats["average_answer_length"] = sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0
            stats["most_used_rerankers"] = dict(sorted(reranker_usage.items(), key=lambda x: x[1], reverse=True)[:5])
            stats["error_rate"] = (stats["failed_interactions"] / stats["total_rag_interactions"]) * 100 if stats["total_rag_interactions"] > 0 else 0
    
    except Exception as e:
        print(f"❌ Error calculating RAG stats: {str(e)}")
    
    return stats

def cleanup_old_data(days_old=30):
    """Clean up data older than specified days"""
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_old)
    
    base_path = Path("user_data")
    if not base_path.exists():
        return
    
    cleaned_count = 0
    for user_folder in base_path.iterdir():
        if user_folder.is_dir():
            for subfolder in user_folder.iterdir():
                if subfolder.is_dir():
                    for file_path in subfolder.glob("*.json"):
                        if file_path.stat().st_mtime < cutoff_date.timestamp():
                            file_path.unlink()
                            cleaned_count += 1
    
    print(f"🧹 Cleaned up {cleaned_count} old files")
    return cleaned_count

# Legacy functions for backward compatibility
def save_interaction(query, documents, model_a, model_b, result_a, result_b, vote=None):
    """Legacy function - now redirects to user-specific functions"""
    user_id = get_or_create_user_session()
    
    if vote is not None:
        return save_vote_interaction(user_id, vote)
    else:
        # This is called from reranker - save as reranking interaction
        # We need to parse the model keys
        method_a, model_name_a = model_a.split("::", 1) if "::" in model_a else (model_a, "unknown")
        method_b, model_name_b = model_b.split("::", 1) if "::" in model_b else (model_b, "unknown")
        
        return save_reranking_interaction(
            user_id, query, method_a, model_name_a, method_b, model_name_b,
            result_a, result_b
        )