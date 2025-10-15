# RAG Tab - Retriever + Dual Reranker + LLM Comparison
import gradio as gr
import json
import time
from utils import get_rankify_methods, get_models_for_method, run_rankify_reranker_from_retriever
from logging_utils import (
    save_retrieval_interaction, 
    save_reranking_interaction, 
    save_vote_interaction,
    save_llm_vote_interaction,
    save_rag_interaction,  # NEW: Save RAG answers
    get_or_create_user_session,
    get_user_stats,
    get_rag_stats,  # NEW: Get RAG statistics
    llm_judge
)
from rankify.dataset.dataset import Document, Question, Answer, Context

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")

# Together AI import
try:
    from together import Together
    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False
    print("⚠️ Together AI not available. Please install: pip install together")

# Global storage for current RAG session data
current_rag_session = {}
# Global storage for LLM evaluations (temporary until user votes)
pending_llm_evaluations = {}
# Global storage to track which queries have been voted on by each user
voted_queries = {}

def compare_rankings_from_contexts(original_contexts, reranked_contexts_a, reranked_contexts_b):
    """Compare two reranking results using statistical methods - adapted for Context objects"""
    if not original_contexts or not reranked_contexts_a or not reranked_contexts_b:
        return "❌ No data to compare"
    
    # Extract document texts for comparison
    original_docs = [ctx.text for ctx in original_contexts]
    reranked_docs_a = [ctx.text for ctx in reranked_contexts_a]
    reranked_docs_b = [ctx.text for ctx in reranked_contexts_b]
    
    # Get positions in reranked results
    a_positions = []
    b_positions = []
    
    for doc in original_docs:
        # Find position in reranked A
        try:
            pos_a = reranked_docs_a.index(doc)
            a_positions.append(pos_a + 1)  # 1-indexed for better interpretation
        except ValueError:
            a_positions.append(len(original_docs) + 1)  # Put unranked docs at the end
        
        # Find position in reranked B
        try:
            pos_b = reranked_docs_b.index(doc)
            b_positions.append(pos_b + 1)  # 1-indexed for better interpretation
        except ValueError:
            b_positions.append(len(original_docs) + 1)  # Put unranked docs at the end
    
    # Calculate statistical measures
    try:
        from scipy.stats import spearmanr, kendalltau
        spearman_corr, spearman_p = spearmanr(a_positions, b_positions)
        kendall_corr, kendall_p = kendalltau(a_positions, b_positions)
        scipy_available = True
    except ImportError:
        # Fallback to simple correlation if scipy not available
        def simple_correlation(x, y):
            n = len(x)
            if n == 0:
                return 0, 1
            
            # Calculate Pearson correlation manually
            mean_x, mean_y = sum(x) / n, sum(y) / n
            num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
            den_x = sum((x[i] - mean_x) ** 2 for i in range(n))
            den_y = sum((y[i] - mean_y) ** 2 for i in range(n))
            
            if den_x == 0 or den_y == 0:
                return 0, 1
            
            corr = num / (den_x * den_y) ** 0.5
            return corr, 0.05  # Placeholder p-value
        
        spearman_corr, spearman_p = simple_correlation(a_positions, b_positions)
        kendall_corr, kendall_p = simple_correlation(a_positions, b_positions)
        scipy_available = False
    
    # Calculate overlap metrics
    def rank_biased_overlap(list1, list2, p=0.9):
        """Calculate Rank-Biased Overlap (RBO) between two rankings"""
        overlap_at_k = []
        for k in range(1, min(len(list1), len(list2)) + 1):
            set1 = set(list1[:k])
            set2 = set(list2[:k])
            overlap = len(set1.intersection(set2)) / k
            overlap_at_k.append(overlap)
        
        if not overlap_at_k:
            return 0
        
        # Weight by position (geometric decay)
        rbo = sum((1 - p) * (p ** (i)) * overlap for i, overlap in enumerate(overlap_at_k))
        return rbo
    
    rbo_score = rank_biased_overlap(reranked_docs_a, reranked_docs_b)
    
    # Calculate position-based metrics
    agreements = sum(1 for i in range(len(original_docs)) if a_positions[i] == b_positions[i])
    total_docs = len(original_docs)
    agreement_rate = (agreements / total_docs * 100) if total_docs > 0 else 0
    
    # Calculate average rank difference
    rank_differences = [abs(a_positions[i] - b_positions[i]) for i in range(len(original_docs))]
    avg_rank_diff = sum(rank_differences) / len(rank_differences) if rank_differences else 0
    max_possible_diff = total_docs
    normalized_rank_diff = (max_possible_diff - avg_rank_diff) / max_possible_diff * 100
    
    # Calculate top-k overlaps
    def top_k_overlap(list1, list2, k):
        if k > min(len(list1), len(list2)):
            k = min(len(list1), len(list2))
        set1 = set(list1[:k])
        set2 = set(list2[:k])
        return len(set1.intersection(set2)) / k * 100 if k > 0 else 0
    
    top1_overlap = top_k_overlap(reranked_docs_a, reranked_docs_b, 1)
    top3_overlap = top_k_overlap(reranked_docs_a, reranked_docs_b, 3) if total_docs >= 3 else 0
    top5_overlap = top_k_overlap(reranked_docs_a, reranked_docs_b, 5) if total_docs >= 5 else 0
    
    # Interpret correlation strength
    def interpret_correlation(corr):
        abs_corr = abs(corr)
        if abs_corr >= 0.9:
            return "Very Strong"
        elif abs_corr >= 0.7:
            return "Strong"
        elif abs_corr >= 0.5:
            return "Moderate"
        elif abs_corr >= 0.3:
            return "Weak"
        else:
            return "Very Weak"
    
    spearman_strength = interpret_correlation(spearman_corr)
    
    # Build comparison visualization
    comparison_html = f"""
    <div style="font-family: Arial, sans-serif; border: 1px solid #ddd; padding: 20px; border-radius: 12px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);">
        <h3 style="color: #1f2937; margin-bottom: 20px; border-bottom: 2px solid #3b82f6; padding-bottom: 10px;">
            📊 RAG Reranking Comparison Analysis
        </h3>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 25px;">
            <!-- Correlation Analysis -->
            <div style="background: white; padding: 18px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h4 style="color: #059669; margin: 0 0 15px 0; font-size: 16px;">🔗 Correlation Analysis</h4>
                <div style="space-y: 8px;">
                    <div style="margin-bottom: 8px;">
                        <strong>Spearman Correlation:</strong> 
                        <span style="color: {'#059669' if spearman_corr > 0.5 else '#dc2626' if spearman_corr < 0 else '#f59e0b'}; font-weight: bold;">
                            {spearman_corr:.3f}
                        </span>
                        <small style="color: #6b7280;">({spearman_strength})</small>
                    </div>
                    <div style="margin-bottom: 8px;">
                        <strong>Kendall's Tau:</strong> 
                        <span style="color: {'#059669' if kendall_corr > 0.5 else '#dc2626' if kendall_corr < 0 else '#f59e0b'}; font-weight: bold;">
                            {kendall_corr:.3f}
                        </span>
                    </div>
                    <div style="margin-bottom: 8px;">
                        <strong>Rank-Biased Overlap:</strong> 
                        <span style="color: {'#059669' if rbo_score > 0.7 else '#f59e0b' if rbo_score > 0.3 else '#dc2626'}; font-weight: bold;">
                            {rbo_score:.3f}
                        </span>
                    </div>
                </div>
            </div>
            
            <!-- Agreement Metrics -->
            <div style="background: white; padding: 18px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h4 style="color: #dc2626; margin: 0 0 15px 0; font-size: 16px;">🎯 Agreement Metrics</h4>
                <div style="space-y: 8px;">
                    <div style="margin-bottom: 8px;">
                        <strong>Exact Position Agreement:</strong> 
                        <span style="font-weight: bold; color: #1f2937;">{agreement_rate:.1f}%</span>
                        <small style="color: #6b7280;">({agreements}/{total_docs})</small>
                    </div>
                    <div style="margin-bottom: 8px;">
                        <strong>Ranking Similarity:</strong> 
                        <span style="font-weight: bold; color: #1f2937;">{normalized_rank_diff:.1f}%</span>
                        <small style="color: #6b7280;">(avg diff: {avg_rank_diff:.1f})</small>
                    </div>
                </div>
            </div>
            
            <!-- Top-K Overlaps -->
            <div style="background: white; padding: 18px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h4 style="color: #7c3aed; margin: 0 0 15px 0; font-size: 16px;">🏆 Top-K Overlaps</h4>
                <div style="space-y: 8px;">
                    <div style="margin-bottom: 8px;">
                        <strong>Top-1 Overlap:</strong> 
                        <span style="font-weight: bold; color: #1f2937;">{top1_overlap:.0f}%</span>
                    </div>"""
    
    if total_docs >= 3:
        comparison_html += f"""
                    <div style="margin-bottom: 8px;">
                        <strong>Top-3 Overlap:</strong> 
                        <span style="font-weight: bold; color: #1f2937;">{top3_overlap:.1f}%</span>
                    </div>"""
    
    if total_docs >= 5:
        comparison_html += f"""
                    <div style="margin-bottom: 8px;">
                        <strong>Top-5 Overlap:</strong> 
                        <span style="font-weight: bold; color: #1f2937;">{top5_overlap:.1f}%</span>
                    </div>"""
    
    comparison_html += f"""
                </div>
            </div>
            
            <!-- Summary & Interpretation -->
            <div style="background: white; padding: 18px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h4 style="color: #1f5582; margin: 0 0 15px 0; font-size: 16px;">💡 Interpretation</h4>
                <div style="font-size: 14px; line-height: 1.5;">"""
    
    # Add interpretation based on the metrics
    if spearman_corr > 0.7:
        interpretation = "🟢 <strong>High Similarity:</strong> Both rerankers show very similar document prioritization strategies for RAG."
    elif spearman_corr > 0.3:
        interpretation = "🟡 <strong>Moderate Similarity:</strong> Rerankers agree on some documents but may produce different RAG answers."
    elif spearman_corr > 0:
        interpretation = "🟠 <strong>Low Similarity:</strong> Rerankers have different strategies, likely producing varied RAG answers."
    else:
        interpretation = "🔴 <strong>Opposite Preferences:</strong> Rerankers show inverse patterns, expect very different RAG answers."
    
    comparison_html += f"""
                    <p style="margin: 0;">{interpretation}</p>
                    <p style="margin: 8px 0 0 0; color: #6b7280; font-size: 13px;">
                        💡 <strong>Tip:</strong> Document ranking similarity affects RAG answer quality. 
                        Different rankings may lead to different answers from the LLM.
                    </p>
                </div>
            </div>
        </div>
        
        <div style="margin-top: 20px; padding: 15px; background-color: #f1f5f9; border-radius: 8px; border-left: 4px solid #3b82f6;">
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; font-size: 14px;">
                <div><strong>Retrieved Documents:</strong> {total_docs}</div>
                <div><strong>Statistical Method:</strong> {'Spearman + Kendall' if scipy_available else 'Pearson (fallback)'}</div>
                <div><strong>Context:</strong> RAG Pipeline Analysis</div>
            </div>
        </div>
    </div>
    """
    
    return comparison_html

def generate_llm_answer(query, reranked_contexts, together_api_key, model_name=""):
    """Generate answer using Together AI Llama 70B based on reranked documents"""
    
    if not TOGETHER_AVAILABLE:
        return {
            "error": "Together AI library not installed. Please install: pip install together",
            "answer": "",
            "id": "",
            "document": ""
        }
    
    if not together_api_key or not together_api_key.strip():
        return {
            "error": "Please provide a valid Together AI API key",
            "answer": "",
            "id": "",
            "document": ""
        }
    
    try:
        # Initialize Together client
        client = Together(api_key=together_api_key.strip())
        
        # Format documents for the prompt
        documents_text = ""
        for i, ctx in enumerate(reranked_contexts, 1):
            documents_text += f"Document {i} (ID: {ctx.id}):\n{ctx.text}\n\n"
        
        # Create the prompt
        system_prompt = f"""You are a helpful AI assistant using reranked documents from {model_name}. Answer the user's question based ONLY on the provided documents. 

IMPORTANT INSTRUCTIONS:
1. Use ONLY the information from the provided documents to answer
2. If the answer cannot be found in the documents, say "I cannot answer this question based on the provided documents"
3. Always respond in the following JSON format:
{{
    "answer": "Your detailed answer here",
    "id": "ID of the primary document used",
    "document": "Text of the primary document used"
}}

4. Choose the most relevant document that contains the key information for your answer
5. If multiple documents are used, pick the PRIMARY/MOST IMPORTANT one for the "id" and "document" fields
6. Make sure your JSON is properly formatted and valid"""

        user_prompt = f"""Question: {query}

Available Documents (reranked by {model_name}):
{documents_text}

Please answer the question using the format specified in the instructions."""

        # Make API call to Together AI
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Low temperature for more deterministic responses
            max_tokens=1000
        )
        
        # Extract response content
        response_content = response.choices[0].message.content.strip()
        print(f"Raw LLM Response for {model_name}: {response_content}")  # Debug
        
        # Try to parse JSON response
        try:
            # Look for JSON in the response (handle cases where LLM adds extra text)
            import re
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed_response = json.loads(json_str)
                
                # Validate required fields
                if all(key in parsed_response for key in ["answer", "id", "document"]):
                    return parsed_response
                else:
                    # Missing fields, create a fallback response
                    return {
                        "answer": parsed_response.get("answer", response_content),
                        "id": parsed_response.get("id", reranked_contexts[0].id if reranked_contexts else "unknown"),
                        "document": parsed_response.get("document", reranked_contexts[0].text if reranked_contexts else "unknown")
                    }
            else:
                # No JSON found, treat entire response as answer
                return {
                    "answer": response_content,
                    "id": reranked_contexts[0].id if reranked_contexts else "unknown",
                    "document": reranked_contexts[0].text if reranked_contexts else "unknown"
                }
                
        except json.JSONDecodeError as e:
            print(f"JSON parsing error for {model_name}: {e}")
            # Fallback: treat the entire response as the answer
            return {
                "answer": response_content,
                "id": reranked_contexts[0].id if reranked_contexts else "unknown",
                "document": reranked_contexts[0].text if reranked_contexts else "unknown"
            }
        
    except Exception as e:
        return {
            "error": f"Together AI API error: {str(e)}",
            "answer": "",
            "id": "",
            "document": ""
        }

def retrieve_documents_rag(query, retriever_type, offline_method, corpus, n_docs):
    """Retrieve documents for RAG pipeline"""
    from rankify.retrievers.retriever import Retriever
    
    # Get user session
    user_id = get_or_create_user_session()
    
    # Show loading state immediately
    loading_outputs = (
        [(None, "🔄 Retrieving documents...")],  # retrieved_chat
        None,  # docs_state
        None,  # orig_contexts_state
        user_id,  # user_id_state
        "🔄 Retrieving documents..."  # status
    )
    yield loading_outputs
    
    if retriever_type == "Online Retriever":
        try:
            progress_outputs = (
                [(None, "🔄 Searching online documents...")],
                None, None, user_id,
                "🔄 Searching online documents..."
            )
            yield progress_outputs
            
            from rankify.retrievers.retriever import Retriever
            retriever = Retriever(method="online", n_docs=n_docs, api_key=SERPER_API_KEY)
            documents = [Document(question=Question(query), answers=Answer([]), contexts=[])]
            retrieved_documents = retriever.retrieve(documents)
            
            if not retrieved_documents:
                error_outputs = (
                    [(None, "❌ No online documents retrieved.")],
                    None, None, user_id,
                    "❌ No online documents retrieved."
                )
                yield error_outputs
                return
                
            # Extract contexts for saving
            original_contexts = retrieved_documents[0].contexts if retrieved_documents else []
            
            # Save retrieval interaction
            retriever_config = {
                "type": retriever_type,
                "method": "online_retriever",
                "corpus": "web",
                "n_docs": n_docs
            }
            
            interaction_id = save_retrieval_interaction(
                user_id, 
                query, 
                retriever_config, 
                original_contexts
            )
            
            # Format retrieved documents for display
            chat_display = format_documents_for_display(retrieved_documents, query)
            success_message = f"✅ Retrieved {len(original_contexts)} online documents (ID: {interaction_id[:8]})"
            
            success_outputs = (
                chat_display,
                retrieved_documents,  
                original_contexts,
                user_id,
                success_message
            )
            yield success_outputs
            return
            
        except Exception as e:
            error_outputs = (
                [(None, f"❌ Error during online retrieval: {str(e)}")],
                None, None, user_id,
                f"❌ Error during online retrieval: {str(e)}"
            )
            yield error_outputs
            return
        
    if not query.strip():
        error_outputs = (
            [(None, "❌ Please enter a query.")],
            None, None, user_id,
            "❌ Please enter a query."
        )
        yield error_outputs
        return
        
    if offline_method is None or corpus is None:
        error_outputs = (
            [(None, "❌ Please select both method and corpus.")],
            None, None, user_id,
            "❌ Please select both method and corpus."
        )
        yield error_outputs
        return
    
    try:
        # Update loading message
        progress_outputs = (
            [(None, "🔄 Initializing retriever...")],
            None, None, user_id,
            "🔄 Initializing retriever..."
        )
        yield progress_outputs
        
        retriever = Retriever(method=offline_method, n_docs=n_docs, index_type=corpus)
        documents = [Document(question=Question(query), answers=Answer([]), contexts=[])]
        
        # Update loading message
        progress_outputs = (
            [(None, "🔄 Searching documents...")],
            None, None, user_id,
            "🔄 Searching documents..."
        )
        yield progress_outputs
        
        retrieved_documents = retriever.retrieve(documents)
        
        if not retrieved_documents:
            error_outputs = (
                [(None, "❌ No documents retrieved.")],
                None, None, user_id,
                "❌ No documents retrieved."
            )
            yield error_outputs
            return
        
        # Extract contexts for saving
        original_contexts = retrieved_documents[0].contexts if retrieved_documents else []
        
        # Save retrieval interaction
        retriever_config = {
            "type": retriever_type,
            "method": offline_method,
            "corpus": corpus,
            "n_docs": n_docs
        }
        
        interaction_id = save_retrieval_interaction(
            user_id, 
            query, 
            retriever_config, 
            original_contexts
        )
        
        # Format retrieved documents for display
        chat_display = format_documents_for_display(retrieved_documents, query)
        success_message = f"✅ Retrieved {len(original_contexts)} documents (ID: {interaction_id[:8]})"
        
        success_outputs = (
            chat_display,
            retrieved_documents,
            original_contexts,
            user_id,
            success_message
        )
        yield success_outputs
        
    except Exception as e:
        error_outputs = (
            [(None, f"❌ Error during retrieval: {str(e)}")],
            None, None, user_id,
            f"❌ Error during retrieval: {str(e)}"
        )
        yield error_outputs

def run_rag_pipeline(query, retrieved_docs, original_contexts, method_a, model_a, method_b, model_b, user_id):
    """Run the complete RAG pipeline: retrieve -> rerank -> generate answers with both models"""
    global current_rag_session, pending_llm_evaluations
    
    if not retrieved_docs or not original_contexts or not user_id:
        error_outputs = ( 
            [(None, "❌ No documents to rerank. Please retrieve documents first.")],  # rerank_chat_a
            [(None, "❌ No documents to rerank.")],  # rerank_chat_b
            [(None, "❌ No documents available for answering.")],  # answer_chat_a
            [(None, "❌ No documents available for answering.")],  # answer_chat_b
            "",  # comparison_display
            "",  # answer_json_a
            "",  # answer_json_b
            "",  # reranking_id_state
            True  # vote_enabled_state
        )
        return error_outputs
    
    # Show loading state
    loading_outputs = (
        [(None, "🔄 Reranking with Model A...")],
        [(None, "🔄 Waiting for Model A...")],
        [(None, "🔄 Waiting for reranking...")],
        [(None, "🔄 Waiting for reranking...")],
        "",
        "", "",
        "",
        True
    )
    yield loading_outputs
    
    try:
        doc = retrieved_docs[0]
        
        # Step 1: Rerank with Model A
        progress_outputs = (
            [(None, f"🔄 Running {method_a} with {model_a}...")],
            [(None, "🔄 Waiting for Model A...")],
            [(None, "🔄 Waiting for reranking...")],
            [(None, "🔄 Waiting for reranking...")],
            "",
            "", "",
            "",
            True
        )
        yield progress_outputs
        
        reranked_contexts_a = run_rankify_reranker_from_retriever(doc, method_a, model_a)
        
        # Step 2: Rerank with Model B
        progress_outputs = (
            [(None, "✅ Model A Reranking Complete")],
            [(None, f"🔄 Running {method_b} with {model_b}...")],
            [(None, "🔄 Waiting for reranking...")],
            [(None, "🔄 Waiting for reranking...")],
            "",
            "", "",
            "",
            True
        )
        yield progress_outputs
        
        reranked_contexts_b = run_rankify_reranker_from_retriever(doc, method_b, model_b)
        
        # Generate ranking comparison
        comparison_result = compare_rankings_from_contexts(original_contexts, reranked_contexts_a, reranked_contexts_b)
        
        # Save reranking interaction
        reranking_id = save_reranking_interaction(
            user_id,
            query,
            method_a, model_a,
            method_b, model_b,
            reranked_contexts_a,
            reranked_contexts_b,
            original_contexts
        )
        
        # Format reranked documents for display
        rerank_display_a = format_reranked_documents_for_display(original_contexts, reranked_contexts_a, query)
        rerank_display_b = format_reranked_documents_for_display(original_contexts, reranked_contexts_b, query)
        
        # Step 3: Generate answer with Model A
        progress_outputs = (
            rerank_display_a,
            rerank_display_b,
            [(None, f"🔄 Generating answer with {method_a}::{model_a} using Llama 70B...")],
            [(None, "🔄 Waiting for Model A answer...")],
            comparison_result,
            "", "",
            reranking_id,
            True
        )
        yield progress_outputs
        together_api_key = TOGETHER_API_KEY
        llm_result_a = generate_llm_answer(query, reranked_contexts_a, together_api_key, f"{method_a}::{model_a}")
        
        # Step 4: Generate answer with Model B
        progress_outputs = (
            rerank_display_a,
            rerank_display_b,
            [(None, f"✅ Answer A Complete")],
            [(None, f"🔄 Generating answer with {method_b}::{model_b} using Llama 70B...")],
            comparison_result,
            json.dumps(llm_result_a, indent=2) if not llm_result_a.get("error") else f"Error: {llm_result_a['error']}",
            "",
            reranking_id,
            True
        )
        yield progress_outputs
        
        llm_result_b = generate_llm_answer(query, reranked_contexts_b, together_api_key, f"{method_b}::{model_b}")
        
        # Store current session data for voting
        current_rag_session[user_id] = {
            "query": query,
            "method_a": method_a,
            "model_a": model_a,
            "method_b": method_b,
            "model_b": model_b,
            "reranked_contexts_a": reranked_contexts_a,
            "reranked_contexts_b": reranked_contexts_b,
            "llm_result_a": llm_result_a,
            "llm_result_b": llm_result_b,
            "reranking_id": reranking_id
        }
        
        # 🆕 SAVE RAG INTERACTION WITH LLM ANSWERS
        try:
            rag_interaction_id = save_rag_interaction(
                user_id=user_id,
                query=query,
                reranking_id=reranking_id,
                method_a=method_a,
                model_a=model_a,
                method_b=method_b,
                model_b=model_b,
                llm_result_a=llm_result_a,
                llm_result_b=llm_result_b,
                reranked_contexts_a=reranked_contexts_a,
                reranked_contexts_b=reranked_contexts_b
            )
            print(f"✅ RAG interaction saved with ID: {rag_interaction_id}")
            
            # Update session with RAG interaction ID
            current_rag_session[user_id]["rag_interaction_id"] = rag_interaction_id
            
        except Exception as e:
            print(f"⚠️ Failed to save RAG interaction: {str(e)}")
            # Continue anyway - don't break the user experience
        
        # Format answers for display
        if "error" in llm_result_a:
            answer_display_a = [(None, f"❌ {llm_result_a['error']}")]
        else:
            answer_text_a = f"""**Generated Answer (Model A):**
{llm_result_a['answer']}

**Primary Source Document:**
- **ID:** {llm_result_a['id']}
- **Text:** {llm_result_a['document'][:200]}{'...' if len(llm_result_a['document']) > 200 else ''}"""
            answer_display_a = [(None, answer_text_a)]
        
        if "error" in llm_result_b:
            answer_display_b = [(None, f"❌ {llm_result_b['error']}")]
        else:
            answer_text_b = f"""**Generated Answer (Model B):**
{llm_result_b['answer']}

**Primary Source Document:**
- **ID:** {llm_result_b['id']}
- **Text:** {llm_result_b['document'][:200]}{'...' if len(llm_result_b['document']) > 200 else ''}"""
            answer_display_b = [(None, answer_text_b)]
        
        # Add success messages
        success_message_a = f"**{method_a}::{model_a}** - ✅ RAG complete (ID: {reranking_id[:8]})"
        success_message_b = f"**{method_b}::{model_b}** - ✅ RAG complete (ID: {reranking_id[:8]})"
        rerank_display_a.append((None, success_message_a))
        rerank_display_b.append((None, success_message_b))
        
        # Format JSON outputs
        answer_json_a = json.dumps(llm_result_a, indent=2) if not llm_result_a.get("error") else f"Error: {llm_result_a['error']}"
        answer_json_b = json.dumps(llm_result_b, indent=2) if not llm_result_b.get("error") else f"Error: {llm_result_b['error']}"
        
        # Trigger LLM evaluation in background (for reranking comparison)
        try:
            model_a_info = {"method": method_a, "model": model_a}
            model_b_info = {"method": method_b, "model": model_b}
            llm_evaluation = llm_judge.evaluate_reranking(
                query, model_a_info, model_b_info, 
                reranked_contexts_a, reranked_contexts_b
            )
            pending_llm_evaluations[user_id] = {
                'evaluation': llm_evaluation,
                'reranking_id': reranking_id,
                'query': query,
                'models': {'model_a': model_a_info, 'model_b': model_b_info}
            }
            print(f"🤖 LLM Judge evaluation completed for user {user_id}: {llm_evaluation['winner']}")
        except Exception as e:
            print(f"⚠️ LLM evaluation failed (will continue without it): {str(e)}")
            pending_llm_evaluations[user_id] = {
                'evaluation': {
                    "winner": "Error",
                    "reasoning": f"LLM evaluation failed: {str(e)}",
                    "full_response": f"Error: {str(e)}"
                },
                'reranking_id': reranking_id,
                'query': query,
                'models': {'model_a': model_a_info, 'model_b': model_b_info}
            }
        
        final_outputs = (
            rerank_display_a,
            rerank_display_b,
            answer_display_a,
            answer_display_b,
            comparison_result,
            answer_json_a,
            answer_json_b,
            reranking_id,
            True  # Enable voting
        )
        yield final_outputs
        
    except Exception as e:
        error_outputs = (
            [(None, f"❌ Error in RAG pipeline: {str(e)}")],
            [(None, f"❌ Error in RAG pipeline: {str(e)}")],
            [(None, f"❌ Error during answer generation: {str(e)}")],
            [(None, f"❌ Error during answer generation: {str(e)}")],
            "",
            json.dumps({"error": str(e)}, indent=2),
            json.dumps({"error": str(e)}, indent=2),
            "",
            True
        )
        yield error_outputs

def vote_winner_rag(model_a, model_b, winner, user_id, reranking_id):
    """Handle voting with proper user session management and LLM comparison for RAG results"""
    global current_rag_session, pending_llm_evaluations, voted_queries
    
    if not user_id or not winner:
        return "❌ Please select a winner to vote.", True  # Button remains enabled
    
    # Check if the user has already voted for this query
    user_query_key = f"{user_id}_{reranking_id}"
    
    try:
        # Save user vote
        vote_id, llm_result = save_vote_interaction(user_id, winner)
        
        # Mark this query as voted
        voted_queries[user_query_key] = True
        
        # Check if we have a pending LLM evaluation
        llm_comparison = ""
        if user_id in pending_llm_evaluations:
            pending_eval = pending_llm_evaluations[user_id]
            llm_evaluation = pending_eval['evaluation']
            
            # Save LLM evaluation to file now that user has voted
            llm_vote_id = save_llm_vote_interaction(user_id, llm_evaluation, winner)
            
            # Prepare comparison message
            llm_winner = llm_evaluation.get('winner', 'Unknown')
            llm_reasoning = llm_evaluation.get('reasoning', 'No reasoning provided')
            
            # Check agreement
            user_choice = winner.lower().replace('model ', '')  # "Model A" -> "a"
            llm_choice = llm_winner.lower().replace('model ', '') if 'model' in llm_winner.lower() else llm_winner.lower()
            
            agreement_status = "✅ Agree" if user_choice == llm_choice else "❌ Disagree"
            if winner.lower() == "tie":
                agreement_status = "🤝 You called it a tie"
            elif llm_winner.lower() == "tie":
                agreement_status = "🤝 LLM called it a tie"
            elif llm_winner.lower() == "error":
                agreement_status = "⚠️ LLM evaluation failed"
            
            llm_comparison = f"\n\n🤖 LLM Judge (Reranking): {llm_winner} | {agreement_status}\n📝 Reasoning: {llm_reasoning}"
            
            # Clean up pending evaluation
            del pending_llm_evaluations[user_id]
        
        # Add RAG-specific information if available
        rag_info = ""
        if user_id in current_rag_session:
            session_data = current_rag_session[user_id]
            llm_result_a = session_data.get('llm_result_a', {})
            llm_result_b = session_data.get('llm_result_b', {})
            rag_interaction_id = session_data.get('rag_interaction_id', 'N/A')
            
            # Compare answer lengths and other metrics
            if not llm_result_a.get("error") and not llm_result_b.get("error"):
                len_a = len(llm_result_a.get('answer', ''))
                len_b = len(llm_result_b.get('answer', ''))
                words_a = len(llm_result_a.get('answer', '').split())
                words_b = len(llm_result_b.get('answer', '').split())
                same_source = llm_result_a.get('id') == llm_result_b.get('id')
                
                rag_info = f"""
📝 RAG Answer Comparison:
• Model A: {len_a} chars, {words_a} words
• Model B: {len_b} chars, {words_b} words  
• Same source doc: {'Yes' if same_source else 'No'}
🆔 RAG Interaction ID: {rag_interaction_id[:8]}"""
            else:
                error_a = "Error" if llm_result_a.get("error") else "Success"
                error_b = "Error" if llm_result_b.get("error") else "Success"
                rag_info = f"""
📝 RAG Results: Model A: {error_a}, Model B: {error_b}
🆔 RAG Interaction ID: {rag_interaction_id[:8]}"""
        
        # Get updated stats including RAG stats
        stats = get_user_stats(user_id)
        rag_stats = get_rag_stats(user_id)
        
        base_message = f"✅ RAG Vote recorded! (ID: {vote_id[:8]})"
        stats_message = f"""
📊 Your Stats: 
• Votes: {stats['vote_count']} | Reranks: {stats['reranking_count']} | LLM Evals: {stats['llm_vote_count']}
• RAG Sessions: {rag_stats['total_rag_interactions']} | Success Rate: {100-rag_stats['error_rate']:.1f}%"""
        
        return base_message + llm_comparison + rag_info + stats_message, False  # Disable the button after voting
        
    except Exception as e:
        return f"❌ Error saving vote: {str(e)}", True  # Keep button enabled if there's an error

def format_documents_for_display(documents, query):
    """Format retrieved documents for chatbot display"""
    combined_info = []
    doc_counter = 1
    query_normalized = query.lower().strip()
    
    for doc in documents:
        for ctx in doc.contexts:
            text = ctx.text.strip()
            
            # Clean text by removing query
            import re
            pattern = re.escape(query_normalized)
            cleaned_text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
            
            info = (
                f"**Document {doc_counter}**\n"
                f"**ID:** {ctx.id}\n"
                f"**Title:** {ctx.title}\n"
                f"**Text:** {cleaned_text}\n"
                f"**Score:** {ctx.score:.4f}"
            )
            combined_info.append(info)
            doc_counter += 1
    
    if combined_info:
        combined_response = f"**Query:** {query}\n\n" + "\n\n".join(combined_info)
    else:
        combined_response = f"**Query:** {query}\n\nNo documents found."
    
    return [(None, combined_response)]

def format_reranked_documents_for_display(original_contexts, reranked_contexts, query):
    """Format reranked documents for chatbot display with movement indicators"""
    combined_info = []
    doc_counter = 1
    query_normalized = query.lower().strip()
    
    # Map original positions for movement arrows
    orig_pos = {ctx.id: i for i, ctx in enumerate(original_contexts)}
    
    for new_idx, ctx in enumerate(reranked_contexts):
        old_idx = orig_pos.get(ctx.id, None)
        if old_idx is None:
            arrow = ""
        elif new_idx < old_idx:
            arrow = "🟢⬆️"
        elif new_idx > old_idx:
            arrow = "🔴⬇️"
        else:
            arrow = "–"
        
        # Clean the text by removing the query
        text = ctx.text.strip()
        if query_normalized in text.lower():
            import re
            pattern = re.escape(query_normalized)
            text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
        
        info = (
            f"**Document {doc_counter}**\n"
            f"{arrow} **Rank {new_idx+1}** | **ID:** {ctx.id} | **Title:** {ctx.title}\n"
            f"**Score:** {ctx.score:.4f}\n"
            f"**Text:** {text}"
        )
        combined_info.append(info)
        doc_counter += 1
    
    if combined_info:
        combined_response = f"**Query:** {query}\n\n" + "\n\n".join(combined_info)
    else:
        combined_response = f"**Query:** {query}\n\nNo documents reranked."
    
    return [(None, combined_response)]

def build_rag_tab():
    with gr.Tab("💬 RAG Arena"):
        gr.Markdown("# 🤖 RAG Arena: Retrieve → Rerank → Generate & Compare")
        gr.Markdown("Complete RAG pipeline: retrieve documents, rerank with two different methods, and generate answers using Llama 70B for comparison")
        
        # User info display
        with gr.Row():
            user_info = gr.Markdown("👤 **Your Session ID will be generated when you start**")
        
        # Step 1: Document Retrieval
        gr.Markdown("## 📚 Step 1: Document Retrieval")
        with gr.Row():
            with gr.Column():
                query_box = gr.Textbox(label="Enter your question", placeholder="What is machine learning?")
                retriever_type = gr.Radio(
                    ["Online Retriever", "Offline Retriever"],
                    value="Offline Retriever",
                    label="Retriever Type"
                )
            with gr.Column():
                offline_method = gr.Dropdown(
                    ["bm25", "dpr-multi", "bge", "colbert", "contriever"],
                    label="Offline Retriever Method",
                    visible=True
                )
                corpus = gr.Dropdown(
                    ["wiki", "msmarco"],
                    label="Corpus",
                    visible=True
                )
                n_docs = gr.Slider(1, 10, value=5, step=1, label="Number of Documents")
        
        retrieve_btn = gr.Button("🔍 Retrieve Documents", variant="primary")
        retrieve_status = gr.Textbox(label="Retrieval Status", visible=True, interactive=False)
        
        # Retrieved documents display
        gr.Markdown("### Retrieved Documents")
        retrieved_chat = gr.Chatbot(height=300, label="Retrieved Documents")
        
        # Hidden state variables
        docs_state = gr.State(None)
        orig_contexts_state = gr.State(None)
        user_id_state = gr.State(None)
        reranking_id_state = gr.State("")
        vote_enabled_state = gr.State(True)
        
        # Step 2: Dual Reranking + Answer Generation
        gr.Markdown("## 🔄 Step 2: Dual Reranking & Answer Generation")
        
        with gr.Row():
            with gr.Column():
                method_names = get_rankify_methods()
                method_a = gr.Dropdown(
                    choices=["Select Reranker"] + method_names, 
                    label="Method for Model A"
                )
                model_a = gr.Dropdown(choices=[], label="Model A")
            with gr.Column():
                method_b = gr.Dropdown(
                    choices=["Select Reranker"] + method_names, 
                    label="Method for Model B"
                )
                model_b = gr.Dropdown(choices=[], label="Model B")
        
        # with gr.Row():
        #     together_api_key = gr.Textbox(
        #         label="Together AI API Key", 
        #         type="password",
        #         placeholder="Enter your Together AI API key"
        #     )
        #     gr.Markdown("💡 Get your free API key at [together.ai](https://together.ai)")
        
        rag_btn = gr.Button("🚀 Run Dual RAG Pipeline", variant="primary")
        
        # Reranked documents display
        gr.Markdown("### 📋 Reranked Documents Comparison")
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Model A Reranked")
                rerank_chat_a = gr.Chatbot(height=400, label="Model A Reranked")
            with gr.Column():
                gr.Markdown("#### Model B Reranked")
                rerank_chat_b = gr.Chatbot(height=400, label="Model B Reranked")
        
        # Statistical comparison section
        gr.Markdown("### 📊 Reranking Comparison Analysis")
        comparison_display = gr.HTML(value="<p style='color: #666; text-align: center; padding: 20px;'>🔄 Run RAG pipeline to see statistical comparison between rerankers</p>")
        
        # Generated answers display
        gr.Markdown("### 🎯 Generated Answers Comparison")
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Answer from Model A")
                answer_chat_a = gr.Chatbot(height=400, label="Model A Answer")
            with gr.Column():
                gr.Markdown("#### Answer from Model B")
                answer_chat_b = gr.Chatbot(height=400, label="Model B Answer")
        
        # JSON outputs
        gr.Markdown("### 📄 Structured JSON Outputs")
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Model A JSON Response")
                answer_json_a = gr.Code(
                    label="Model A JSON",
                    language="json",
                    value='{\n  "answer": "",\n  "id": "",\n  "document": ""\n}',
                    lines=8
                )
            with gr.Column():
                gr.Markdown("#### Model B JSON Response")
                answer_json_b = gr.Code(
                    label="Model B JSON",
                    language="json",
                    value='{\n  "answer": "",\n  "id": "",\n  "document": ""\n}',
                    lines=8
                )
        
        # Voting section
        gr.Markdown("### 🗳️ Vote for Better RAG Performance")
        vote_btn = gr.Radio(
            choices=["Model A", "Model B", "Tie"], 
            label="Which model provided better RAG results? (Consider both reranking and answer quality)"
        )
        vote_submit = gr.Button("Vote", variant="secondary", interactive=True)
        vote_output = gr.Textbox(label="Vote Result", lines=8)
        
        # Info section
        with gr.Row():
            gr.HTML("""
            <div style="background-color: #e8f4fd; border: 1px solid #b3d9ff; border-radius: 8px; padding: 15px; margin: 10px 0;">
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <span style="font-size: 20px; margin-right: 8px;">ℹ️</span>
                    <strong style="color: #1f5582;">RAG Arena Information</strong>
                </div>
                <ul style="margin: 0; padding-left: 20px; color: #374151;">
                    <li><strong>Retrieval:</strong> Online web search OR BM25, DPR, BGE, ColBERT, Contriever from Wiki/MSMARCO</li>
                    <li><strong>Reranking:</strong> Compare two different Rankify rerankers side-by-side</li>
                    <li><strong>Generation:</strong> Llama 3.3 70B Instruct via Together AI for both models</li>
                    <li><strong>Comparison:</strong> Statistical reranking analysis + answer quality evaluation</li>
                    <li><strong>Output:</strong> Structured JSON with answer, source document ID, and text</li>
                    <li><strong>Voting:</strong> Rate which reranker produces better RAG results</li>
                </ul>
            </div>
            """)
        
        # Event handlers
        def toggle_offline_fields(retriever_type):
            if retriever_type == "Offline Retriever":
                return gr.update(visible=True), gr.update(visible=True)
            else:  # Online Retriever
                return gr.update(visible=False), gr.update(visible=False)
        
        def update_user_info(user_id):
            if user_id:
                stats = get_user_stats(user_id)
                return gr.update(value=f"👤 **Session: {user_id}** ")
            return gr.update(value="👤 **Your Session ID will be generated when you start**")
        
        retriever_type.change(
            toggle_offline_fields,
            [retriever_type],
            [offline_method, corpus]
        )
        
        method_a.change(
            lambda m: gr.update(choices=get_models_for_method(m)),
            method_a,
            model_a
        )
        
        method_b.change(
            lambda m: gr.update(choices=get_models_for_method(m)),
            method_b,
            model_b
        )
        
        # Retrieve documents
        retrieve_btn.click(
            fn=retrieve_documents_rag,
            inputs=[query_box, retriever_type, offline_method, corpus, n_docs],
            outputs=[retrieved_chat, docs_state, orig_contexts_state, user_id_state, retrieve_status]
        )
        
        # Update user info when user_id changes
        user_id_state.change(
            update_user_info,
            [user_id_state],
            [user_info]
        )
        
        # Run dual RAG pipeline
        rag_btn.click(
            fn=run_rag_pipeline,
            inputs=[query_box, docs_state, orig_contexts_state, method_a, model_a, method_b, model_b, user_id_state],
            outputs=[rerank_chat_a, rerank_chat_b, answer_chat_a, answer_chat_b, comparison_display, answer_json_a, answer_json_b, reranking_id_state, vote_enabled_state]
        ).then(
            lambda vote_enabled: gr.update(interactive=vote_enabled),
            inputs=[vote_enabled_state],
            outputs=[vote_submit]
        )
        
        # Vote handling
        vote_submit.click(
            vote_winner_rag, 
            [model_a, model_b, vote_btn, user_id_state, reranking_id_state], 
            [vote_output, vote_enabled_state]
        ).then(
            lambda vote_enabled: gr.update(interactive=vote_enabled),
            [vote_enabled_state],
            [vote_submit]
        )