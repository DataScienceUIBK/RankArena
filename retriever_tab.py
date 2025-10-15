#Rtreiver tab
import gradio as gr
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from utils import get_rankify_methods, get_models_for_method, run_rankify_reranker_from_retriever
from logging_utils import (
    save_retrieval_interaction, 
    save_reranking_interaction, 
    save_vote_interaction,
    save_llm_vote_interaction,
    get_or_create_user_session,
    get_user_stats,
    llm_judge
)
from rankify.dataset.dataset import Document, Question, Answer, Context
import time
SERPER_API_KEY  = os.getenv("SERPER_API_KEY", "")


# Global storage for current reranking session data
current_reranking_session = {}

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
        # Simplified RBO calculation
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
            📊 Statistical Reranking Comparison Analysis
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
        interpretation = "🟢 <strong>High Similarity:</strong> Both rerankers show very similar preferences for retrieved documents."
    elif spearman_corr > 0.3:
        interpretation = "🟡 <strong>Moderate Similarity:</strong> Rerankers agree on some documents but prioritize others differently."
    elif spearman_corr > 0:
        interpretation = "🟠 <strong>Low Similarity:</strong> Rerankers have different strategies for ranking retrieved documents."
    else:
        interpretation = "🔴 <strong>Opposite Preferences:</strong> Rerankers show inverse ranking patterns for retrieved documents."
    
    comparison_html += f"""
                    <p style="margin: 0;">{interpretation}</p>
                    <p style="margin: 8px 0 0 0; color: #6b7280; font-size: 13px;">
                        💡 <strong>Tip:</strong> Values closer to 1.0 indicate more similar reranking behavior. 
                        This analysis shows how different rerankers prioritize retrieved documents.
                    </p>
                </div>
            </div>
        </div>
        
        <div style="margin-top: 20px; padding: 15px; background-color: #f1f5f9; border-radius: 8px; border-left: 4px solid #3b82f6;">
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; font-size: 14px;">
                <div><strong>Retrieved Documents:</strong> {total_docs}</div>
                <div><strong>Statistical Method:</strong> {'Spearman + Kendall' if scipy_available else 'Pearson (fallback)'}</div>
                <div><strong>RBO Parameter:</strong> p=0.9</div>
            </div>
        </div>
    </div>
    """
    
    return comparison_html

def extract_context_display_info(documents, query):
    """Format retrieved documents for chatbot display as a single message with query at the top."""
    combined_info = []
    doc_counter = 1  # Single counter for all documents
    query_normalized = query.lower().strip()
    
    for doc in documents:
        for ctx in doc.contexts:
            text = ctx.text.strip()
            print(f"Before cleaning - ID: {ctx.id}, Text: {text}")  # Debug
            
            # Remove all instances of the query from the text (case-insensitive)
            import re
            pattern = re.escape(query_normalized)
            cleaned_text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
            
            # If the text still ends with the query (unlikely after re.sub, but for safety)
            if cleaned_text.lower().endswith(query_normalized):
                cleaned_text = cleaned_text[:-len(query_normalized)].strip()
            
            print(f"After cleaning - ID: {ctx.id}, Text: {cleaned_text}")  # Debug
            
            info = (
                f"**Document {doc_counter}**\n"
                f"**ID:** {ctx.id}\n"
                f"**Title:** {ctx.title}\n"
                f"**Text:** {cleaned_text}\n"
                f"**Score:** {ctx.score}"
            )
            combined_info.append(info)
            doc_counter += 1

    # Combine all document info into a single response with the query at the top
    if combined_info:
        combined_response = f"**Query:** {query}\n\n" + "\n\n".join(combined_info)
    else:
        combined_response = f"**Query:** {query}\n\nNo documents retrieved."
    
    chat_messages = [(None, combined_response)]  # Use None as the user input to avoid query repetition
    print("Final chat messages:", chat_messages)  # Debug
    return chat_messages

def visualize_context_movement(original_contexts, reranked_contexts, query):
    """Format reranked documents for chatbot display as a single message with query at the top."""
    combined_info = []
    doc_counter = 1  # Single counter for all documents
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
        print(f"Before cleaning (rerank) - ID: {ctx.id}, Text: {text}")  # Debug
        if query_normalized in text.lower():
            import re
            pattern = re.escape(query_normalized)
            text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
        if text.lower().endswith(query_normalized):
            text = text[:-len(query_normalized)].strip()
        print(f"After cleaning (rerank) - ID: {ctx.id}, Text: {text}")  # Debug
        
        info = (
            f"**Document {doc_counter}**\n"
            f"{arrow} **Rank {new_idx+1}** | **ID:** {ctx.id} | **Title:** {ctx.title}\n"
            f"**Score:** {ctx.score}\n"
            f"**Text:** {text}"
        )
        combined_info.append(info)
        doc_counter += 1
    
    # Combine all document info into a single response with the query at the top
    if combined_info:
        combined_response = f"**Query:** {query}\n\n" + "\n\n".join(combined_info)
    else:
        combined_response = f"**Query:** {query}\n\nNo documents reranked."
    
    chat_messages = [(None, combined_response)]
    print("Rerank chat messages:", chat_messages)  # Debug
    return chat_messages

def retrieve_documents(query, retriever_type, offline_method, corpus, n_docs):
    from rankify.retrievers.retriever import Retriever
    
    # Get user session
    user_id = get_or_create_user_session()
    
    # Show loading state immediately
    loading_message = [(None, "🔄 Retrieving documents...")]
    initial_outputs = (
        [gr.update(value=loading_message, visible=True)]
        + [gr.update(value="", visible=False)]
        + [None, None, user_id]
    )
    yield initial_outputs
    
    if retriever_type == "Online Retriever":
        try:
            # Update loading message
            progress_outputs = (
                [gr.update(value=[(None, "🔄 Searching online documents...")], visible=True)]
                + [gr.update(value="", visible=False)]
                + [None, None, user_id]
            )
            yield progress_outputs
            
            retriever = Retriever(method="online", n_docs=n_docs, api_key=SERPER_API_KEY)
            documents = [Document(question=Question(query), answers=Answer([]), contexts=[])]
            
            # Update loading message
            progress_outputs = (
                [gr.update(value=[(None, "🔄 Processing online results...")], visible=True)]
                + [gr.update(value="", visible=False)]
                + [None, None, user_id]
            )
            yield progress_outputs
            
            retrieved_documents = retriever.retrieve(documents)
            print("Retrieved online documents:", retrieved_documents)  # Debug
            
            if not retrieved_documents:
                final_outputs = (
                    [gr.update(value=[(None, "❌ No online documents retrieved.")], visible=True)]
                    + [gr.update(value="❌ No online documents retrieved.", visible=True)]
                    + [None, None, user_id]
                )
                yield final_outputs
                return
            
            # Extract contexts for saving
            original_contexts = retrieved_documents[0].contexts if retrieved_documents else []
            print("Online contexts:", original_contexts)  # Debug
            
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
            
            # Format retrieved documents for chatbot
            chat_messages = extract_context_display_info(retrieved_documents, query)
            print("Online chat messages:", chat_messages)  # Debug
            success_message = f"✅ Retrieved {len(original_contexts)} online documents (ID: {interaction_id[:8]})"
            
            final_outputs = (
                [gr.update(value=chat_messages, visible=True)]
                + [gr.update(value=success_message, visible=True)]
                + [retrieved_documents, original_contexts, user_id]
            )
            yield final_outputs
            
        except Exception as e:
            error_outputs = (
                [gr.update(value=[(None, f"❌ Error during online retrieval: {str(e)}")], visible=True)]
                + [gr.update(value=f"❌ Error during online retrieval: {str(e)}", visible=True)]
                + [None, None, user_id]
            )
            yield error_outputs
            return
        
    if not query.strip():
        final_outputs = (
            [gr.update(value=[(None, "Please enter a query.")], visible=True)]
            + [gr.update(value="", visible=False)]
            + [None, None, user_id]
        )
        yield final_outputs
        return
        
    if offline_method is None or corpus is None:
        final_outputs = (
            [gr.update(value=[(None, "Please select both method and corpus.")], visible=True)]
            + [gr.update(value="", visible=False)]
            + [None, None, user_id]
        )
        yield final_outputs
        return
    
    try:
        # Update loading message
        progress_outputs = (
            [gr.update(value=[(None, "🔄 Initializing retriever...")], visible=True)]
            + [gr.update(value="", visible=False)]
            + [None, None, user_id]
        )
        yield progress_outputs
        
        retriever = Retriever(method=offline_method, n_docs=n_docs, index_type=corpus)
        documents = [Document(question=Question(query), answers=Answer([]), contexts=[])]
        
        # Update loading message
        progress_outputs = (
            [gr.update(value=[(None, "🔄 Searching documents...")], visible=True)]
            + [gr.update(value="", visible=False)]
            + [None, None, user_id]
        )
        yield progress_outputs
        
        retrieved_documents = retriever.retrieve(documents)
        print("Retrieved documents:", retrieved_documents)  # Debug: Check all documents
        
        if not retrieved_documents:
            final_outputs = (
                [gr.update(value=[(None, "No documents retrieved.")], visible=True)]
                + [gr.update(value="", visible=False)]
                + [None, None, user_id]
            )
            yield final_outputs
            return
        
        # Extract contexts for saving
        original_contexts = retrieved_documents[0].contexts if retrieved_documents else []
        print("Original contexts:", original_contexts)  # Debug: Check contexts
        
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
        
        # Format retrieved documents for chatbot
        chat_messages = extract_context_display_info(retrieved_documents, query)
        print("Chat messages:", chat_messages)  # Debug: Check chat output
        success_message = f"✅ Retrieved {len(original_contexts)} documents (ID: {interaction_id[:8]})"
        
        final_outputs = (
            [gr.update(value=chat_messages, visible=True)]
            + [gr.update(value=success_message, visible=True)]
            + [retrieved_documents, original_contexts, user_id]
        )
        yield final_outputs
        
    except Exception as e:
        error_outputs = (
            [gr.update(value=[(None, f"Error during retrieval: {str(e)}")], visible=True)]
            + [gr.update(value=f"Error during retrieval: {str(e)}", visible=True)]
            + [None, None, user_id]
        )
        yield error_outputs

def rerank_and_visualize(query, retrieved_docs, original_contexts, method_a, model_a, method_b, model_b, user_id):
    global current_reranking_session
    
    if not retrieved_docs or not original_contexts or not user_id:
        empty_outputs = (
            [gr.update(value=[(None, "❌ No documents to rerank.")], visible=True)]
            + [gr.update(value=[(None, "❌ No documents to rerank.")], visible=True)]
            + [""]  # Empty comparison
        )
        return empty_outputs
    
    # Show loading state for both columns
    loading_outputs_a = [(None, "🔄 Reranking with Model A...")]
    loading_outputs_b = [(None, "🔄 Reranking with Model B...")]
    yield [gr.update(value=loading_outputs_a, visible=True), gr.update(value=loading_outputs_b, visible=True), ""]
    
    try:
        doc = retrieved_docs[0]
        
        # Update loading message for Model A
        progress_outputs_a = [(None, f"🔄 Running {method_a} with {model_a}...")]
        progress_outputs_b = [(None, "🔄 Waiting for Model A...")]
        yield [gr.update(value=progress_outputs_a, visible=True), gr.update(value=progress_outputs_b, visible=True), ""]
        
        # Rerank with Model A
        reranked_contexts_a = run_rankify_reranker_from_retriever(doc, method_a, model_a)
        
        # Update loading message for Model B
        progress_outputs_a = [(None, "✅ Model A Complete")]
        progress_outputs_b = [(None, f"🔄 Running {method_b} with {model_b}...")]
        yield [gr.update(value=progress_outputs_a, visible=True), gr.update(value=progress_outputs_b, visible=True), ""]
        
        # Rerank with Model B
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
        
        # Store current session data for later LLM evaluation
        current_reranking_session[user_id] = {
            "query": query,
            "method_a": method_a,
            "model_a": model_a,
            "method_b": method_b,
            "model_b": model_b,
            "reranked_contexts_a": reranked_contexts_a,
            "reranked_contexts_b": reranked_contexts_b,
            "reranking_id": reranking_id
        }
        
        # Generate chat messages for visualizations
        chat_messages_a = visualize_context_movement(original_contexts, reranked_contexts_a, query)
        chat_messages_b = visualize_context_movement(original_contexts, reranked_contexts_b, query)
        
        # Append success message as a separate entry
        success_message_a = f"**{method_a}::{model_a}** - ✅ Reranking complete (ID: {reranking_id[:8]})"
        success_message_b = f"**{method_b}::{model_b}** - ✅ Reranking complete (ID: {reranking_id[:8]})"
        chat_messages_a.append((None, success_message_a))
        chat_messages_b.append((None, success_message_b))
        
        yield [gr.update(value=chat_messages_a, visible=True), gr.update(value=chat_messages_b, visible=True), comparison_result]
        
    except Exception as e:
        error_message = f"❌ Error during reranking: {str(e)}"
        error_outputs = (
            [gr.update(value=[(None, error_message)], visible=True)]
            + [gr.update(value=[(None, error_message)], visible=True)]
            + [""]  # Empty comparison on error
        )
        yield error_outputs

def vote_winner(winner, user_id):
    global current_reranking_session
    
    if not user_id:
        return "❌ No active session found."
    
    if user_id not in current_reranking_session:
        return "❌ No reranking data found. Please rerank first."
    
    session_data = current_reranking_session[user_id]
    
    try:
        # Save user vote first
        if winner:  # User made a choice
            vote_id, _ = save_vote_interaction(user_id, winner)
            vote_message = f"✅ Your vote for '{winner}' recorded! (ID: {vote_id[:8]})"
        else:
            vote_message = "⏭️ No vote submitted."
        
        # **ALWAYS RUN LLM EVALUATION** regardless of user vote
        llm_evaluation = None
        llm_message = ""
        
        try:
            # Prepare model info for LLM judge
            model_a_info = {
                "method": session_data["method_a"],
                "model": session_data["model_a"]
            }
            model_b_info = {
                "method": session_data["method_b"],
                "model": session_data["model_b"]
            }
            
            # Get LLM evaluation
            llm_evaluation = llm_judge.evaluate_reranking(
                session_data["query"], 
                model_a_info, 
                model_b_info, 
                session_data["reranked_contexts_a"], 
                session_data["reranked_contexts_b"]
            )
            
            # Save LLM evaluation with user vote info
            llm_vote_id = save_llm_vote_interaction(user_id, llm_evaluation, user_vote=winner)
            
            llm_winner = llm_evaluation.get('winner', 'Unknown')
            llm_reasoning = llm_evaluation.get('reasoning', 'No reasoning provided')
            
            llm_message = f"\n\n🤖 LLM Judge Result: {llm_winner}\n📝 Reasoning: {llm_reasoning}\n" #🆔 LLM Evaluation ID: {llm_vote_id[:8]}
            
            print(f"🤖 LLM Judge evaluation completed for user {user_id}: {llm_winner}")
            print(f"📁 LLM Vote saved with ID: {llm_vote_id}")
            
        except Exception as e:
            print(f"⚠️ LLM evaluation failed: {str(e)}")
            # Save error state
            error_evaluation = {
                "winner": "Error",
                "reasoning": f"LLM evaluation failed: {str(e)}",
                "full_response": f"Error: {str(e)}"
            }
            llm_vote_id = save_llm_vote_interaction(user_id, error_evaluation, user_vote=winner)
            llm_message = f"\n\n❌ **LLM Judge Error**: {str(e)}\n🆔 **Error Log ID**: {llm_vote_id[:8]}"
        
        # Get updated stats
        stats = get_user_stats(user_id)
        stats_message = f"\n📊 Your Stats: {stats['vote_count']} votes, {stats['reranking_count']} reranks, {stats['llm_vote_count']} LLM evaluations"
        
        return vote_message + llm_message + stats_message
        
    except Exception as e:
        return f"❌ Error processing vote: {str(e)}"

# ... (previous imports and functions remain the same)

def build_retriever_tab():
    with gr.Tab("🛠️ Retriever + Reranker"):
        gr.Markdown("# 🧑‍💻 Retriever & Reranker Arena")
        
        # User info display
        with gr.Row():
            user_info = gr.Markdown("👤 **Your Session ID will be generated when you start**")
        
        query_box = gr.Textbox(label="Enter your query")
        retriever_type = gr.Radio(
            ["Online Retriever", "Offline Retriever"],
            value="Offline Retriever",
            label="Retriever Type"
        )
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
        n_docs = gr.Slider(1, 10, value=3, step=1, label="Number of Documents")
        retrieve_btn = gr.Button("Retrieve Documents", variant="primary")
        retrieve_error = gr.Textbox(label="Status", visible=False)
        docs_state = gr.State(None)
        orig_contexts_state = gr.State(None)
        user_id_state = gr.State(None)

        # Retriever results (top)
        gr.Markdown("### Retrieved Contexts")
        retrieved_chat = gr.Chatbot(height=400, label="Retrieved Documents")

        def toggle_offline_fields(retriever_type):
            if retriever_type == "Offline Retriever":
                return gr.update(visible=True), gr.update(visible=True)
            else:
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

        # Use the generator function for progressive updates
        retrieve_btn.click(
            retrieve_documents,
            [query_box, retriever_type, offline_method, corpus, n_docs],
            [retrieved_chat, retrieve_error, docs_state, orig_contexts_state, user_id_state]
        )

        # Update user info when user_id changes
        user_id_state.change(
            update_user_info,
            [user_id_state],
            [user_info]
        )

        gr.Markdown("## Rerank Retrieved Documents")

        method_names = get_rankify_methods()
        with gr.Row():
            with gr.Column():
                method_a = gr.Dropdown(choices=["Select Reranker"] +method_names, label="Method for Model A")
                model_a = gr.Dropdown(choices=[], label="Model A")
            with gr.Column():
                method_b = gr.Dropdown(choices=["Select Reranker"] +method_names, label="Method for Model B")
                model_b = gr.Dropdown(choices=[], label="Model B")

        method_a.change(lambda m: gr.update(choices=get_models_for_method(m)), method_a, model_a)
        method_b.change(lambda m: gr.update(choices=get_models_for_method(m)), method_b, model_b)

        rerank_btn = gr.Button("Rerank and Show Results (with Movement Arrows)", variant="primary")

        # Reranked results: side by side
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Model A Reranked")
                rerank_chat_a = gr.Chatbot(height=400, label="Model A Reranked")
            with gr.Column():
                gr.Markdown("### Model B Reranked")
                rerank_chat_b = gr.Chatbot(height=400, label="Model B Reranked")

        # Ranking comparison section
        gr.Markdown("### 📊 Reranking Comparison Analysis")
        comparison_display = gr.HTML(value="<p style='color: #666; text-align: center; padding: 20px;'>🔄 Rerank documents to see statistical comparison between the two rerankers</p>")

        # Use the generator function for progressive updates with comparison
        rerank_btn.click(
            rerank_and_visualize,
            [query_box, docs_state, orig_contexts_state, method_a, model_a, method_b, model_b, user_id_state],
            [rerank_chat_a, rerank_chat_b, comparison_display]
        )

        # Updated voting section with Tie option
        gr.Markdown("## Vote for the Better Reranker")
        vote_btn = gr.Radio(
            choices=["Model A", "Model B", "Tie"], 
            label="Which reranker performed better? (LLM Judge will evaluate regardless of your choice)"
        )
        vote_submit = gr.Button("Submit Vote & Get LLM Judge Result", variant="secondary")
        vote_output = gr.Textbox(label="Vote & LLM Judge Results", lines=8)
        
        vote_submit.click(
            vote_winner, 
            [vote_btn, user_id_state], 
            vote_output 
        )