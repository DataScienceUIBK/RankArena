import gradio as gr
from utils import get_rankify_methods, get_models_for_method, run_rankify_reranker
from logging_utils import (
    save_reranking_interaction,
    save_vote_interaction,
    save_llm_vote_interaction,
    get_or_create_user_session,
    get_user_stats,
    llm_judge
)
import json

# Global storage for LLM evaluations (temporary until user votes)
pending_llm_evaluations = {}
# Global storage to track which queries have been voted on by each user
voted_queries = {}

def parse_inputs(doc_mode, doc_file, doc_text1, doc_text2, *extra_docs):
    if doc_mode == "Upload JSON":
        if doc_file is None:
            return None, [], "⚠️ No file uploaded."
        try:
            with open(doc_file.name, "r", encoding="utf-8") as f:
                data = json.load(f)
                query = data.get("query", "")
                docs = data.get("documents", [])
                return query, docs, None
        except Exception as e:
            return None, [], f"❌ Failed to parse uploaded JSON: {e}"
    else:
        docs = [doc_text1, doc_text2] + list(extra_docs)
        docs = [d for d in docs if d.strip()]
        return None, docs, None

def compare_rankings(original_docs, reranked_a, reranked_b):
    """Compare two reranking results using statistical methods"""
    if not original_docs or not reranked_a or not reranked_b:
        return "❌ No data to compare"
    
    # Get positions in reranked results
    a_positions = []
    b_positions = []
    
    for doc in original_docs:
        # Find position in reranked A
        try:
            pos_a = reranked_a.index(doc)
            a_positions.append(pos_a + 1)  # 1-indexed for better interpretation
        except ValueError:
            a_positions.append(len(original_docs) + 1)  # Put unranked docs at the end
        
        # Find position in reranked B
        try:
            pos_b = reranked_b.index(doc)
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
    
    rbo_score = rank_biased_overlap(reranked_a, reranked_b)
    
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
    
    top1_overlap = top_k_overlap(reranked_a, reranked_b, 1)
    top3_overlap = top_k_overlap(reranked_a, reranked_b, 3) if total_docs >= 3 else 0
    top5_overlap = top_k_overlap(reranked_a, reranked_b, 5) if total_docs >= 5 else 0
    
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
            📊 Statistical Ranking Comparison Analysis
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
        interpretation = "🟢 <strong>High Similarity:</strong> Both rerankers show very similar preferences and ranking patterns."
    elif spearman_corr > 0.3:
        interpretation = "🟡 <strong>Moderate Similarity:</strong> Rerankers agree on some documents but differ on others."
    elif spearman_corr > 0:
        interpretation = "🟠 <strong>Low Similarity:</strong> Rerankers have different ranking strategies."
    else:
        interpretation = "🔴 <strong>Opposite Preferences:</strong> Rerankers show inverse ranking patterns."
    
    comparison_html += f"""
                    <p style="margin: 0;">{interpretation}</p>
                    <p style="margin: 8px 0 0 0; color: #6b7280; font-size: 13px;">
                        💡 <strong>Tip:</strong> Values closer to 1.0 indicate more similar rankings. 
                        Spearman correlation measures rank-order similarity.
                    </p>
                </div>
            </div>
        </div>
        
        <div style="margin-top: 20px; padding: 15px; background-color: #f1f5f9; border-radius: 8px; border-left: 4px solid #3b82f6;">
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; font-size: 14px;">
                <div><strong>Total Documents:</strong> {total_docs}</div>
                <div><strong>Statistical Method:</strong> {'Spearman + Kendall' if scipy_available else 'Pearson (fallback)'}</div>
                <div><strong>RBO Parameter:</strong> p=0.9</div>
            </div>
        </div>
    </div>
    """
    
    return comparison_html

def model_response(prompt, doc_texts, method, model):
    """Get reranked results from a specific method and model"""
    reranked_docs = run_rankify_reranker(prompt, doc_texts, method, model)
    return reranked_docs

def handle_chat(prompt, hist_a, hist_b, method_a, model_a, method_b, model_b,
                doc_mode, doc_file, doc_text1, doc_text2, user_id, *extra_docs ,
                hide_labels=False):
    # Get or create user session
    if not user_id:
        user_id = get_or_create_user_session()
    
    # Show initial loading state
    loading_a = [(None, "🔄 Processing with Model A...")]
    loading_b = [(None, "🔄 Processing with Model B...")]
    yield loading_a, loading_b, user_id, "", True, ""  # Added empty string for comparison
    
    # Parse inputs
    uploaded_query, doc_texts, error = parse_inputs(doc_mode, doc_file, doc_text1, doc_text2, *extra_docs)
    if error:
        error_a = [(None, error)]
        error_b = [(None, error)]
        yield error_a, error_b, user_id, "", True, ""
        return
    
    if not doc_texts:
        error_msg = "❌ Please provide documents to rerank."
        error_a = [(None, error_msg)]
        error_b = [(None, error_msg)]
        yield error_a, error_b, user_id, "", True, ""
        return
    
    query = uploaded_query if uploaded_query is not None else prompt
    
    try:
        # Update loading for Model A
        if hide_labels:
            progress_a = [(None, "🔄 Running Reranker A...")]
            progress_b = [(None, "🔄 Running Reranker B...")]
        else:
            progress_a = [(None, f"🔄 Running {method_a} with {model_a}...")]
            progress_b = [(None, "🔄 Waiting for Model A...")]
        yield progress_a, progress_b, user_id, "", True, ""
        
        # Get results from Model A
        reranked_docs_a = model_response(query, doc_texts, method_a, model_a)
        
        # Update loading for Model B
        if hide_labels:
            progress_a = [(None, "✅ Model A Complete")]
            progress_b = [(None, "🔄 Running Reranker B...")]
        else:
            progress_a = [(None, "✅ Model A Complete")]
            progress_b = [(None, f"🔄 Running {method_b} with {model_b}...")]
        yield progress_a, progress_b, user_id, "", True, ""
        
        # Get results from Model B
        reranked_docs_b = model_response(query, doc_texts, method_b, model_b)
        
        # Generate ranking comparison
        comparison_result = compare_rankings(doc_texts, reranked_docs_a, reranked_docs_b)
        
        # Convert doc_texts to Context-like objects for logging
        original_contexts = []
        for i, doc_text in enumerate(doc_texts):
            class SimpleContext:
                def __init__(self, text, idx):
                    self.id = f"doc_{idx}"
                    self.title = f"Document {idx+1}"
                    self.text = text
                    self.score = 0.0
            original_contexts.append(SimpleContext(doc_text, i))
        
        # Convert reranked results to Context-like objects
        reranked_contexts_a = []
        for i, doc_text in enumerate(reranked_docs_a):
            class SimpleContext:
                def __init__(self, text, idx):
                    orig_idx = next((j for j, orig in enumerate(doc_texts) if orig == text), idx)
                    self.id = f"doc_{orig_idx}"
                    self.title = f"Document {orig_idx+1}"
                    self.text = text
                    self.score = 1.0 - (i * 0.1)  # Simple decreasing score
            reranked_contexts_a.append(SimpleContext(doc_text, i))
        
        reranked_contexts_b = []
        for i, doc_text in enumerate(reranked_docs_b):
            class SimpleContext:
                def __init__(self, text, idx):
                    orig_idx = next((j for j, orig in enumerate(doc_texts) if orig == text), idx)
                    self.id = f"doc_{orig_idx}"
                    self.title = f"Document {orig_idx+1}"
                    self.text = text
                    self.score = 1.0 - (i * 0.1)  # Simple decreasing score
            reranked_contexts_b.append(SimpleContext(doc_text, i))
        
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
        
        # Trigger LLM evaluation in background
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
        
        # Format the reranked documents for display (similar to retriever_tab)
        query_normalized = query.lower().strip()
        
        # Model A response
        combined_info_a = []
        doc_counter = 1
        for doc in reranked_docs_a:
            text = doc.strip()
            print(f"After cleaning (rerank A) - Doc {doc_counter}, Text: {text}")  # Debug
            
            info = (
                f"**Document {doc_counter}**\n"
                f"**Text:** {text}\n"
                f"**Score:** {1.0 - (doc_counter - 1) * 0.1}"
            )
            combined_info_a.append(info)
            doc_counter += 1
        
        a_resp = f"**Query:** {query}\n\n" + "\n\n".join(combined_info_a) if combined_info_a else f"**Query:** {query}\n\nNo documents reranked."
        chat_messages_a = [(None, a_resp)]
        
        # Model B response
        combined_info_b = []
        doc_counter = 1
        for doc in reranked_docs_b:
            text = doc.strip()
           
            info = (
                f"**Document {doc_counter}**\n"
                f"**Text:** {text}\n"
                f"**Score:** {1.0 - (doc_counter - 1) * 0.1}"
            )
            combined_info_b.append(info)
            doc_counter += 1
        
        b_resp = f"**Query:** {query}\n\n" + "\n\n".join(combined_info_b) if combined_info_b else f"**Query:** {query}\n\nNo documents reranked."
        chat_messages_b = [(None, b_resp)]
        
        # Append success message
        success_message = f"**{method_a}::{model_a}** - ✅ Reranking complete (ID: {reranking_id[:8]})"
        chat_messages_a.append((None, success_message))
        success_message = f"**{method_b}::{model_b}** - ✅ Reranking complete (ID: {reranking_id[:8]})"
        chat_messages_b.append((None, success_message))
        
        yield chat_messages_a, chat_messages_b, user_id, reranking_id, True, comparison_result  # Added comparison result
        
    except Exception as e:
        error_msg = f"❌ Error during reranking: {str(e)}"
        error_a = [(None, error_msg)]
        error_b = [(None, error_msg)]
        yield error_a, error_b, user_id, "", True, ""

def vote_winner(model_a, model_b, winner, user_id, reranking_id):
    """Handle voting with proper user session management and LLM comparison"""
    if not user_id or not winner:
        return "❌ Please select a winner to vote.", True  # Button remains enabled
    
    # Check if the user has already voted for this query (using reranking_id as a proxy for query)
    user_query_key = f"{user_id}_{reranking_id}"
    # if user_query_key in voted_queries:
    #     return "❌ You have already voted for this query.", False  # Button remains disabled
    
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
            
            llm_comparison = f"\n\n🤖 LLM Judge: {llm_winner} | {agreement_status}\n📝 Reasoning: {llm_reasoning}"
            
            # Clean up pending evaluation
            del pending_llm_evaluations[user_id]
        
        stats = get_user_stats(user_id)
        base_message = f"✅ Vote recorded! (ID: {vote_id[:8]}) | Your stats: {stats['vote_count']} votes, {stats['reranking_count']} reranks"
        
        return base_message + llm_comparison, False  # Disable the button after voting
        
    except Exception as e:
        return f"❌ Error saving vote: {str(e)}", True  # Keep button enabled if there's an error

def toggle_doc_inputs(mode, current_count, extra_textboxes):
    """Toggle between JSON upload and manual document input"""
    if mode == "Upload JSON":
        file_vis = gr.update(visible=True)
        info_vis = gr.update(visible=True)
        tb_updates = [gr.update(visible=False) for _ in range(2)]
        tb_updates += [gr.update(visible=False) for _ in extra_textboxes]
        add_btn = gr.update(visible=False)
        return [file_vis, info_vis] + tb_updates + [add_btn, 0]
    else:
        file_vis = gr.update(visible=False)
        info_vis = gr.update(visible=False)
        tb_updates = [gr.update(visible=True), gr.update(visible=True)]  # Always show doc_text_1 and doc_text_2
        # Fixed: Show all textboxes up to current_count, not just one
        tb_updates += [gr.update(visible=(i < current_count)) for i in range(len(extra_textboxes))]
        add_btn = gr.update(visible=current_count < len(extra_textboxes))
        return [file_vis, info_vis] + tb_updates + [add_btn, current_count]

def add_textbox(current_count, extra_textboxes):
    """Add additional document textbox"""
    # The issue: you're setting visibility based on new_count - 1, but you should show the textbox at index current_count
    new_count = min(current_count + 1, len(extra_textboxes))
    
    # Fixed: Show all textboxes up to new_count (cumulative visibility)
    tb_updates = [gr.update(visible=(i < new_count)) for i in range(len(extra_textboxes))]
    
    # Hide add button when we reach maximum
    add_btn = gr.update(visible=new_count < len(extra_textboxes))
    
    return tb_updates + [add_btn, new_count]

def update_user_info(user_id):
    """Update user information display"""
    if user_id:
        stats = get_user_stats(user_id)
        return gr.update(value=f"👤 **Session: {user_id}** ") #| Reranks: {stats['reranking_count']} | Votes: {stats['vote_count']}
    return gr.update(value="👤 **Your Session ID will be generated when you start**")

def build_reranker_tab():
    with gr.Tab("⚔️ 1v1 Reranker"):
        gr.Markdown("# 🔍 RerankArena: Compare Two Rankify Rerankers")
        
        # User info display
        with gr.Row():
            user_info = gr.Markdown("👤 **Your Session ID will be generated when you start**")
        
        # Model selection
        method_names = get_rankify_methods()
        with gr.Row():
            with gr.Column():
                method_a = gr.Dropdown(choices=["Select Reranker"] +method_names, label="Method for Model A")
                model_a = gr.Dropdown(choices=[], label="Model A")
            with gr.Column():
                method_b = gr.Dropdown(choices=["Select Reranker"] +method_names, label="Method for Model B")
                model_b = gr.Dropdown(choices=[], label="Model B")
        
        # Chat interface
        with gr.Row():
            chat_a = gr.Chatbot(label="Model A", height=400)
            chat_b = gr.Chatbot(label="Model B", height=400)
        
        # User input
        user_input = gr.Textbox(placeholder="Enter your query...", label="Query")
        
        # Document input mode with info
        with gr.Row():
            doc_mode = gr.Radio(
                choices=["Upload JSON", "Write Documents"],
                value="Upload JSON",
                label="Document Input Mode"
            )
        
        # JSON info section
        with gr.Row():
            with gr.Column(scale=3):
                # Document inputs
                doc_json = gr.File(label="Upload JSON file", visible=True)
            with gr.Column(scale=1):
                # Info section for JSON format
                json_info = gr.HTML("""
                <div style="background-color: #e8f4fd; border: 1px solid #b3d9ff; border-radius: 8px; padding: 15px; margin: 10px 0;">
                    <div style="display: flex; align-items: center; margin-bottom: 10px;">
                        <span style="font-size: 20px; margin-right: 8px;">ℹ️</span>
                        <strong style="color: #1f5582;">JSON File Format</strong>
                    </div>
                    <div style="font-family: monospace; background-color: #f8f9fa; padding: 10px; border-radius: 4px; font-size: 12px; border: 1px solid #dee2e6;">
<pre>{
  "query": "who is barack obama?",
  "documents": [
    "Barack Obama is former president of USA",
    "Abdelrahman Abdallah is a researcher at computer science"
  ]
}</pre>
                    </div>
                    <small style="color: #6c757d; margin-top: 5px; display: block;">
                        💡 <strong>Required fields:</strong><br>
                        • "query": your search question<br>
                        • "documents": array of text documents
                    </small>
                </div>
                """, visible=True)
        
        doc_text_1 = gr.Textbox(label="Document Text 1", visible=False, lines=3)
        doc_text_2 = gr.Textbox(label="Document Text 2", visible=False, lines=3)

        extra_textboxes = [gr.Textbox(label=f"Document Text {i + 3}", visible=False, lines=3) for i in range(5)]

        add_text_btn = gr.Button("Add More Documents", visible=False)
        extra_count = gr.State(0)
        user_id_state = gr.State(None)
        reranking_id_state = gr.State("")  # New state to track reranking_id
        vote_enabled_state = gr.State(True)  # New state to track if voting is enabled
        
        # Submit button
        submit_btn = gr.Button("Submit", variant="primary")
        
        # Ranking comparison section
        gr.Markdown("### 📊 Ranking Comparison Analysis")
        comparison_display = gr.HTML(value="<p style='color: #666; text-align: center; padding: 20px;'>🔄 Submit a query to see how the two rerankers differ in their document ordering</p>")
        
        # Voting section
        gr.Markdown("### Vote for Better Model")
        vote_btn = gr.Radio(choices=["Model A", "Model B", "Tie"], label="Which model is better?")
        vote_submit = gr.Button("Vote", variant="secondary", interactive=True)  # Initially enabled
        vote_output = gr.Textbox(label="Vote Result")
        
        # Event handlers
        def toggle_doc_inputs_with_info(mode, count):
            """Toggle between JSON upload and manual document input, including info visibility"""
            if mode == "Upload JSON":
                file_vis = gr.update(visible=True)
                info_vis = gr.update(visible=True)
                tb_updates = [gr.update(visible=False) for _ in range(2)]
                tb_updates += [gr.update(visible=False) for _ in extra_textboxes]
                add_btn = gr.update(visible=False)
                return [file_vis, info_vis] + tb_updates + [add_btn, 0]
            else:
                file_vis = gr.update(visible=False)
                info_vis = gr.update(visible=False)
                tb_updates = [gr.update(visible=True), gr.update(visible=True)]  # Always show doc_text_1 and doc_text_2
                # Fixed: Show all textboxes up to current_count, not just one
                tb_updates += [gr.update(visible=(i < count)) for i in range(len(extra_textboxes))]
                add_btn = gr.update(visible=count < len(extra_textboxes))
                return [file_vis, info_vis] + tb_updates + [add_btn, count]
        
        doc_mode.change(
            lambda mode, count: toggle_doc_inputs_with_info(mode, count),
            [doc_mode, extra_count],
            [doc_json, json_info, doc_text_1, doc_text_2] + extra_textboxes + [add_text_btn, extra_count]
        )
        
        add_text_btn.click(
            lambda count: add_textbox(count, extra_textboxes),
            [extra_count],
            extra_textboxes + [add_text_btn, extra_count]
        )
        
        method_a.change(lambda m: gr.update(choices=get_models_for_method(m)), method_a, model_a)
        method_b.change(lambda m: gr.update(choices=get_models_for_method(m)), method_b, model_b)
        
        # Use generator function for progressive updates
        submit_btn.click(
            fn=handle_chat,
            inputs=[
                user_input, chat_a, chat_b,
                method_a, model_a, method_b, model_b,
                doc_mode, doc_json, doc_text_1, doc_text_2, user_id_state
            ] + extra_textboxes,
            outputs=[chat_a, chat_b, user_id_state, reranking_id_state, vote_enabled_state, comparison_display]
        ).then(
            lambda vote_enabled: gr.update(interactive=vote_enabled),
            inputs=[vote_enabled_state],
            outputs=[vote_submit]
        )
        
        # Update user info when user_id changes
        user_id_state.change(
            update_user_info,
            [user_id_state],
            [user_info]
        )
        
        # Vote handling with LLM comparison
        vote_submit.click(
            vote_winner, 
            [model_a, model_b, vote_btn, user_id_state, reranking_id_state], 
            [vote_output, vote_enabled_state]
        ).then(
            lambda vote_enabled: gr.update(interactive=vote_enabled),
            [vote_enabled_state],
            [vote_submit]
        )