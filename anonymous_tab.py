#anonymous_tab
import gradio as gr
import random
from utils import get_rankify_methods, get_models_for_method
from reranker_tab import handle_chat, add_textbox, vote_winner
from logging_utils import (
    get_or_create_user_session,
    get_user_stats
)

def compare_rankings(original_docs, reranked_a, reranked_b):
    """Compare two reranking results using statistical methods - same as 1v1 tab"""
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
    
    # Build comparison visualization (anonymous style)
    comparison_html = f"""
    <div style="font-family: Arial, sans-serif; border: 1px solid #ddd; padding: 20px; border-radius: 12px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);">
        <h3 style="color: #1f2937; margin-bottom: 20px; border-bottom: 2px solid #6366f1; padding-bottom: 10px;">
            🎭 Anonymous Reranker Comparison Analysis
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
                <h4 style="color: #1f5582; margin: 0 0 15px 0; font-size: 16px;">💡 Anonymous Interpretation</h4>
                <div style="font-size: 14px; line-height: 1.5;">"""
    
    # Add interpretation based on the metrics
    if spearman_corr > 0.7:
        interpretation = "🟢 <strong>High Similarity:</strong> Both anonymous rerankers show very similar ranking strategies."
    elif spearman_corr > 0.3:
        interpretation = "🟡 <strong>Moderate Similarity:</strong> Anonymous rerankers have some overlap but differ in their approaches."
    elif spearman_corr > 0:
        interpretation = "🟠 <strong>Low Similarity:</strong> Anonymous rerankers use quite different ranking methods."
    else:
        interpretation = "🔴 <strong>Opposite Preferences:</strong> Anonymous rerankers show contrasting ranking behaviors."
    
    comparison_html += f"""
                    <p style="margin: 0;">{interpretation}</p>
                    <p style="margin: 8px 0 0 0; color: #6b7280; font-size: 13px;">
                        🎭 <strong>Anonymous Mode:</strong> This analysis compares two randomly selected rerankers 
                        without revealing their identities until after voting.
                    </p>
                </div>
            </div>
        </div>
        
        <div style="margin-top: 20px; padding: 15px; background-color: #fef3e2; border-radius: 8px; border-left: 4px solid #f59e0b;">
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; font-size: 14px;">
                <div><strong>Total Documents:</strong> {total_docs}</div>
                <div><strong>Statistical Method:</strong> {'Spearman + Kendall' if scipy_available else 'Pearson (fallback)'}</div>
                <div><strong>Mode:</strong> 🎭 Anonymous Comparison</div>
            </div>
        </div>
    </div>
    """
    
    return comparison_html

def pick_random_rerankers():
    methods = get_rankify_methods()
    method_a = random.choice(methods)
    model_a = random.choice(get_models_for_method(method_a))

    method_b = random.choice(methods)
    model_b = random.choice(get_models_for_method(method_b))

    while method_b == method_a and model_b == model_a:
        method_b = random.choice(methods)
        model_b = random.choice(get_models_for_method(method_b))
    print(f"Selected Method A: {method_a} ({model_a}), Method B: {method_b} ({model_b})")
    return method_a, model_a, method_b, model_b

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
        tb_updates += [gr.update(visible=(i < current_count)) for i in range(len(extra_textboxes))]
        add_btn = gr.update(visible=current_count < len(extra_textboxes))
        return [file_vis, info_vis] + tb_updates + [add_btn, current_count]

def update_user_info(user_id):
    """Update user information display"""
    if user_id:
        stats = get_user_stats(user_id)
        return gr.update(value=f"👤 **Session: {user_id}** | Reranks: {stats['reranking_count']} | Votes: {stats['vote_count']}")
    return gr.update(value="👤 **Your Session ID will be generated when you start**")

def build_anonymous_reranker_tab():
    with gr.Tab("👤 Anonymous Reranker"):
        gr.Markdown("### 🎭 Anonymous Reranker Battle (Blind Comparison)")
        
        # User info display (same as other tabs)
        with gr.Row():
            user_info = gr.Markdown("👤 **Your Session ID will be generated when you start**")

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
        
        # JSON info section (same as 1v1 tab)
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

        # State management
        user_id_state = gr.State(None)
        reranking_id_state = gr.State("")
        vote_enabled_state = gr.State(True)
        
        # Hidden states to store actual model info for voting
        actual_method_a = gr.State("")
        actual_model_a = gr.State("")
        actual_method_b = gr.State("")
        actual_model_b = gr.State("")

        # Submit button
        submit_btn = gr.Button("Submit", variant="primary")

        # Ranking comparison section
        gr.Markdown("### 📊 Anonymous Ranking Comparison Analysis")
        comparison_display = gr.HTML(value="<p style='color: #666; text-align: center; padding: 20px;'>🎭 Submit a query to see statistical comparison between two anonymous rerankers</p>")

        # Voting section
        gr.Markdown("### Vote for Better Model")
        vote_btn = gr.Radio(choices=["Model A", "Model B", "Tie"], label="Which model is better?")
        vote_submit = gr.Button("Vote", variant="secondary", interactive=True)
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

        # Hidden reranker execution with full session management
        def run_anonymous_chat(query, hist_a, hist_b, doc_mode, doc_file, doc1, doc2, uid, *extras):
            # Get or create user session
            if not uid:
                uid = get_or_create_user_session()
            
            # Pick random rerankers
            method_a, model_a, method_b, model_b = pick_random_rerankers()
            
            # Use the handle_chat function from reranker_tab with hidden labels
            for result in handle_chat(
                query, hist_a, hist_b, method_a, model_a, method_b, model_b,
                doc_mode, doc_file, doc1, doc2, uid, *extras,
                hide_labels=True  # Hide model names for anonymous comparison
            ):
                if len(result) == 6:  # Updated return: chat_a, chat_b, user_id, reranking_id, vote_enabled, comparison
                    chat_a_msgs, chat_b_msgs, user_id, reranking_id, vote_enabled, comparison = result
                    yield chat_a_msgs, chat_b_msgs, user_id, reranking_id, vote_enabled, method_a, model_a, method_b, model_b, comparison
                elif len(result) == 5:  # Fallback for older version without comparison
                    chat_a_msgs, chat_b_msgs, user_id, reranking_id, vote_enabled = result
                    yield chat_a_msgs, chat_b_msgs, user_id, reranking_id, vote_enabled, method_a, model_a, method_b, model_b, ""
                else:
                    # Handle any unexpected return format
                    yield result + (method_a, model_a, method_b, model_b, "")

        # Enhanced vote winner function for anonymous mode
        def vote_winner_anonymous(model_a, model_b, winner, user_id, reranking_id):
            """Handle voting in anonymous mode with actual model info"""
            if not user_id or not winner:
                return "❌ Please select a winner to vote.", True
            
            print(f"Anonymous vote: winner={winner}, user_id={user_id}, reranking_id={reranking_id}")
            print(f"Models: A={model_a}, B={model_b}")
            
            # Use the actual model info for voting (stored in hidden states)
            return vote_winner(model_a, model_b, winner, user_id, reranking_id)

        # Submit handler
        submit_btn.click(
            fn=run_anonymous_chat,
            inputs=[
                user_input, chat_a, chat_b,
                doc_mode, doc_json, doc_text_1, doc_text_2, user_id_state
            ] + extra_textboxes,
            outputs=[
                chat_a, chat_b, user_id_state, reranking_id_state, vote_enabled_state,
                actual_method_a, actual_model_a, actual_method_b, actual_model_b, comparison_display
            ]
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

        # Vote handling with actual model info
        vote_submit.click(
            lambda model_a, model_b, winner, user_id, reranking_id: vote_winner_anonymous(model_a, model_b, winner, user_id, reranking_id),
            [actual_model_a, actual_model_b, vote_btn, user_id_state, reranking_id_state],
            [vote_output, vote_enabled_state]
        ).then(
            lambda vote_enabled: gr.update(interactive=vote_enabled),
            [vote_enabled_state],
            [vote_submit]
        )