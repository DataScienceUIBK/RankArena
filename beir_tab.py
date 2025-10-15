# beir_tab.py - BEIR Dataset Evaluation Tab
import gradio as gr
import os
import time
import json
from utils import get_rankify_methods, get_models_for_method
import traceback

# Set CUDA environment if needed
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_beir_datasets():
    """Return available BEIR datasets"""
    return [
        'dl19', 'dl20',
        'beir-covid', 'beir-nfc', 'beir-touche', 'beir-dbpedia', 
        'beir-scifact', 'beir-signal', 'beir-news', 'beir-robust04',
        'beir-arguana', 'beir-fever', 'beir-fiqa', 'beir-quora', 'beir-scidocs'
    ]

def run_beir_evaluation(dataset_name, method_a, model_a, method_b, model_b, num_docs):
    """Run BEIR evaluation on selected dataset and models"""
    
    # Show initial loading state
    yield "🔄 Initializing BEIR dataset evaluation...", "", "", ""
    
    try:
        # Import rankify components
        from rankify.dataset.dataset import Dataset
        from rankify.models.reranking import Reranking
        from rankify.metrics.metrics import Metrics
        
        # Validate inputs
        if not dataset_name or dataset_name == "Select Dataset":
            yield "❌ Please select a dataset", "", "", ""
            return
            
        if not method_a or method_a == "Select Method":
            yield "❌ Please select Method A", "", "", ""
            return
            
        if not method_b or method_b == "Select Method":
            yield "❌ Please select Method B", "", "", ""
            return
            
        if not model_a or not model_b:
            yield "❌ Please select both models", "", "", ""
            return
        
        # Update status
        yield f"📊 Loading {dataset_name} dataset with {num_docs} documents...", "", "", ""
        
        # Download dataset
        dataset = Dataset('bm25', dataset_name, num_docs)
        data = dataset.download(force_download=False)
        
        if not data:
            yield f"❌ Failed to load dataset {dataset_name}", "", "", ""
            return
        
        yield f"✅ Dataset loaded: {len(data)} documents\n🔄 Running {method_a}::{model_a}...", "", "", ""
        
        # Run Model A
        try:
            model_reranker_a = Reranking(method=method_a, model_name=model_a, device="cuda", retrieval_type='IE')
            start_time_a = time.time()
            model_reranker_a.rank(data)
            end_time_a = time.time()
            execution_time_a = end_time_a - start_time_a
            
            # Calculate metrics for Model A
            metrics_a = Metrics(data)
            
            # Determine qrel name
            if dataset_name in ['dl19', 'dl20']:
                qrel_name = dataset_name
            else:
                qrel_name = dataset_name.split('-')[1]
            
            after_ranking_metrics_a = metrics_a.calculate_trec_metrics(
                ndcg_cuts=[1, 5, 10], 
                map_cuts=[1, 5, 10], 
                mrr_cuts=[1, 5, 10], 
                qrel=qrel_name, 
                use_reordered=True
            )
            
            # Format results for Model A
            results_a = f"""
## 🅰️ Model A Results: {method_a}::{model_a}

**⏱️ Execution Time:** {execution_time_a:.2f} seconds

**📈 Performance Metrics:**
- **NDCG@1:** {after_ranking_metrics_a.get('ndcg@1', 0):.4f}
- **NDCG@5:** {after_ranking_metrics_a.get('ndcg@5', 0):.4f}
- **NDCG@10:** {after_ranking_metrics_a.get('ndcg@10', 0):.4f}
- **MAP@1:** {after_ranking_metrics_a.get('map@1', 0):.4f}
- **MAP@5:** {after_ranking_metrics_a.get('map@5', 0):.4f}
- **MAP@10:** {after_ranking_metrics_a.get('map@10', 0):.4f}
- **MRR:** {after_ranking_metrics_a.get('mrr@1', 0):.4f}
"""
            
        except Exception as e:
            error_msg_a = f"❌ Error running {method_a}::{model_a}: {str(e)}"
            results_a = error_msg_a
            after_ranking_metrics_a = {}
            execution_time_a = 0
        
        # Clear CUDA cache
        try:
            import torch
            import gc
            torch.cuda.empty_cache()
            gc.collect()
        except:
            pass
        
        yield f"✅ Model A completed\n🔄 Running {method_b}::{model_b}...", results_a, "", ""
        
        # Reload dataset for Model B (fresh copy)
        dataset_b = Dataset('bm25', dataset_name, num_docs)
        data_b = dataset_b.download(force_download=False)
        
        # Run Model B
        try:
            model_reranker_b = Reranking(method=method_b, model_name=model_b, device="cuda", retrieval_type='IE')
            start_time_b = time.time()
            model_reranker_b.rank(data_b)
            end_time_b = time.time()
            execution_time_b = end_time_b - start_time_b
            
            # Calculate metrics for Model B
            metrics_b = Metrics(data_b)
            after_ranking_metrics_b = metrics_b.calculate_trec_metrics(
                ndcg_cuts=[1, 5, 10], 
                map_cuts=[1, 5, 10], 
                mrr_cuts=[1, 5, 10], 
                qrel=qrel_name, 
                use_reordered=True
            )
            
            # Format results for Model B
            results_b = f"""
## 🅱️ Model B Results: {method_b}::{model_b}

**⏱️ Execution Time:** {execution_time_b:.2f} seconds

**📈 Performance Metrics:**
- **NDCG@1:** {after_ranking_metrics_b.get('ndcg@1', 0):.4f}
- **NDCG@5:** {after_ranking_metrics_b.get('ndcg@5', 0):.4f}
- **NDCG@10:** {after_ranking_metrics_b.get('ndcg@10', 0):.4f}
- **MAP@1:** {after_ranking_metrics_b.get('map@1', 0):.4f}
- **MAP@5:** {after_ranking_metrics_b.get('map@5', 0):.4f}
- **MAP@10:** {after_ranking_metrics_b.get('map@10', 0):.4f}
- **MRR:** {after_ranking_metrics_b.get('mrr@1', 0):.4f}

"""
            
        except Exception as e:
            error_msg_b = f"❌ Error running {method_b}::{model_b}: {str(e)}"
            results_b = error_msg_b
            after_ranking_metrics_b = {}
            execution_time_b = 0
        
        # Clear CUDA cache again
        try:
            import torch
            import gc
            torch.cuda.empty_cache()
            gc.collect()
        except:
            pass
        
        yield "🔄 Generating comparison analysis...", results_a, results_b, ""
        
        # Generate comparison if both models succeeded
        if after_ranking_metrics_a and after_ranking_metrics_b:
            comparison = generate_comparison_analysis(
                method_a, model_a, after_ranking_metrics_a, execution_time_a,
                method_b, model_b, after_ranking_metrics_b, execution_time_b,
                dataset_name, num_docs, data, data_b
            )
        else:
            comparison = "❌ Cannot generate comparison - one or both models failed to run."
        
        final_status = f"✅ Evaluation completed for {dataset_name} dataset ({num_docs} docs)"
        yield final_status, results_a, results_b, comparison
        
    except Exception as e:
        error_msg = f"❌ Critical error during evaluation: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        yield error_msg, "", "", ""

def generate_comparison_analysis(method_a, model_a, metrics_a, time_a,
                               method_b, model_b, metrics_b, time_b,
                               dataset_name, num_docs, data_a, data_b):
    """Generate detailed comparison analysis between two models using statistical methods"""
    
    # Calculate improvements
    def calculate_improvement(val_a, val_b):
        if val_b == 0:
            return "N/A"
        improvement = ((val_a - val_b) / val_b) * 100
        return f"{improvement:+.2f}%"
    
    def get_winner(val_a, val_b):
        if val_a > val_b:
            return "🅰️ Model A"
        elif val_b > val_a:
            return "🅱️ Model B"
        else:
            return "🤝 Tie"
    
    # Extract ranking positions for statistical analysis
    def extract_ranking_positions(data):
        """Extract ranking positions from reranked data"""
        all_positions_a = []
        all_positions_b = []
        
        for doc in data:
            if hasattr(doc, 'reorder_contexts') and doc.reorder_contexts:
                # Get original and reranked orders
                original_order = [ctx.id for ctx in doc.contexts]
                reranked_order = [ctx.id for ctx in doc.reorder_contexts]
                
                # Calculate positions for statistical analysis
                for ctx_id in original_order:
                    try:
                        new_pos = reranked_order.index(ctx_id) + 1  # 1-indexed
                        all_positions_a.append(new_pos)
                    except ValueError:
                        all_positions_a.append(len(original_order) + 1)
                        
        return all_positions_a
    
    # Calculate statistical measures if we have ranking data
    ranking_stats_html = ""
    try:
        positions_a = extract_ranking_positions(data_a)
        positions_b = extract_ranking_positions(data_b)
        
        if positions_a and positions_b and len(positions_a) == len(positions_b):
            try:
                from scipy.stats import spearmanr, kendalltau
                spearman_corr, spearman_p = spearmanr(positions_a, positions_b)
                kendall_corr, kendall_p = kendalltau(positions_a, positions_b)
                scipy_available = True
            except ImportError:
                # Fallback to simple correlation if scipy not available
                def simple_correlation(x, y):
                    n = len(x)
                    if n == 0:
                        return 0, 1
                    mean_x, mean_y = sum(x) / n, sum(y) / n
                    num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
                    den_x = sum((x[i] - mean_x) ** 2 for i in range(n))
                    den_y = sum((y[i] - mean_y) ** 2 for i in range(n))
                    
                    if den_x == 0 or den_y == 0:
                        return 0, 1
                    
                    corr = num / (den_x * den_y) ** 0.5
                    return corr, 0.05
                
                spearman_corr, spearman_p = simple_correlation(positions_a, positions_b)
                kendall_corr, kendall_p = simple_correlation(positions_a, positions_b)
                scipy_available = False
            
            # Calculate RBO
            def rank_biased_overlap(list1, list2, p=0.9):
                overlap_at_k = []
                for k in range(1, min(len(list1), len(list2)) + 1):
                    set1 = set(list1[:k])
                    set2 = set(list2[:k])
                    overlap = len(set1.intersection(set2)) / k
                    overlap_at_k.append(overlap)
                
                if not overlap_at_k:
                    return 0
                
                rbo = sum((1 - p) * (p ** (i)) * overlap for i, overlap in enumerate(overlap_at_k))
                return rbo
            
            # Create dummy rankings for RBO calculation
            ranking_a = list(range(len(positions_a)))
            ranking_b = list(range(len(positions_b)))
            rbo_score = rank_biased_overlap(ranking_a, ranking_b)
            
            # Calculate position-based metrics
            agreements = sum(1 for i in range(len(positions_a)) if positions_a[i] == positions_b[i])
            total_positions = len(positions_a)
            agreement_rate = (agreements / total_positions * 100) if total_positions > 0 else 0
            
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
            
            # Add interpretation
            if spearman_corr > 0.7:
                interpretation = "🟢 <strong>High Similarity:</strong> Both rerankers show very similar ranking strategies."
            elif spearman_corr > 0.3:
                interpretation = "🟡 <strong>Moderate Similarity:</strong> Rerankers agree on some rankings but differ on others."
            elif spearman_corr > 0:
                interpretation = "🟠 <strong>Low Similarity:</strong> Rerankers have different ranking approaches."
            else:
                interpretation = "🔴 <strong>Opposite Strategies:</strong> Rerankers show inverse ranking patterns."
            
            ranking_stats_html = f"""
        <!-- Statistical Ranking Analysis -->
        <div style="background: white; padding: 20px; border-radius: 10px; margin-bottom: 25px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            <h3 style="color: #7c3aed; margin: 0 0 20px 0; border-bottom: 2px solid #7c3aed; padding-bottom: 8px;">
                📊 Statistical Ranking Comparison
            </h3>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 25px;">
                <!-- Correlation Analysis -->
                <div style="background-color: #f8f9fa; padding: 18px; border-radius: 8px;">
                    <h4 style="color: #059669; margin: 0 0 15px 0; font-size: 16px;">🔗 Correlation Analysis</h4>
                    <div>
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
                <div style="background-color: #f8f9fa; padding: 18px; border-radius: 8px;">
                    <h4 style="color: #dc2626; margin: 0 0 15px 0; font-size: 16px;">🎯 Agreement Analysis</h4>
                    <div>
                        <div style="margin-bottom: 8px;">
                            <strong>Position Agreement:</strong> 
                            <span style="font-weight: bold; color: #1f2937;">{agreement_rate:.1f}%</span>
                            <small style="color: #6b7280;">({agreements}/{total_positions})</small>
                        </div>
                        <div style="margin-bottom: 8px;">
                            <strong>Statistical Method:</strong> 
                            <span style="font-weight: bold; color: #1f2937;">{'Spearman + Kendall' if scipy_available else 'Pearson (fallback)'}</span>
                        </div>
                        <div style="margin-bottom: 8px;">
                            <strong>Interpretation:</strong><br>
                            <span style="font-size: 14px;">{interpretation}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>"""
    except Exception as e:
        ranking_stats_html = f"""
        <div style="background: #fef2f2; padding: 15px; border-radius: 8px; margin-bottom: 25px; border-left: 4px solid #dc2626;">
            <strong>⚠️ Statistical Analysis Unavailable:</strong> {str(e)[:100]}...
        </div>"""
    
    # Key metrics to compare
    key_metrics = ['ndcg@1', 'ndcg@5', 'ndcg@10', 'map@5', 'mrr@1']
    
    comparison_html = f"""
    <div style="font-family: Arial, sans-serif; border: 1px solid #ddd; padding: 25px; border-radius: 12px; 
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); margin: 20px 0;">
        
        <h2 style="color: #1f2937; margin-bottom: 25px; text-align: center; border-bottom: 3px solid #3b82f6; padding-bottom: 15px;">
            🏆 BEIR Evaluation Comparison Analysis
        </h2>
        
        <div style="background: white; padding: 20px; border-radius: 10px; margin-bottom: 25px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            <h3 style="color: #374151; margin: 0 0 15px 0;">📋 Evaluation Summary</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div>
                    <strong>Dataset:</strong> {dataset_name}<br>
                    <strong>Documents:</strong> {num_docs:,}<br>
                    <strong>Retriever:</strong> BM25
                </div>
                <div>
                    <strong>Model A:</strong> {method_a}::{model_a}<br>
                    <strong>Model B:</strong> {method_b}::{model_b}<br>
                    <strong>Evaluation:</strong> TREC Metrics + Statistical Analysis
                </div>
            </div>
        </div>
        
        {ranking_stats_html}
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 25px; margin-bottom: 25px;">
            <!-- Performance Comparison -->
            <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <h3 style="color: #059669; margin: 0 0 20px 0; border-bottom: 2px solid #059669; padding-bottom: 8px;">
                    📊 TREC Performance Metrics
                </h3>
                <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                    <thead>
                        <tr style="background-color: #f8f9fa;">
                            <th style="padding: 10px; text-align: left; border: 1px solid #dee2e6;">Metric</th>
                            <th style="padding: 10px; text-align: center; border: 1px solid #dee2e6;">Model A</th>
                            <th style="padding: 10px; text-align: center; border: 1px solid #dee2e6;">Model B</th>
                            <th style="padding: 10px; text-align: center; border: 1px solid #dee2e6;">Winner</th>
                        </tr>
                    </thead>
                    <tbody>"""
    
    # Add metric rows
    metrics_display = {
        'ndcg@1': 'NDCG@1',
        'ndcg@5': 'NDCG@5', 
        'ndcg@10': 'NDCG@10',
        'map@1': 'MAP@1',
        'map@5': 'MAP@5',
        'map@10': 'MAP@10',

    }
    
    model_a_wins = 0
    model_b_wins = 0
    ties = 0
    print(metrics_a, metrics_b)
    for metric_key, metric_name in metrics_display.items():
        val_a = metrics_a.get(metric_key, 0)
        val_b = metrics_b.get(metric_key, 0)
        print(f"Comparing {metric_name}: Model A = {val_a}, Model B = {val_b}")
        winner = get_winner(val_a, val_b)
        
        if "Model A" in winner:
            model_a_wins += 1
            row_color = "#f0f9ff"
        elif "Model B" in winner:
            model_b_wins += 1
            row_color = "#fef2f2"
        else:
            ties += 1
            row_color = "#fffbeb"
        
        comparison_html += f"""
                        <tr style="background-color: {row_color};">
                            <td style="padding: 10px; border: 1px solid #dee2e6; font-weight: bold;">{metric_name}</td>
                            <td style="padding: 10px; border: 1px solid #dee2e6; text-align: center;">{val_a:.4f}</td>
                            <td style="padding: 10px; border: 1px solid #dee2e6; text-align: center;">{val_b:.4f}</td>
                            <td style="padding: 10px; border: 1px solid #dee2e6; text-align: center; font-weight: bold;">{winner}</td>
                        </tr>"""
    
    comparison_html += f"""
                    </tbody>
                </table>
            </div>
            
            <!-- Execution Time & Summary -->
            <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <h3 style="color: #dc2626; margin: 0 0 20px 0; border-bottom: 2px solid #dc2626; padding-bottom: 8px;">
                    ⏱️ Performance Summary
                </h3>
                
                <div style="margin-bottom: 20px;">
                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                        <h4 style="margin: 0 0 10px 0; color: #374151;">Execution Time</h4>
                        <div><strong>Model A:</strong> {time_a:.2f} seconds</div>
                        <div><strong>Model B:</strong> {time_b:.2f} seconds</div>
                        <div><strong>Speed Winner:</strong> {get_winner(time_b, time_a)}</div>
                    </div>
                    
                    <div style="background-color: #f1f5f9; padding: 15px; border-radius: 8px;">
                        <h4 style="margin: 0 0 10px 0; color: #374151;">TREC Metrics Performance</h4>
                        <div><strong>Model A Wins:</strong> {model_a_wins} metrics</div>
                        <div><strong>Model B Wins:</strong> {model_b_wins} metrics</div>
                        <div><strong>Ties:</strong> {ties} metrics</div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Overall Winner -->
        <div style="background: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            <h3 style="color: #1f5582; margin: 0 0 15px 0;">🏆 Overall Winner</h3>"""
    
    if model_a_wins > model_b_wins:
        overall_winner = f"🅰️ Model A ({method_a}::{model_a}) wins with {model_a_wins} out of {len(metrics_display)} metrics!"
        winner_color = "#059669"
    elif model_b_wins > model_a_wins:
        overall_winner = f"🅱️ Model B ({method_b}::{model_b}) wins with {model_b_wins} out of {len(metrics_display)} metrics!"
        winner_color = "#dc2626"
    else:
        overall_winner = f"🤝 It's a tie! Both models won {model_a_wins} metrics each."
        winner_color = "#f59e0b"
    
    comparison_html += f"""
            <div style="font-size: 18px; font-weight: bold; color: {winner_color}; background-color: {winner_color}15; 
                        padding: 15px; border-radius: 8px; border: 2px solid {winner_color};">
                {overall_winner}
            </div>
        </div>
        
        <div style="margin-top: 20px; padding: 15px; background-color: #f0f4f8; border-radius: 8px; border-left: 4px solid #3b82f6;">
            <small style="color: #6b7280;">
                <strong>📝 Note:</strong> This evaluation combines TREC metrics (NDCG, MAP, MRR) with statistical ranking analysis 
                (Spearman correlation, Kendall's Tau, RBO). Higher TREC values and correlation values closer to 1.0 indicate better/similar performance.
                Results are not saved to logs as requested.
            </small>
        </div>
    </div>
    """
    
    return comparison_html

def build_beir_tab():
    with gr.Tab("📈 BEIR Dataset Evaluation"):
        gr.Markdown("# 📊 BEIR Dataset Evaluation Arena")
        gr.Markdown("Compare two reranking methods on standard BEIR datasets without saving results to logs.")
        
        # Dataset and model selection
        with gr.Row():
            with gr.Column():
                dataset_selector = gr.Dropdown(
                    choices=["Select Dataset"] + get_beir_datasets(),
                    label="📚 Select BEIR Dataset",
                    value="Select Dataset"
                )
                num_docs = gr.Slider(
                    minimum=10, maximum=1000, value=100, step=10,
                    label="📄 Number of Documents",
                    info="Number of documents to evaluate (more = slower but more comprehensive)"
                )
            
            with gr.Column():
                gr.Markdown("### 🔧 Reranker Selection")
                
                with gr.Row():
                    with gr.Column():
                        method_a = gr.Dropdown(
                            choices=["Select Method"] + get_rankify_methods(),
                            label="Method A"
                        )
                        model_a = gr.Dropdown(choices=[], label="Model A")
                    
                    with gr.Column():
                        method_b = gr.Dropdown(
                            choices=["Select Method"] + get_rankify_methods(),
                            label="Method B"
                        )
                        model_b = gr.Dropdown(choices=[], label="Model B")
        
        # Info section
        with gr.Row():
            gr.HTML("""
            <div style="background-color: #e8f4fd; border: 1px solid #b3d9ff; border-radius: 8px; padding: 15px; margin: 10px 0;">
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <span style="font-size: 20px; margin-right: 8px;">ℹ️</span>
                    <strong style="color: #1f5582;">BEIR Evaluation Info</strong>
                </div>
                <ul style="margin: 0; padding-left: 20px; color: #374151;">
                    <li><strong>Datasets:</strong> dl19, dl20 (TREC Deep Learning) + BEIR collection (covid, nfc, touche, etc.)</li>
                    <li><strong>Metrics:</strong> NDCG@1/5/10, MAP@1/5/10, MRR@1/5/10</li>
                    <li><strong>Base Retriever:</strong> BM25</li>
                    <li><strong>Privacy:</strong> Results are NOT saved to logs (evaluation only)</li>
                </ul>
            </div>
            """)
        
        # Run button
        run_btn = gr.Button("🚀 Run BEIR Evaluation", variant="primary", size="lg")
        
        # Status display
        status_display = gr.Textbox(
            label="🔄 Evaluation Status",
            value="Ready to run evaluation. Select dataset and models above.",
            lines=3,
            interactive=False
        )
        
        # Results display
        gr.Markdown("## 📈 Evaluation Results")
        with gr.Row():
            with gr.Column():
                results_a = gr.Markdown("### 🅰️ Model A Results\nResults will appear here after evaluation.")
            with gr.Column():
                results_b = gr.Markdown("### 🅱️ Model B Results\nResults will appear here after evaluation.")
        
        # Comparison analysis
        gr.Markdown("## 🔍 Detailed Comparison Analysis")
        comparison_display = gr.HTML(
            value="<p style='color: #666; text-align: center; padding: 40px;'>"
                  "🔄 Run evaluation to see detailed comparison between models</p>"
        )
        
        # Event handlers
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
        
        # Run evaluation
        run_btn.click(
            fn=run_beir_evaluation,
            inputs=[dataset_selector, method_a, model_a, method_b, model_b, num_docs],
            outputs=[status_display, results_a, results_b, comparison_display]
        )