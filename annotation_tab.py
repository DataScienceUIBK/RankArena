# FIXED ANNOTATION TAB - PROPER STRUCTURE AND LAYOUT
# Replace your annotation_tab.py with this properly structured version

import gradio as gr
import json
import datetime
import hashlib
from pathlib import Path
from utils import get_rankify_methods, get_models_for_method
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from logging_utils import (
    get_or_create_user_session,
    get_user_stats,
    ensure_user_folders
)
from rankify.dataset.dataset import Document, Question, Answer, Context
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")

# Global storage for current annotation session
current_annotation_session = {}

def save_annotation_interaction(user_id, query, retriever_config, original_documents, 
                               annotated_documents, annotation_metadata):
    """Save annotation interaction to user-specific folder"""
    
    # Ensure annotation folder exists
    folders = ensure_user_folders(user_id)
    annotation_folder = folders.get('annotate')
    if not annotation_folder:
        base_path = Path("user_data") / user_id
        annotation_folder = base_path / "annotate"
        annotation_folder.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now()
    annotation_id = f"annot_{hashlib.md5(f'{user_id}_{timestamp.isoformat()}_{query}'.encode()).hexdigest()[:12]}"
    
    def calculate_movement_metrics(original_docs, annotated_docs):
        movements = []
        original_positions = {doc.get('id', f'doc_{i}'): i for i, doc in enumerate(original_docs)}
        
        for new_pos, doc in enumerate(annotated_docs):
            doc_id = doc.get('id', f'doc_{new_pos}')
            original_pos = original_positions.get(doc_id, new_pos)
            movement = original_pos - new_pos
            
            movements.append({
                "document_id": doc_id,
                "original_position": original_pos + 1,
                "new_position": new_pos + 1,
                "movement": movement,
                "moved_direction": "up" if movement > 0 else "down" if movement < 0 else "same"
            })
        
        return {
            "document_movements": movements,
            "total_documents": len(annotated_docs),
            "documents_moved": len([m for m in movements if m["movement"] != 0]),
            "average_movement": sum([abs(m["movement"]) for m in movements]) / len(movements) if movements else 0,
            "largest_movement": max([abs(m["movement"]) for m in movements]) if movements else 0,
            "documents_moved_up": len([m for m in movements if m["movement"] > 0]),
            "documents_moved_down": len([m for m in movements if m["movement"] < 0])
        }
    
    movement_metrics = calculate_movement_metrics(original_documents, annotated_documents)
    
    annotation_data = {
        "annotation_id": annotation_id,
        "user_id": user_id,
        "timestamp": timestamp.isoformat(),
        "query": query,
        "retriever_config": retriever_config,
        "original_documents": [
            {
                "id": doc.get('id', f"orig_{i}"),
                "title": doc.get('title', f"Document {i+1}"),
                "text": doc.get('text', str(doc)),
                "score": doc.get('score', 0.0),
                "original_rank": i + 1
            }
            for i, doc in enumerate(original_documents)
        ],
        "annotated_documents": [
            {
                "id": doc.get('id', f"annot_{i}"),
                "title": doc.get('title', f"Document {i+1}"),
                "text": doc.get('text', str(doc)),
                "original_score": doc.get('score', 0.0),
                "user_relevance_rank": i + 1,
                "relevance_label": "highly_relevant" if i < 2 else "relevant" if i < 5 else "less_relevant"
            }
            for i, doc in enumerate(annotated_documents)
        ],
        "movement_metrics": movement_metrics,
        "annotation_metadata": {
            **annotation_metadata,
            "annotation_time_seconds": annotation_metadata.get('annotation_time', 0),
            "total_documents_annotated": len(annotated_documents),
            "user_feedback": annotation_metadata.get('user_feedback', ''),
            "annotation_quality": annotation_metadata.get('quality_rating', 'unknown')
        }
    }
    
    filename = f"annotate_{annotation_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    file_path = annotation_folder / filename
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(annotation_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Annotation saved for user {user_id}: {file_path}")
        print(f"📊 Movement summary: {movement_metrics['documents_moved']}/{movement_metrics['total_documents']} docs moved")
        
    except Exception as e:
        print(f"❌ Error saving annotation: {str(e)}")
    
    return annotation_id

def retrieve_documents_for_annotation(query, retriever_type, offline_method, corpus, n_docs):
    """Retrieve documents for annotation"""
    from rankify.retrievers.retriever import Retriever
    
    user_id = get_or_create_user_session()
    
    if retriever_type == "Online Retriever":
        try:
            retriever = Retriever(method="online", n_docs=n_docs, api_key=SERPER_API_KEY)
            documents = [Document(question=Question(query), answers=Answer([]), contexts=[])]
            retrieved_documents = retriever.retrieve(documents)
            
            if not retrieved_documents or not retrieved_documents[0].contexts:
                return "❌ No online documents retrieved.", None, user_id, None, []
            
            # Extract contexts and format for annotation UI
            contexts = retrieved_documents[0].contexts
            retriever_config = {
                "type": retriever_type,
                "method": "online_retriever",
                "corpus": "web",
                "n_docs": n_docs
            }
            
            # Format documents for the annotation UI
            documents_list = []
            for i, ctx in enumerate(contexts):
                doc_item = {
                    "id": getattr(ctx, 'id', f"web_doc_{i}"),
                    "title": getattr(ctx, 'title', f"Web Document {i+1}"),
                    "text": getattr(ctx, 'text', str(ctx)),  # FULL TEXT - not truncated
                    "score": getattr(ctx, 'score', 0.0),
                    "original_rank": i + 1
                }
                documents_list.append(doc_item)
            
            # Store session data
            current_annotation_session[user_id] = {
                "query": query,
                "retriever_config": retriever_config,
                "original_documents": documents_list,
                "start_time": datetime.datetime.now()
            }
            
            return (
                f"✅ Retrieved {len(documents_list)} online documents. Rank them below!",
                documents_list, user_id, retriever_config, documents_list
            )
            
        except Exception as e:
            return f"❌ Error during online retrieval: {str(e)}", None, user_id, None, []
        
    if not query.strip():
        return "❌ Please enter a query.", None, user_id, None, []
        
    if offline_method is None or corpus is None:
        return "❌ Please select both method and corpus.", None, user_id, None, []
    
    try:
        retriever = Retriever(method=offline_method, n_docs=n_docs, index_type=corpus)
        documents = [Document(question=Question(query), answers=Answer([]), contexts=[])]
        retrieved_documents = retriever.retrieve(documents)
        
        if not retrieved_documents or not retrieved_documents[0].contexts:
            return "❌ No documents retrieved.", None, user_id, None, []
        
        # Extract contexts and format for annotation UI
        contexts = retrieved_documents[0].contexts
        retriever_config = {
            "type": retriever_type,
            "method": offline_method,
            "corpus": corpus,
            "n_docs": n_docs
        }
        
        # Format documents for the annotation UI
        documents_list = []
        for i, ctx in enumerate(contexts):
            doc_item = {
                "id": getattr(ctx, 'id', f"doc_{i}"),
                "title": getattr(ctx, 'title', f"Document {i+1}"),
                "text": getattr(ctx, 'text', str(ctx)),  # FULL TEXT - not truncated
                "score": getattr(ctx, 'score', 0.0),
                "original_rank": i + 1
            }
            documents_list.append(doc_item)
        
        # Store session data
        current_annotation_session[user_id] = {
            "query": query,
            "retriever_config": retriever_config,
            "original_documents": documents_list,
            "start_time": datetime.datetime.now()
        }
        
        return (
            f"✅ Retrieved {len(documents_list)} documents. Rank them below!",
            documents_list, user_id, retriever_config, documents_list
        )
        
    except Exception as e:
        return f"❌ Error during retrieval: {str(e)}", None, user_id, None, []

def apply_ranking(docs_state, *rank_selections):
    """Apply ranking based on user dropdown selections"""
    print(f"DEBUG: docs_state = {len(docs_state) if docs_state else 0} documents")
    print(f"DEBUG: rank_selections = {rank_selections}")
    
    if not docs_state:
        return (
            "<p style='color: #dc2626;'>❌ No documents loaded</p>",
            "❌ No documents to rank",
            []
        )
    
    # Get the number of documents
    n_docs = len(docs_state)
    
    # Get rank selections (only take as many as we have documents)
    ranks = rank_selections[:n_docs]
    
    print(f"DEBUG: Processing {n_docs} documents with ranks: {ranks}")
    
    # Validate all ranks are selected
    if not all(rank and rank.startswith("Rank ") for rank in ranks):
        missing_ranks = [i+1 for i, rank in enumerate(ranks) if not rank or not rank.startswith("Rank ")]
        return (
            f"<p style='color: #dc2626;'>❌ Please select ranks for all documents. Missing: Documents {missing_ranks}</p>",
            f"❌ Missing ranks for documents {missing_ranks}",
            []
        )
    
    # Extract rank numbers
    try:
        rank_numbers = [int(rank.split()[-1]) for rank in ranks]
    except ValueError as e:
        return (
            f"<p style='color: #dc2626;'>❌ Invalid rank format: {e}</p>",
            "❌ Invalid rank format",
            []
        )
    
    # Check for duplicates
    if len(set(rank_numbers)) != len(rank_numbers):
        duplicates = [r for r in set(rank_numbers) if rank_numbers.count(r) > 1]
        return (
            f"<p style='color: #dc2626;'>❌ Duplicate ranks found: {duplicates}. Each document needs a unique rank.</p>",
            f"❌ Duplicate ranks: {duplicates}",
            []
        )
    
    # Check if all ranks are in valid range
    expected_ranks = set(range(1, n_docs + 1))
    provided_ranks = set(rank_numbers)
    if provided_ranks != expected_ranks:
        missing = expected_ranks - provided_ranks
        extra = provided_ranks - expected_ranks
        error_msg = []
        if missing:
            error_msg.append(f"Missing ranks: {sorted(missing)}")
        if extra:
            error_msg.append(f"Invalid ranks: {sorted(extra)}")
        return (
            f"<p style='color: #dc2626;'>❌ {', '.join(error_msg)}</p>",
            f"❌ Invalid rank selection",
            []
        )
    
    # Create ranking pairs and sort by rank
    ranking_pairs = []
    for i, (rank_num, doc) in enumerate(zip(rank_numbers, docs_state)):
        ranking_pairs.append((rank_num, doc, i))
    
    # Sort by rank (ascending - 1 is most relevant)
    ranking_pairs.sort(key=lambda x: x[0])
    reordered_docs = [pair[1] for pair in ranking_pairs]
    
    print(f"DEBUG: Reordered {len(reordered_docs)} documents")
    
    # Create visual display of new ranking with movement indicators
    display_html = f"""
    <div style="background: #f0f9ff; border: 2px solid #3b82f6; border-radius: 12px; padding: 20px; margin: 10px 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        <h3 style="color: #1e40af; margin: 0 0 15px 0;">✅ Documents Ranked Successfully!</h3>
        <div style="background: #dbeafe; border: 1px solid #93c5fd; border-radius: 8px; padding: 15px; margin-bottom: 15px;">
            <strong>📊 Ranking Summary:</strong> {len(reordered_docs)} documents reordered based on your relevance rankings.
        </div>
    """
    
    movements_count = 0
    for i, doc in enumerate(reordered_docs):
        # Find original position
        orig_pos = next((j for j, orig_doc in enumerate(docs_state) if orig_doc['id'] == doc['id']), i)
        
        # Movement indicator
        if i < orig_pos:
            movement = f"🟢 ⬆️ Moved up {orig_pos - i} positions"
            movement_color = "#16a34a"
            movements_count += 1
        elif i > orig_pos:
            movement = f"🔴 ⬇️ Moved down {i - orig_pos} positions"  
            movement_color = "#dc2626"
            movements_count += 1
        else:
            movement = "➖ No change"
            movement_color = "#6b7280"
        
        display_html += f"""
        <div style="background: white; border-radius: 8px; padding: 15px; margin: 8px 0; border-left: 4px solid #3b82f6; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="background: linear-gradient(135deg, #3b82f6, #1d4ed8); color: white; padding: 6px 12px; border-radius: 16px; font-weight: bold;">
                        #{i+1}
                    </span>
                    <strong style="color: #1f2937; font-size: 1.1em;">{doc['title']}</strong>
                </div>
                <span style="color: {movement_color}; font-size: 0.9em; font-weight: 600;">
                    {movement}
                </span>
            </div>
            <div style="color: #6b7280; font-size: 0.8em; font-family: monospace; background: #f9fafb; padding: 4px 8px; border-radius: 4px; display: inline-block; margin-bottom: 8px;">
                Score: {doc['score']:.4f} | ID: {doc['id']} | Original Position: #{orig_pos + 1}
            </div>
        </div>
        """
    
    display_html += f"""
        <div style="background: #dcfce7; border: 1px solid #16a34a; border-radius: 8px; padding: 15px; margin-top: 15px;">
            <strong>🎯 Ready to Save:</strong> You have successfully ranked {len(reordered_docs)} documents with {movements_count} movements. 
            Click "Save Annotation" below to save your rankings.
        </div>
    </div>
    """
    
    success_msg = f"✅ Successfully ranked {len(reordered_docs)} documents! {movements_count} documents moved from original positions."
    
    return display_html, success_msg, reordered_docs

def save_annotation_rankings(user_id, quality_rating, user_feedback, reranked_docs):
    """Save the annotation rankings"""
    global current_annotation_session
    
    print(f"DEBUG: Saving annotation for user {user_id}")
    print(f"DEBUG: Reranked docs count: {len(reranked_docs) if reranked_docs else 0}")
    
    if not user_id or user_id not in current_annotation_session:
        return "❌ No annotation session found. Please retrieve documents first."
    
    if not reranked_docs:
        return "❌ No ranked documents found. Please rank documents first using the dropdowns and click 'Apply Ranking'."
    
    session_data = current_annotation_session[user_id]
    start_time = session_data.get('start_time', datetime.datetime.now())
    annotation_time = (datetime.datetime.now() - start_time).total_seconds()
    
    try:
        original_documents = session_data['original_documents']
        
        # Calculate actual movements
        movements = 0
        for i, reranked_doc in enumerate(reranked_docs):
            orig_pos = next((j for j, orig_doc in enumerate(original_documents) if orig_doc['id'] == reranked_doc['id']), i)
            if i != orig_pos:
                movements += 1
        
        annotation_metadata = {
            "annotation_time": annotation_time,
            "user_feedback": user_feedback or "",
            "quality_rating": quality_rating or "good",
            "annotation_method": "gradio_proper_structure_layout",
            "interface_version": "11.0_fixed_structure_full_content",
            "total_movements": movements,
            "ranking_method": "user_dropdown_selection"
        }
        
        annotation_id = save_annotation_interaction(
            user_id=user_id,
            query=session_data['query'],
            retriever_config=session_data['retriever_config'],
            original_documents=original_documents,
            annotated_documents=reranked_docs,
            annotation_metadata=annotation_metadata
        )
        
        # Get updated stats
        stats = get_user_stats(user_id)
        
        # Clean up session
        del current_annotation_session[user_id]
        
        success_message = f"""✅ Annotation saved successfully! (ID: {annotation_id[:8]})

📊 Your Stats:
• Retrievals: {stats.get('retrieval_count', 0)} 
• Reranks: {stats.get('reranking_count', 0)} 
• Votes: {stats.get('vote_count', 0)}
• RAG Sessions: {stats.get('rag_count', 0)}
• Annotations: {stats.get('annotation_count', 0)}

⏱️ Annotation Time: {annotation_time:.1f} seconds
📝 Quality Rating: {quality_rating}
💬 Feedback: {user_feedback or 'None provided'}
🔄 Documents Moved: {movements} out of {len(reranked_docs)}

📁 Saved to: user_data/{user_id}/annotate/

🎉 Your custom rankings have been captured and saved!"""
        
        return success_message
        
    except Exception as e:
        return f"❌ Error saving annotation: {str(e)}"


def build_annotation_tab():
    """Build annotation tab with proper structure: Document (left) + Rank (right)"""
    
    with gr.Tab("✏️ Document Annotation"):
        gr.Markdown("# 📝 Document Annotation: Manual Relevance Ranking")
        gr.Markdown("**✅ Proper layout - Document content (left) + Rank selector (right)**")
        
        # User info display
        with gr.Row():
            user_info = gr.Markdown("👤 **Your Session ID will be generated when you start**")
        
        # Step 1: Document Retrieval
        gr.Markdown("## 📚 Step 1: Retrieve Documents for Annotation")
        with gr.Row():
            with gr.Column():
                query_box = gr.Textbox(
                    label="Enter your query", 
                    placeholder="What is the impact of climate change on agriculture?"
                )
                retriever_type = gr.Radio(
                    ["Online Retriever", "Offline Retriever"],
                    value="Offline Retriever",
                    label="Retriever Type"
                )
            with gr.Column():
                offline_method = gr.Dropdown(
                    ["bm25", "dpr-multi", "bge", "colbert", "contriever"],
                    label="Retriever Method",
                    visible=True
                )
                corpus = gr.Dropdown(
                    ["wiki", "msmarco"],
                    label="Corpus",
                    visible=True
                )
                n_docs = gr.Slider(5, 12, value=8, step=1, label="Number of Documents to Annotate")
        
        retrieve_btn = gr.Button("🔍 Retrieve Documents for Annotation", variant="primary")
        retrieve_status = gr.Textbox(label="Retrieval Status", interactive=False)
        
        # Hidden state variables
        user_id_state = gr.State(None)
        documents_state = gr.State([])
        reranked_docs_state = gr.State([])
        
        # Step 2: Proper Document Display and Ranking Layout
        ranking_section = gr.Column(visible=False)
        
        with ranking_section:
            gr.Markdown("## 🎯 Step 2: Rank Documents by Relevance")
            gr.HTML("""
            <div style="background: #dbeafe; border: 1px solid #93c5fd; border-radius: 8px; padding: 15px; margin-bottom: 15px;">
                <strong>📋 Instructions:</strong> 
                Review each document below and select its relevance rank (Rank 1 = most relevant, higher numbers = less relevant).
                Each document must have a unique rank.
            </div>
            """)
            
            # Create PROPER layout: Document (LEFT, wider) + Rank dropdown (RIGHT, narrower)
            doc_components = []
            for i in range(12):  # Support up to 12 documents
                with gr.Row(visible=False) as doc_row:
                    with gr.Column(scale=4):  # Document takes 4/5 of width (LEFT)
                        doc_display = gr.HTML(value="")
                    with gr.Column(scale=1, min_width=150):  # Rank dropdown takes 1/5 width (RIGHT)
                        rank_dropdown = gr.Dropdown(
                            choices=[],
                            label=f"Rank",
                            container=True
                        )
                
                doc_components.append((doc_row, rank_dropdown, doc_display))
            
            # Apply ranking button and results
            apply_ranking_btn = gr.Button("🔄 Apply Ranking", variant="primary", size="lg")
            ranking_display = gr.HTML(value="")
            ranking_status = gr.Textbox(label="Ranking Status", interactive=False)
        
        # Step 3: Save Annotation
        gr.Markdown("## 💾 Step 3: Save Your Annotation")
        with gr.Row():
            with gr.Column():
                quality_rating = gr.Radio(
                    choices=["excellent", "good", "fair", "poor"],
                    value="good",
                    label="How would you rate the quality of retrieved documents?"
                )
            with gr.Column():
                user_feedback = gr.Textbox(
                    label="Additional Feedback (Optional)",
                    placeholder="Any comments about the retrieval quality or annotation process...",
                    lines=3
                )
        
        save_annotation_btn = gr.Button("💾 Save Annotation", variant="secondary")
        annotation_result = gr.Textbox(label="Annotation Result", lines=8, interactive=False)
        
        # Event handlers
        def toggle_offline_fields(retriever_type):
            if retriever_type == "Offline Retriever":
                return gr.update(visible=True), gr.update(visible=True)
            else:
                return gr.update(visible=False), gr.update(visible=False)
        
        def update_user_info(user_id):
            if user_id:
                stats = get_user_stats(user_id)
                numeric_stats = {k: v for k, v in stats.items() if isinstance(v, (int, float))}
                total_sessions = sum(numeric_stats.values()) if numeric_stats else 0
                return gr.update(
                    value=f"👤 **Session: {user_id}** | 📊 Total Sessions: {total_sessions}"
                )
            return gr.update(value="👤 **Your Session ID will be generated when you start**")
        
        def handle_document_retrieval(query, retriever_type, offline_method, corpus, n_docs):
            """Handle document retrieval and setup proper structure interface"""
            
            # Get retrieval results
            status, docs, user_id, config, ui_docs = retrieve_documents_for_annotation(
                query, retriever_type, offline_method, corpus, n_docs
            )
            
            if not docs:
                # Hide all document components
                hide_updates = []
                for i in range(12):
                    hide_updates.extend([
                        gr.update(visible=False),  # doc_row
                        gr.update(visible=False),  # rank_dropdown
                        gr.update(value="")        # doc_display
                    ])
                
                return (
                    status, user_id, [],
                    gr.update(visible=False), "", "", 
                    *hide_updates
                )
            
            # Setup ranking dropdowns with compact, clean document display
            rank_choices = [f"Rank {i}" for i in range(1, len(docs) + 1)]
            
            ranking_updates = []
            for i in range(12):
                if i < len(docs):
                    doc = docs[i]
                    
                    # Create clean, compact document display with FULL content
                    doc_display_html = f"""
                    <div style="
                        background: #f8fafc; 
                        border: 1px solid #e2e8f0; 
                        border-radius: 8px; 
                        padding: 16px; 
                        margin: 4px 0; 
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                        height: 160px;
                        overflow-y: auto;
                    ">
                        <!-- Header -->
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; padding-bottom: 6px; border-bottom: 1px solid #e2e8f0;">
                            <span style="background: #3b82f6; color: white; padding: 4px 10px; border-radius: 12px; font-weight: bold; font-size: 0.9em;">
                                Document #{i+1}
                            </span>
                            <span style="background: #f1f5f9; color: #475569; padding: 3px 8px; border-radius: 6px; font-size: 0.8em;">
                                Score: {doc['score']:.4f}
                            </span>
                        </div>
                        
                        <!-- Title -->
                        <div style="font-weight: 600; color: #1f2937; margin-bottom: 8px; font-size: 1.0em; line-height: 1.3;">
                            {doc['title']}
                        </div>
                        
                        <!-- FULL Content -->
                        <div style="
                            color: #374151; 
                            line-height: 1.5; 
                            font-size: 0.85em; 
                            background: white; 
                            border: 1px solid #e5e7eb; 
                            border-radius: 4px; 
                            padding: 10px;
                            max-height: 100px;
                            overflow-y: auto;
                        ">
                            {doc['text']}
                        </div>
                        
                        <!-- Footer -->
                        <div style="font-size: 0.7em; color: #6b7280; margin-top: 6px; text-align: right;">
                            ID: {doc['id']} | Length: {len(doc['text'])} chars | Original Position: #{i+1}
                        </div>
                    </div>
                    """
                    
                    ranking_updates.extend([
                        gr.update(visible=True),  # doc_row
                        gr.update(choices=rank_choices, value=f"Rank {i+1}", label=f"Rank for Doc #{i+1}"),  # rank_dropdown
                        gr.update(value=doc_display_html)  # doc_display
                    ])
                else:
                    ranking_updates.extend([
                        gr.update(visible=False),  # doc_row
                        gr.update(choices=[], value=None, label="Rank"),  # ✅ CLEAR choices for hidden dropdowns
                        gr.update(value="")        # doc_display
                    ])
            
            return (
                status, user_id, docs,
                gr.update(visible=True),  # ranking_section
                "", "",  # ranking_display, ranking_status
                *ranking_updates
            )
        
        # Wire up events
        retriever_type.change(
            toggle_offline_fields,
            [retriever_type],
            [offline_method, corpus]
        )
        
        # Prepare outputs for retrieval
        all_outputs = []
        all_dropdowns = []
        for doc_row, rank_dropdown, doc_display in doc_components:
            all_outputs.extend([doc_row, rank_dropdown, doc_display])
            all_dropdowns.append(rank_dropdown)
        
        retrieve_btn.click(
            handle_document_retrieval,
            [query_box, retriever_type, offline_method, corpus, n_docs],
            [retrieve_status, user_id_state, documents_state, ranking_section, ranking_display, ranking_status] + all_outputs
        )
        
        user_id_state.change(
            update_user_info,
            [user_id_state],
            [user_info]
        )
        
        # Apply ranking - capture all dropdown values
        apply_ranking_btn.click(
            apply_ranking,
            inputs=[documents_state] + all_dropdowns,
            outputs=[ranking_display, ranking_status, reranked_docs_state]
        )
        
        # Save annotation
        save_annotation_btn.click(
            save_annotation_rankings,
            [user_id_state, quality_rating, user_feedback, reranked_docs_state],
            [annotation_result]
        )