#direct tab
import gradio as gr
import json
from utils import get_rankify_methods, get_models_for_method, run_rankify_reranker
from logging_utils import (
    save_reranking_interaction,
    get_or_create_user_session,
    get_user_stats
)

def parse_inputs(doc_mode, doc_file, doc_text1, doc_text2, *extra_docs):
    if doc_mode == "Upload JSON":
        if doc_file is None:
            return "", [], "⚠️ No file uploaded."
        try:
            with open(doc_file.name, "r", encoding="utf-8") as f:
                data = json.load(f)
                query = data.get("query", "")
                docs = data.get("documents", [])
                return query, docs, None
        except Exception as e:
            return "", [], f"❌ Failed to parse JSON: {e}"
    else:
        docs = [doc_text1, doc_text2] + list(extra_docs)
        docs = [d for d in docs if d.strip()]
        return "", docs, None

def run_direct_rerank(query, method, model, doc_mode, doc_file, doc_text1, doc_text2, user_id, *extra_docs):
    # Get or create user session
    if not user_id:
        user_id = get_or_create_user_session()
    
    # Show loading state
    yield [(None, "🔄 Reranking in progress...")], user_id, "", gr.update(value=f"👤 **Session: {user_id}**")

    query_from_json, docs, error = parse_inputs(doc_mode, doc_file, doc_text1, doc_text2, *extra_docs)
    if error:
        yield [(None, error)], user_id, "", gr.update(value=f"👤 **Session: {user_id}**")
        return

    query = query_from_json if query_from_json else query
    if not docs:
        yield [(None, "⚠️ Please provide documents to rerank.")], user_id, "", gr.update(value=f"👤 **Session: {user_id}**")
        return

    try:
        # Run reranker
        reranked = run_rankify_reranker(query, docs, method, model)

        # Convert doc_texts to Context-like objects for logging (same as reranker_tab.py)
        original_contexts = []
        for i, doc_text in enumerate(docs):
            class SimpleContext:
                def __init__(self, text, idx):
                    self.id = f"doc_{idx}"
                    self.title = f"Document {idx+1}"
                    self.text = text
                    self.score = 0.0
            original_contexts.append(SimpleContext(doc_text, i))

        # Convert reranked results to Context-like objects
        reranked_contexts = []
        for i, doc_text in enumerate(reranked):
            class SimpleContext:
                def __init__(self, text, idx):
                    orig_idx = next((j for j, orig in enumerate(docs) if orig == text), idx)
                    self.id = f"doc_{orig_idx}"
                    self.title = f"Document {orig_idx+1}"
                    self.text = text
                    self.score = 1.0 - (i * 0.1)  # Simple decreasing score
            reranked_contexts.append(SimpleContext(doc_text, i))

        # Save reranking interaction (matching reranker_tab.py format)
        reranking_id = save_reranking_interaction(
            user_id,
            query,
            method, model,  # method_a, model_a
            None, None,     # method_b, model_b (None for direct reranker)
            reranked_contexts,  # reranked_contexts_a
            [],                 # reranked_contexts_b (empty for direct)
            original_contexts
        )

        # Format output for display (same as reranker_tab.py)
        combined_info = []
        doc_counter = 1
        for doc in reranked:
            text = doc.strip()
            print(f"After cleaning (direct rerank) - Doc {doc_counter}, Text: {text}")  # Debug
            
            info = (
                f"**Document {doc_counter}**\n"
                f"**Text:** {text}\n"
                f"**Score:** {1.0 - (doc_counter - 1) * 0.1}"
            )
            combined_info.append(info)
            doc_counter += 1

        response = f"**Query:** {query}\n\n" + "\n\n".join(combined_info) if combined_info else f"**Query:** {query}\n\nNo documents reranked."
        chat = [(None, response)]
        
        # Add success message with reranking ID
        success_message = f"**{method}::{model}** - ✅ Reranking complete (ID: {reranking_id[:8]})"
        chat.append((None, success_message))

        # Get updated user stats for display
        stats = get_user_stats(user_id)
        user_info_update = gr.update(value=f"👤 **Session: {user_id}** | Reranks: {stats['reranking_count']} | Votes: {stats['vote_count']}")

        yield chat, user_id, reranking_id, user_info_update

    except Exception as e:
        error_msg = f"❌ Error during reranking: {str(e)}"
        yield [(None, error_msg)], user_id or "anonymous", "", gr.update(value=f"👤 **Session: {user_id or 'anonymous'}**")

def toggle_inputs(mode, current_count, extra_textboxes):
    if mode == "Upload JSON":
        file_vis = gr.update(visible=True)
        hint_vis = gr.update(visible=True)  # Show JSON hint
        tb_updates = [gr.update(visible=False) for _ in range(2)]
        tb_updates += [gr.update(visible=False) for _ in extra_textboxes]
        add_btn = gr.update(visible=False)
        return [file_vis, hint_vis] + tb_updates + [add_btn, 0]
    else:
        file_vis = gr.update(visible=False)
        hint_vis = gr.update(visible=False)  # ✅ ADD THIS LINE - Hide JSON hint
        tb_updates = [gr.update(visible=True), gr.update(visible=True)]  # Always show doc_text_1 and doc_text_2
        tb_updates += [gr.update(visible=(i < current_count)) for i in range(len(extra_textboxes))]
        add_btn = gr.update(visible=current_count < len(extra_textboxes))
        return [file_vis, hint_vis] + tb_updates + [add_btn, current_count]  # ✅ ADD hint_vis here too

def add_more_docs(current_count, extra_textboxes):
    new_count = min(current_count + 1, len(extra_textboxes))
    tb_updates = [gr.update(visible=(i < new_count)) for i in range(len(extra_textboxes))]
    add_btn = gr.update(visible=new_count < len(extra_textboxes))
    return tb_updates + [add_btn, new_count]

def update_user_info(user_id):
    """Update user information display"""
    if user_id:
        stats = get_user_stats(user_id)
        return gr.update(value=f"👤 **Session: {user_id}** | Reranks: {stats['reranking_count']} | Votes: {stats['vote_count']}")
    return gr.update(value="👤 **Your Session ID will be generated when you start**")

def build_direct_reranker_tab():
    with gr.Tab("🎯 Direct Reranker"):
        gr.Markdown("### 🎯 Direct Reranker (Single Method + Model)")
        
        # User info display (same as reranker_tab.py)
        with gr.Row():
            user_info = gr.Markdown("👤 **Your Session ID will be generated when you start**")

        # Method and model selection
        methods = get_rankify_methods()
        method_select = gr.Dropdown(choices=["Select Methods"] + methods, label="Select Method")
        model_select = gr.Dropdown(choices=[], label="Select Model")

        method_select.change(lambda m: gr.update(choices=get_models_for_method(m)), method_select, model_select)

        # Query input
        query = gr.Textbox(label="Query", placeholder="Enter your query...")

        # Document input mode
        doc_mode = gr.Radio(
            choices=["Upload JSON", "Write Documents"], 
            value="Upload JSON", 
            label="Document Input Mode"
        )
        
        # Document inputs
        with gr.Row():
            with gr.Column(scale=2):
                doc_file = gr.File(label="Upload JSON file", visible=True)
            with gr.Column(scale=1):
                # JSON format hint - positioned beside the file upload
                json_hint = gr.HTML("""
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

        doc_text1 = gr.Textbox(label="Document Text 1", visible=False, lines=3)
        doc_text2 = gr.Textbox(label="Document Text 2", visible=False, lines=3)
        extra_textboxes = [gr.Textbox(label=f"Document Text {i+3}", visible=False, lines=3) for i in range(5)]
        add_btn = gr.Button("Add More Documents", visible=False)
        extra_count = gr.State(0)

        # State management
        user_id_state = gr.State(None)
        reranking_id_state = gr.State("")

        # Submit button
        submit = gr.Button("Submit", variant="primary")
        
        # Output
        chat_output = gr.Chatbot(label="Reranker Output", height=400)

        # Event handlers
        doc_mode.change(
            lambda mode, count: toggle_inputs(mode, count, extra_textboxes),
            inputs=[doc_mode, extra_count],
            outputs=[doc_file, json_hint, doc_text1, doc_text2] + extra_textboxes + [add_btn, extra_count]
        )

        add_btn.click(
            lambda count: add_more_docs(count, extra_textboxes),
            inputs=[extra_count],
            outputs=extra_textboxes + [add_btn, extra_count]
        )

        # Submit with full logging
        submit.click(
            fn=run_direct_rerank,
            inputs=[query, method_select, model_select, doc_mode, doc_file, doc_text1, doc_text2, user_id_state] + extra_textboxes,
            outputs=[chat_output, user_id_state, reranking_id_state, user_info]
        )
        
        # Update user info when user_id changes
        user_id_state.change(
            update_user_info,
            [user_id_state],
            [user_info]
        )