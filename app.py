#app.py
import gradio as gr
from reranker_tab import build_reranker_tab 
from retriever_tab import build_retriever_tab
from anonymous_tab import build_anonymous_reranker_tab
from direct_tab import build_direct_reranker_tab  # ✅ NEW
from leaderboard_tab import build_leaderboard_tab  # ✅ NEW
from beir_tab import build_beir_tab
from rag import build_rag_tab
from annotation_tab import build_annotation_tab
from information_tab import build_information_tab  # ✅ NEW

with gr.Blocks() as demo:
    gr.Markdown("# 🔍 RerankArena")
    gr.HTML("""
        <style>
            .chatbot .user-message {
                text-align: left;
                background-color: #f0f0f0;
                padding: 10px;
                border-radius: 10px;
                margin: 5px 0;
            }
            .chatbot .assistant-message {
                text-align: right;
                background-color: #e6f0fa;
                padding: 10px;
                border-radius: 10px;
                margin: 5px 0;
            }
        </style>
        """)

    build_rag_tab()
    # Shared extra textboxes for reranker tab
    #extra_textboxes = [gr.Textbox(label=f"Document Text {i + 3}", visible=False) for i in range(5)]
    build_direct_reranker_tab()  # ✅ NEW

    build_reranker_tab()  # Pass the defined extra_textboxes instead of [] extra_textboxes
    build_retriever_tab()
    build_anonymous_reranker_tab()  # ✅ Add this
    build_beir_tab()
    
    build_annotation_tab()

    # Build leaderboard tab with auto-refresh functionality
    try:
        leaderboard_outputs = build_leaderboard_tab()
        if leaderboard_outputs is not None and len(leaderboard_outputs) == 17:  # Updated count: 4 basic + 8 plots + 1 refresh function
            *components, refresh_function = leaderboard_outputs
            # Auto-refresh leaderboard when the demo loads
            demo.load(
                fn=refresh_function,
                outputs=components
            )
            print("✅ Leaderboard tab setup successful with auto-refresh")
        else:
            print(f"⚠️ Leaderboard tab returned {len(leaderboard_outputs) if leaderboard_outputs else 0} components, expected 13")
    except Exception as e:
        print(f"❌ Error setting up leaderboard tab: {e}")
        # Fallback: just build the tab without auto-refresh
        build_leaderboard_tab()
        
    build_information_tab()
# demo.launch()
demo.launch(server_name="0.0.0.0", server_port=7860, share=False)