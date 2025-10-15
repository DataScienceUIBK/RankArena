# information_tab.py - Comprehensive System Information Tab
import gradio as gr

def build_information_tab():
    """Build a comprehensive information tab describing the RankArena system"""
    
    with gr.Tab("ℹ️ System Information"):
        gr.HTML("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 40px; border-radius: 15px; margin-bottom: 30px; text-align: center;">
            <h1 style="font-size: 3em; margin-bottom: 20px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                🔍 RankArena
            </h1>
            <h2 style="font-size: 1.5em; margin-bottom: 10px; opacity: 0.9;">
                The Ultimate Reranking Evaluation Platform
            </h2>
            <p style="font-size: 1.2em; opacity: 0.8; max-width: 800px; margin: 0 auto;">
                A comprehensive arena for comparing, evaluating, and analyzing document reranking methods 
                across multiple benchmarks with user voting, LLM judges, and statistical analysis.
            </p>
        </div>
        """)
        
        # System Overview
        with gr.Row():
            gr.HTML("""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 25px; margin-bottom: 25px;">
                <h2 style="color: #1e40af; margin-bottom: 20px; display: flex; align-items: center;">
                    <span style="font-size: 1.5em; margin-right: 10px;">🎯</span>
                    System Overview
                </h2>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 25px;">
                    <div>
                        <h3 style="color: #059669; margin-bottom: 15px;">🚀 Core Features</h3>
                        <ul style="color: #374151; line-height: 1.8;">
                            <li><strong>Multi-Modal Evaluation:</strong> User votes, LLM judges, benchmark scores</li>
                            <li><strong>Real-time Competition:</strong> Arena-style head-to-head comparisons</li>
                            <li><strong>Comprehensive Benchmarks:</strong> BEIR, DL19, DL20, and custom datasets</li>
                            <li><strong>Advanced Analytics:</strong> Statistical significance, clustering, correlations</li>
                            <li><strong>RAG Pipeline Testing:</strong> End-to-end retrieval-augmented generation</li>
                        </ul>
                    </div>
                    <div>
                        <h3 style="color: #dc2626; margin-bottom: 15px;">📊 Supported Methods</h3>
                        <ul style="color: #374151; line-height: 1.8;">
                            <li><strong>Retrieval:</strong> BM25, DPR, BGE, ColBERT, Contriever, Online Search</li>
                            <li><strong>Reranking:</strong> Multiple Rerankers methods and models</li>
                            <li><strong>Generation:</strong> Llama 3.3 70B via Together AI</li>
                            <li><strong>Evaluation:</strong> NDCG, MAP, MRR, RBO, ELO ratings</li>
                            <li><strong>Datasets:</strong> Wikipedia, MS MARCO, BEIR collection</li>
                        </ul>
                    </div>
                </div>
            </div>
            """)
        
        # Tab Descriptions
        gr.HTML("""
        <div style="background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 12px; padding: 25px; margin-bottom: 25px;">
            <h2 style="color: #0369a1; margin-bottom: 25px; display: flex; align-items: center;">
                <span style="font-size: 1.5em; margin-right: 10px;">📚</span>
                Tab Descriptions & Use Cases
            </h2>
        </div>
        """)
        
        # Individual Tab Descriptions
        tab_descriptions = [
            {
                "icon": "💬",
                "title": "RAG Arena",
                "subtitle": "Complete RAG Pipeline Evaluation",
                "description": "Test end-to-end Retrieval-Augmented Generation pipelines with dual reranker comparison. Retrieve documents, rerank with two different methods, and generate answers using Llama 70B.",
                "features": [
                    "Online & offline document retrieval",
                    "Side-by-side reranker comparison",
                    "LLM answer generation with source attribution",
                    "Statistical reranking analysis",
                    "JSON-structured outputs for integration"
                ],
                "use_cases": [
                    "Compare reranking methods for QA systems",
                    "Evaluate RAG pipeline performance",
                    "Test different retrieval strategies",
                    "Analyze answer quality and consistency"
                ],
                "color": "#10b981"
            },
            {
                "icon": "🎯",
                "title": "Direct Reranker",
                "subtitle": "Single Method Testing",
                "description": "Test individual reranking methods with your own documents or JSON uploads. Perfect for quick testing and method evaluation without complex setups.",
                "features": [
                    "JSON file upload support",
                    "Manual document input",
                    "Real-time reranking results",
                    "Session tracking and logging",
                    "Expandable document input (up to 7 docs)"
                ],
                "use_cases": [
                    "Quick method testing",
                    "Custom document evaluation",
                    "Method prototyping",
                    "Educational demonstrations"
                ],
                "color": "#3b82f6"
            },
            {
                "icon": "⚔️",
                "title": "1v1 Reranker Arena",
                "subtitle": "Head-to-Head Comparison",
                "description": "Compare two reranking methods side-by-side with user voting and LLM judge evaluation. The core arena experience with comprehensive statistical analysis.",
                "features": [
                    "Dual reranker comparison",
                    "User voting system",
                    "LLM judge evaluation",
                    "Statistical correlation analysis",
                    "Ranking visualization with movement arrows"
                ],
                "use_cases": [
                    "Method performance comparison",
                    "User preference analysis",
                    "Model benchmarking",
                    "Research validation"
                ],
                "color": "#dc2626"
            },
            {
                "icon": "🛠️",
                "title": "Retriever + Reranker",
                "subtitle": "Full Pipeline Testing",
                "description": "Complete retrieval and reranking pipeline with document retrieval from various sources, dual reranking, and comprehensive comparison analysis.",
                "features": [
                    "Multiple retrieval methods (BM25, DPR, BGE, etc.)",
                    "Online web search via Serper API",
                    "Corpus selection (Wiki, MS MARCO)",
                    "Movement tracking with visual indicators",
                    "Detailed statistical comparison"
                ],
                "use_cases": [
                    "End-to-end pipeline evaluation",
                    "Retrieval method comparison",
                    "Real-world search testing",
                    "Academic research"
                ],
                "color": "#7c3aed"
            },
            {
                "icon": "🎭",
                "title": "Anonymous Arena",
                "subtitle": "Blind Evaluation",
                "description": "Unbiased comparison where method identities are hidden until after voting. Eliminates bias and provides pure performance-based evaluation.",
                "features": [
                    "Hidden method identities",
                    "Bias-free evaluation",
                    "Statistical comparison",
                    "Post-voting reveal",
                    "Anonymous ranking analysis"
                ],
                "use_cases": [
                    "Unbiased method evaluation",
                    "Research validation",
                    "Fair competition",
                    "Objective benchmarking"
                ],
                "color": "#f59e0b"
            },
            {
                "icon": "📈",
                "title": "BEIR Evaluation",
                "subtitle": "Standard Benchmark Testing",
                "description": "Evaluate methods on standard BEIR datasets and TREC Deep Learning tasks with comprehensive metrics (NDCG, MAP, MRR) and statistical analysis.",
                "features": [
                    "14+ BEIR datasets (COVID, NFCorpus, etc.)",
                    "TREC DL19/DL20 evaluation",
                    "Standard IR metrics (NDCG@k, MAP@k, MRR)",
                    "Statistical significance testing",
                    "Performance visualization"
                ],
                "use_cases": [
                    "Academic benchmarking",
                    "Method validation",
                    "Publication-ready results",
                    "Standard evaluation protocol"
                ],
                "color": "#0891b2"
            },
            {
                "icon": "✏️",
                "title": "Document Annotation",
                "subtitle": "Manual Relevance Labeling",
                "description": "Create ground truth annotations by manually ranking document relevance. Essential for training data creation and evaluation validation.",
                "features": [
                    "Intuitive drag-and-drop ranking",
                    "Full document content display",
                    "Quality rating system",
                    "Movement tracking",
                    "Annotation export"
                ],
                "use_cases": [
                    "Training data creation",
                    "Evaluation validation",
                    "Quality assessment",
                    "Dataset curation"
                ],
                "color": "#059669"
            },
            {
                "icon": "🏆",
                "title": "Arena Leaderboard",
                "subtitle": "Comprehensive Rankings",
                "description": "View comprehensive rankings combining user votes, LLM evaluations, and benchmark scores with advanced analytics and visualizations.",
                "features": [
                    "ELO rating system",
                    "Multi-dimensional performance analysis",
                    "Statistical significance testing",
                    "Model clustering analysis",
                    "Interactive visualizations"
                ],
                "use_cases": [
                    "Method comparison",
                    "Performance tracking",
                    "Research insights",
                    "Publication figures"
                ],
                "color": "#dc2626"
            }
        ]
        
        # Create tab description cards
        for i, tab in enumerate(tab_descriptions):
            with gr.Row():
                gr.HTML(f"""
                <div style="background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 25px; margin-bottom: 20px; 
                            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); border-left: 4px solid {tab['color']};">
                    <div style="display: flex; align-items: center; margin-bottom: 15px;">
                        <span style="font-size: 2em; margin-right: 15px;">{tab['icon']}</span>
                        <div>
                            <h3 style="color: {tab['color']}; margin: 0; font-size: 1.4em;">{tab['title']}</h3>
                            <p style="color: #6b7280; margin: 5px 0 0 0; font-weight: 500;">{tab['subtitle']}</p>
                        </div>
                    </div>
                    
                    <p style="color: #374151; line-height: 1.6; margin-bottom: 20px;">
                        {tab['description']}
                    </p>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 25px;">
                        <div>
                            <h4 style="color: {tab['color']}; margin-bottom: 10px; font-size: 1.1em;">🔧 Key Features</h4>
                            <ul style="color: #374151; margin: 0; padding-left: 20px; line-height: 1.6;">
                                {"".join([f"<li>{feature}</li>" for feature in tab['features']])}
                            </ul>
                        </div>
                        <div>
                            <h4 style="color: {tab['color']}; margin-bottom: 10px; font-size: 1.1em;">💡 Use Cases</h4>
                            <ul style="color: #374151; margin: 0; padding-left: 20px; line-height: 1.6;">
                                {"".join([f"<li>{use_case}</li>" for use_case in tab['use_cases']])}
                            </ul>
                        </div>
                    </div>
                </div>
                """)
        
        # Technical Architecture
        gr.HTML("""
        <div style="background: #fef3e2; border: 1px solid #fbbf24; border-radius: 12px; padding: 25px; margin: 25px 0;">
            <h2 style="color: #92400e; margin-bottom: 25px; display: flex; align-items: center;">
                <span style="font-size: 1.5em; margin-right: 10px;">🏗️</span>
                Technical Architecture
            </h2>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 25px;">
                <div>
                    <h3 style="color: #059669; margin-bottom: 15px;">📚 Data Layer</h3>
                    <ul style="color: #374151; line-height: 1.8;">
                        <li><strong>User Data:</strong> Session tracking, vote history</li>
                        <li><strong>Interactions:</strong> JSON logging for all activities</li>
                        <li><strong>Benchmarks:</strong> BEIR datasets, custom evaluations</li>
                        <li><strong>Annotations:</strong> Manual relevance judgments</li>
                    </ul>
                </div>
                
                <div>
                    <h3 style="color: #dc2626; margin-bottom: 15px;">🔧 Processing Layer</h3>
                    <ul style="color: #374151; line-height: 1.8;">
                        <li><strong>Rankify:</strong> Multiple reranking methods</li>
                        <li><strong>Retrievers:</strong> BM25, neural retrievers</li>
                        <li><strong>LLM Judge:</strong> Automated evaluation</li>
                        <li><strong>Analytics:</strong> Statistical analysis engine</li>
                    </ul>
                </div>
                
                <div>
                    <h3 style="color: #7c3aed; margin-bottom: 15px;">🎨 Presentation Layer</h3>
                    <ul style="color: #374151; line-height: 1.8;">
                        <li><strong>Gradio UI:</strong> Interactive web interface</li>
                        <li><strong>Plotly Charts:</strong> Advanced visualizations</li>
                        <li><strong>Real-time Updates:</strong> Progressive feedback</li>
                        <li><strong>Export Options:</strong> CSV, JSON downloads</li>
                    </ul>
                </div>
            </div>
        </div>
        """)
        
        # Getting Started Guide
        gr.HTML("""
        <div style="background: #ecfdf5; border: 1px solid #10b981; border-radius: 12px; padding: 25px; margin: 25px 0;">
            <h2 style="color: #047857; margin-bottom: 25px; display: flex; align-items: center;">
                <span style="font-size: 1.5em; margin-right: 10px;">🚀</span>
                Getting Started Guide
            </h2>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 25px;">
                <div>
                    <h3 style="color: #059669; margin-bottom: 15px;">🎯 For Beginners</h3>
                    <ol style="color: #374151; line-height: 1.8; padding-left: 20px;">
                        <li>Start with <strong>Direct Reranker</strong> for simple testing</li>
                        <li>Try <strong>1v1 Arena</strong> to compare two methods</li>
                        <li>Use <strong>Anonymous Arena</strong> for unbiased evaluation</li>
                        <li>Check <strong>Leaderboard</strong> to see overall rankings</li>
                    </ol>
                </div>
                
                <div>
                    <h3 style="color: #dc2626; margin-bottom: 15px;">🔬 For Researchers</h3>
                    <ol style="color: #374151; line-height: 1.8; padding-left: 20px;">
                        <li>Use <strong>BEIR Evaluation</strong> for standard benchmarks</li>
                        <li>Try <strong>RAG Arena</strong> for end-to-end testing</li>
                        <li>Create annotations with <strong>Document Annotation</strong></li>
                        <li>Analyze results in <strong>Advanced Analytics</strong></li>
                    </ol>
                </div>
            </div>
            
            <div style="background: white; border-radius: 8px; padding: 20px; margin-top: 20px;">
                <h4 style="color: #047857; margin-bottom: 15px;">💡 Pro Tips</h4>
                <ul style="color: #374151; line-height: 1.8; columns: 2; column-gap: 30px;">
                    <li>Each session gets a unique ID for tracking</li>
                    <li>All interactions are logged for analysis</li>
                    <li>LLM judge runs automatically after user votes</li>
                    <li>Statistical analysis requires multiple comparisons</li>
                    <li>Benchmark results are not saved to logs</li>
                    <li>Download results for offline analysis</li>
                </ul>
            </div>
        </div>
        """)
        
        # API and Integration
        gr.HTML("""
        <div style="background: #f1f5f9; border: 1px solid #64748b; border-radius: 12px; padding: 25px; margin: 25px 0;">
            <h2 style="color: #334155; margin-bottom: 25px; display: flex; align-items: center;">
                <span style="font-size: 1.5em; margin-right: 10px;">🔌</span>
                APIs & Integration
            </h2>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 25px;">
                <div>
                    <h3 style="color: #059669; margin-bottom: 15px;">🌐 External APIs</h3>
                    <ul style="color: #374151; line-height: 1.8;">
                        <li><strong>Serper API:</strong> Real-time web search</li>
                        <li><strong>Together AI:</strong> Llama 70B generation</li>
                        <li><strong>Rankify Methods:</strong> Multiple reranking models</li>
                        <li><strong>BEIR Datasets:</strong> Standard benchmarks</li>
                    </ul>
                </div>
                
                <div>
                    <h3 style="color: #dc2626; margin-bottom: 15px;">📊 Data Formats</h3>
                    <ul style="color: #374151; line-height: 1.8;">
                        <li><strong>Input:</strong> JSON, CSV, plain text</li>
                        <li><strong>Output:</strong> JSON logs, CSV exports</li>
                        <li><strong>Metrics:</strong> NDCG, MAP, MRR, ELO</li>
                        <li><strong>Visualizations:</strong> Plotly interactive charts</li>
                    </ul>
                </div>
            </div>
        </div>
        """)
        
        # Footer
        gr.HTML("""
        <div style="background: linear-gradient(135deg, #6a71d7 0%, #374151 100%); 
                    color: white; padding: 30px; border-radius: 12px; margin-top: 30px; text-align: center;">
            <h3 style="margin-bottom: 15px;">🌟 RankArena - Powered by Advanced ML</h3>
            <p style="opacity: 0.8; margin-bottom: 15px;">
                The most comprehensive platform for evaluating and comparing document reranking methods
            </p>
            <div style="display: flex; justify-content: center; gap: 30px; margin-top: 20px;">
                <span>🔍 Multi-Modal Evaluation</span>
                <span>⚡ Real-time Processing</span>
                <span>📊 Advanced Analytics</span>
                <span>🏆 Fair Competition</span>
            </div>
        </div>
        """)