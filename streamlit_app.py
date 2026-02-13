"""
Medical Embedding Visualization - Professional Portfolio Demo
Fine-tuned BioBERT with Triplet Margin Loss
Author: Amulya Penikialapati
"""

import streamlit as st
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import umap

# Page config with custom theme
st.set_page_config(
    page_title="Medical Embedding Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    /* Main background and text */
    .main {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
        color: #FEC5F6;
    }
    
    /* Headers with gradient */
    h1 {
        background: linear-gradient(90deg, #00d9ff 0%, #00ffff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 0.5rem !important;
    }
    
    h2 {
        color: #00d9ff !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        margin-top: 2rem !important;
        border-bottom: 2px solid #00d9ff;
        padding-bottom: 0.5rem;
    }
    
    h3 {
        color: #00ffff !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
    }
    
    /* Metric cards */
    .stMetric {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #00d9ff;
        box-shadow: 0 0 30px rgba(0, 217, 255, 0.3);
        transition: transform 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 0 0 40px rgba(0, 217, 255, 0.5);
    }
    
    .stMetric label {
        color: #FEC5F6 !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #00ffff !important;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
    }
    
    .stMetric [data-testid="stMetricDelta"] {
        font-size: 1.2rem !important;
        font-weight: 700 !important;
    }
    
    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1e 0%, #1a1a2e 100%);
        border-right: 3px solid #00d9ff;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(90deg, #00d9ff 0%, #00ffff 100%);
        color: #000000;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        box-shadow: 0 0 20px rgba(0, 217, 255, 0.5);
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 30px rgba(0, 255, 255, 0.7);
    }
    
    /* Info boxes */
    .stAlert {
        background: rgba(0, 217, 255, 0.1);
        border-left: 5px solid #00d9ff;
        border-radius: 10px;
        color: #FEC5F6;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(0, 217, 255, 0.1);
        border-radius: 10px;
        color: #00ffff !important;
        font-weight: 600;
    }
    
    /* Selectbox */
    .stSelectbox label {
        color: #FEC5F6 !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(0, 217, 255, 0.1);
        border-radius: 10px;
        color: #FEC5F6;
        font-weight: 600;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #00d9ff 0%, #00ffff 100%);
        color: #000000;
    }
    
    /* Dataframe */
    .dataframe {
        border: 2px solid #00d9ff !important;
        border-radius: 10px;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00d9ff 0%, #00ffff 100%);
    }
</style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown("# 🧬 Medical Embedding Analysis Platform")
st.markdown("""
<p style='text-align: center; font-size: 1.3rem; color: #FEC5F6; margin-bottom: 2rem;'>
Fine-tuning BioBERT with Triplet Margin Loss for Medical Specialty Classification
</p>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # Model selection with icons
    model_options = {
        'sentence-bert': '🔤 Sentence-BERT (General)',
        'biobert': '🧪 BioBERT (Biomedical)',
        'clinical-bert': '🏥 ClinicalBERT (Clinical)',
        'triplet-biobert': '⭐ Fine-tuned BioBERT'
    }

    selected_model = st.selectbox(
        "Embedding Model:",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=3
    )

    # Dimensionality reduction
    reduction_method = st.radio(
        "Dimensionality Reduction:",
        ['t-SNE', 'UMAP'],
        help="t-SNE: Pre-computed | UMAP: On-demand"
    )

    st.markdown("---")

    # Quick stats
    st.markdown("### 📊 Quick Stats")
    st.metric("Dataset Size", "900 notes")
    st.metric("Specialties", "6")
    st.metric("Training Time", "~6 hours")

    st.markdown("---")

    # About
    with st.expander("ℹ️ About"):
        st.markdown("""
        **Author:** Amulya Penikialapati
        
        **Technologies:**
        - PyTorch, Transformers
        - FastAPI, Streamlit
        - t-SNE, UMAP, Plotly
        
        **Links:**
        - [GitHub](https://github.com/yourusername)
        - [LinkedIn](https://linkedin.com/in/yourprofile)
        """)

# Load data
@st.cache_data
def load_data():
    with open('backend/cache/data_info.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

@st.cache_data
def load_embeddings(model_name):
    return np.load(f'backend/cache/embeddings_{model_name}.npy')

@st.cache_data
def load_tsne(model_name):
    try:
        return np.load(f'backend/cache/tsne_{model_name}.npy')
    except:
        return None

@st.cache_data
def compute_metrics(embeddings, labels):
    """Compute clustering quality metrics"""
    sil_score = silhouette_score(embeddings, labels)
    db_index = davies_bouldin_score(embeddings, labels)

    # Precision@5
    precisions = []
    for i, (emb, label) in enumerate(zip(embeddings, labels)):
        sims = cosine_similarity([emb], embeddings)[0]
        top5_indices = np.argsort(sims)[::-1][1:6]
        matches = sum(1 for idx in top5_indices if labels[idx] == label)
        precisions.append(matches / 5)

    precision_at_5 = np.mean(precisions)

    return {
        'silhouette': sil_score,
        'davies_bouldin': db_index,
        'precision_at_5': precision_at_5
    }

# Load data
data = load_data()
texts = data['texts']
specialties = data['specialties']

# Specialty colors
specialty_colors = {
    'Surgery': '#ff1493',
    'Orthopedic': '#00ffff',
    'Radiology': '#ffd700',
    'Gastroenterology': '#ff8c00',
    'Neurology': '#00ff00',
    'Cardiovascular / Pulmonary': '#ff00ff'
}

# Tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance Metrics", "🎯 3D Visualization", "📈 Comparison Analysis", "🔬 Technical Details"])

# ==================== TAB 1: PERFORMANCE METRICS ====================
with tab1:
    st.markdown("## 🎯 Model Performance Overview")

    # Load both models for comparison
    biobert_emb = load_embeddings('biobert')
    triplet_emb = load_embeddings('triplet-biobert')

    # Compute metrics
    with st.spinner('Computing metrics...'):
        biobert_metrics = compute_metrics(biobert_emb, specialties)
        triplet_metrics = compute_metrics(triplet_emb, specialties)

    # Hero metrics
    st.markdown("### 🏆 Key Achievements")
    col1, col2, col3 = st.columns(3)

    with col1:
        improvement = ((triplet_metrics['precision_at_5'] - biobert_metrics['precision_at_5']) / biobert_metrics['precision_at_5']) * 100
        st.metric(
            "Precision@5",
            f"{triplet_metrics['precision_at_5']:.1%}",
            delta=f"+{improvement:.1f}%",
            help="Retrieval accuracy: % of top-5 results that match the query specialty"
        )

    with col2:
        db_improvement = ((biobert_metrics['davies_bouldin'] - triplet_metrics['davies_bouldin']) / biobert_metrics['davies_bouldin']) * 100
        st.metric(
            "Davies-Bouldin Index",
            f"{triplet_metrics['davies_bouldin']:.2f}",
            delta=f"-{db_improvement:.1f}%",
            delta_color="inverse",
            help="Cluster separation (lower is better)"
        )

    with col3:
        st.metric(
            "Silhouette Score",
            f"{triplet_metrics['silhouette']:.3f}",
            delta=f"From {biobert_metrics['silhouette']:.3f}",
            help="Clustering quality (-1 to 1, higher is better)"
        )

    st.markdown("---")

    # Detailed comparison
    st.markdown("### 📊 Detailed Metrics Comparison")

    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Metric': ['Silhouette Score', 'Davies-Bouldin Index', 'Precision@5'],
        'Original BioBERT': [
            f"{biobert_metrics['silhouette']:.4f}",
            f"{biobert_metrics['davies_bouldin']:.4f}",
            f"{biobert_metrics['precision_at_5']:.2%}"
        ],
        'Fine-tuned BioBERT': [
            f"{triplet_metrics['silhouette']:.4f}",
            f"{triplet_metrics['davies_bouldin']:.4f}",
            f"{triplet_metrics['precision_at_5']:.2%}"
        ],
        'Improvement': [
            f"+{((triplet_metrics['silhouette'] - biobert_metrics['silhouette']) / abs(biobert_metrics['silhouette']) * 100):.1f}%",
            f"-{db_improvement:.1f}%",
            f"+{improvement:.1f}%"
        ]
    })

    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # Visual comparison charts
    col1, col2 = st.columns(2)

    with col1:
        # Bar chart for metrics
        fig = go.Figure()

        metrics_names = ['Precision@5', 'Silhouette Score']
        original_values = [biobert_metrics['precision_at_5'], biobert_metrics['silhouette']]
        finetuned_values = [triplet_metrics['precision_at_5'], triplet_metrics['silhouette']]

        fig.add_trace(go.Bar(
            name='Original BioBERT',
            x=metrics_names,
            y=original_values,
            marker_color='#ff8c00',
            text=[f"{v:.3f}" for v in original_values],
            textposition='auto'
        ))

        fig.add_trace(go.Bar(
            name='Fine-tuned BioBERT',
            x=metrics_names,
            y=finetuned_values,
            marker_color='#00d9ff',
            text=[f"{v:.3f}" for v in finetuned_values],
            textposition='auto'
        ))

        fig.update_layout(
            title="Performance Comparison",
            barmode='group',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FEC5F6', size=14),
            showlegend=True,
            legend=dict(bgcolor='rgba(0,0,0,0.5)'),
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Improvement percentage chart
        improvements = [
            ('Precision@5', improvement),
            ('Cluster Separation', db_improvement),
            ('Silhouette', ((triplet_metrics['silhouette'] - biobert_metrics['silhouette']) / abs(biobert_metrics['silhouette']) * 100))
        ]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=[name for name, _ in improvements],
            y=[val for _, val in improvements],
            marker_color=['#00ff00' if val > 0 else '#ff0000' for _, val in improvements],
            text=[f"+{val:.1f}%" if val > 0 else f"{val:.1f}%" for _, val in improvements],
            textposition='auto'
        ))

        fig.update_layout(
            title="Improvement Percentages",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FEC5F6', size=14),
            yaxis_title="Improvement (%)",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 2: 3D VISUALIZATION ====================
with tab2:
    st.markdown("## 🎯 Interactive 3D Embedding Space")

    # Load embeddings and coordinates
    embeddings = load_embeddings(selected_model)

    if reduction_method == 't-SNE':
        coords_3d = load_tsne(selected_model)
        if coords_3d is None:
            st.warning("t-SNE not pre-computed. Computing UMAP instead...")
            with st.spinner('Computing UMAP...'):
                reducer = umap.UMAP(n_components=3, random_state=42)
                coords_3d = reducer.fit_transform(embeddings)
    else:
        with st.spinner('Computing UMAP... This may take a minute.'):
            reducer = umap.UMAP(n_components=3, random_state=42)
            coords_3d = reducer.fit_transform(embeddings)

    # Info metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Points", len(texts))
    with col2:
        st.metric("Specialties", len(set(specialties)))
    with col3:
        st.metric("Embedding Dim", embeddings.shape[1])
    with col4:
        st.metric("Model", model_options[selected_model].split(' ')[1])

    # 3D Plot
    fig = go.Figure()

    for specialty in sorted(set(specialties)):
        mask = [s == specialty for s in specialties]
        indices = [i for i, m in enumerate(mask) if m]

        fig.add_trace(go.Scatter3d(
            x=coords_3d[indices, 0],
            y=coords_3d[indices, 1],
            z=coords_3d[indices, 2],
            mode='markers',
            name=specialty,
            marker=dict(
                size=6,
                color=specialty_colors.get(specialty, '#ffffff'),
                opacity=0.8,
                line=dict(width=0.5, color='white')
            ),
            text=[f"<b>{specialty}</b><br>{texts[i][:150]}..." for i in indices],
            hovertemplate='%{text}<extra></extra>'
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                showbackground=True,
                backgroundcolor='rgba(0,0,0,0.3)',
                showgrid=True,
                gridcolor='#333',
                title=f'{reduction_method} 1'
            ),
            yaxis=dict(
                showbackground=True,
                backgroundcolor='rgba(0,0,0,0.3)',
                showgrid=True,
                gridcolor='#333',
                title=f'{reduction_method} 2'
            ),
            zaxis=dict(
                showbackground=True,
                backgroundcolor='rgba(0,0,0,0.3)',
                showgrid=True,
                gridcolor='#333',
                title=f'{reduction_method} 3'
            ),
            bgcolor='#000000'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FEC5F6', size=12),
        height=700,
        showlegend=True,
        legend=dict(
            bgcolor='rgba(0,0,0,0.8)',
            bordercolor='#00d9ff',
            borderwidth=2,
            font=dict(size=12)
        ),
        title=dict(
            text=f"Medical Specialties - {model_options[selected_model]}",
            font=dict(size=20, color='#00d9ff')
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # Specialty distribution
    st.markdown("### 📊 Specialty Distribution")
    specialty_counts = pd.Series(specialties).value_counts()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=specialty_counts.index,
        y=specialty_counts.values,
        marker_color=[specialty_colors.get(s, '#ffffff') for s in specialty_counts.index],
        text=specialty_counts.values,
        textposition='auto'
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FEC5F6'),
        xaxis_title="Specialty",
        yaxis_title="Number of Notes",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 3: COMPARISON ANALYSIS ====================
with tab3:
    st.markdown("## 📈 Before vs After Analysis")

    st.info("🔍 Compare Original BioBERT vs Fine-tuned BioBERT side-by-side")

    # Side by side 3D plots
    col1, col2 = st.columns(2)

    for col, model_id, title in [(col1, 'biobert', 'Original BioBERT'), (col2, 'triplet-biobert', 'Fine-tuned BioBERT')]:
        with col:
            st.markdown(f"### {title}")

            emb = load_embeddings(model_id)
            coords = load_tsne(model_id)

            if coords is None:
                st.warning("Pre-computing coordinates...")
                reducer = umap.UMAP(n_components=3, random_state=42)
                coords = reducer.fit_transform(emb)

            fig = go.Figure()

            for specialty in sorted(set(specialties)):
                mask = [s == specialty for s in specialties]
                indices = [i for i, m in enumerate(mask) if m]

                fig.add_trace(go.Scatter3d(
                    x=coords[indices, 0],
                    y=coords[indices, 1],
                    z=coords[indices, 2],
                    mode='markers',
                    name=specialty,
                    marker=dict(
                        size=4,
                        color=specialty_colors.get(specialty, '#ffffff'),
                        opacity=0.7
                    ),
                    showlegend=False
                ))

            fig.update_layout(
                scene=dict(
                    xaxis=dict(showbackground=False, showgrid=True, gridcolor='#333'),
                    yaxis=dict(showbackground=False, showgrid=True, gridcolor='#333'),
                    zaxis=dict(showbackground=False, showgrid=True, gridcolor='#333'),
                    bgcolor='#000000'
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=0, b=0),
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

    # Metrics evolution
    st.markdown("### 📊 Metrics Evolution")

    metrics_df = pd.DataFrame({
        'Metric': ['Silhouette Score', 'Davies-Bouldin', 'Precision@5'],
        'Original': [biobert_metrics['silhouette'], biobert_metrics['davies_bouldin'], biobert_metrics['precision_at_5']],
        'Fine-tuned': [triplet_metrics['silhouette'], triplet_metrics['davies_bouldin'], triplet_metrics['precision_at_5']]
    })

    fig = make_subplots(rows=1, cols=3, subplot_titles=metrics_df['Metric'].tolist())

    for i, metric in enumerate(metrics_df['Metric'], 1):
        row = metrics_df[metrics_df['Metric'] == metric].iloc[0]

        fig.add_trace(
            go.Scatter(
                x=['Original', 'Fine-tuned'],
                y=[row['Original'], row['Fine-tuned']],
                mode='lines+markers',
                line=dict(color='#00d9ff', width=3),
                marker=dict(size=12, color=['#ff8c00', '#00ff00']),
                showlegend=False
            ),
            row=1, col=i
        )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FEC5F6'),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 4: TECHNICAL DETAILS ====================
with tab4:
    st.markdown("## 🔬 Technical Implementation")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🏗️ Model Architecture")
        st.code("""
Base Model: BioBERT (dmis-lab/biobert-v1.1)
├── Transformer Layers: 12
├── Hidden Size: 768
├── Attention Heads: 12
└── Projection Head: 768 → 768
    ├── Linear(768, 768)
    ├── ReLU
    ├── Dropout(0.1)
    └── Linear(768, 768)
        """, language="text")

        st.markdown("### ⚡ Training Configuration")
        st.code("""
Loss Function: Triplet Margin Loss
├── Margin: 0.5
├── Distance: L2 (Euclidean)
└── Reduction: Mean

Optimizer: AdamW
├── Learning Rate: 2e-5
├── Weight Decay: 0.01
└── Betas: (0.9, 0.999)

Scheduler: Cosine Annealing
└── T_max: 5 epochs
        """, language="text")

    with col2:
        st.markdown("### 📊 Training Data")
        training_data = pd.DataFrame({
            'Parameter': ['Training Triplets', 'Validation Triplets', 'Batch Size', 'Epochs', 'Total Samples'],
            'Value': ['2000', '400', '8', '5', '900 notes']
        })
        st.dataframe(training_data, use_container_width=True, hide_index=True)

        st.markdown("### 🎯 Evaluation Metrics")
        st.markdown("""
        **Silhouette Score** 
        - Range: [-1, 1]
        - Measures: Intra-cluster cohesion vs inter-cluster separation
        - Formula: (b - a) / max(a, b)
        
        **Davies-Bouldin Index**
        - Range: [0, ∞]
        - Measures: Average similarity ratio of clusters
        - Lower is better
        
        **Precision@K**
        - Range: [0, 1]
        - Measures: Retrieval accuracy in top-K results
        - K=5 for this project
        """)

    st.markdown("---")

    st.markdown("### 🚀 Applications to Robotics")

    robotics_apps = pd.DataFrame({
        'NLP Technique': ['Triplet Loss', 'Embedding Clustering', 'Distance Metrics', 'Dimensionality Reduction'],
        'Robotics Application': [
            'Object Re-identification',
            'Point Cloud Segmentation',
            'Sensor Fusion',
            'State Space Compression'
        ],
        'Use Case': [
            'Same object from different viewpoints',
            'Segmenting 3D sensor data',
            'Combining multi-modal sensor readings',
            'Reducing high-D state representations'
        ]
    })

    st.dataframe(robotics_apps, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Training timeline
    st.markdown("### ⏱️ Training Timeline")

    timeline_data = {
        'Phase': ['Data Preparation', 'Triplet Generation', 'Model Training (5 epochs)', 'Evaluation', 'Total'],
        'Time': ['~5 min', '~2 min', '~6 hours', '~3 min', '~6.2 hours']
    }

    timeline_df = pd.DataFrame(timeline_data)
    st.dataframe(timeline_df, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #FEC5F6; padding: 20px;'>
    <p style='font-size: 1.2rem; margin-bottom: 10px;'><b>Medical Embedding Analysis Platform</b></p>
    <p>Built with using PyTorch, Transformers, and Streamlit</p>
    <p style='margin-top: 10px;'>
        <a href='https://github.com/yourusername' style='color: #00d9ff; text-decoration: none; margin: 0 15px;'>GitHub</a> |
        <a href='https://linkedin.com/in/yourprofile' style='color: #00d9ff; text-decoration: none; margin: 0 15px;'>LinkedIn</a> |
        <a href='mailto:your.email@example.com' style='color: #00d9ff; text-decoration: none; margin: 0 15px;'>Email</a>
    </p>
</div>
""", unsafe_allow_html=True)