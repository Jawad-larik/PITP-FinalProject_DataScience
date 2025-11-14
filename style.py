import streamlit as st

def load_css(dark_mode: bool = True):
    if dark_mode:
        theme_vars = """
        :root { 
            --purple1: #6a11cb; 
            --purple2: #2575fc; 
            --card-bg: #0F1724; 
            --text: #E6EEF8; 
            --muted: #9AA4B2;
        }
        """
    else:
        theme_vars = """
        :root { 
            --purple1: #6a11cb; 
            --purple2: #2575fc; 
            --card-bg: #FFFFFF; 
            --text: #111827; 
            --muted: #6B7280;
        }
        """
    animated_title_css = """
    h1 {
        font-size: 2.8rem !important;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, var(--purple1) 20%, var(--purple2) 80%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent !important;
        animation: textShine 8s linear infinite;
        transition: all 0.3s ease;
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    h1:hover {
        transform: scale(1.02);
        text-shadow: 0 0 15px rgba(37, 117, 252, 0.3);
    }

    @keyframes textShine {
        to {
            background-position: 200% center;
        }
    }
    """
    footer_css = """
    .footer {
        text-align: center;
        padding: 2rem 1rem 1rem 1rem;
        color: var(--muted);
        font-size: 0.9rem;
    }
    """
    other_styles = """
    .prediction-card {
        background: var(--card-bg);
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 6px 20px rgba(12,16,23,0.12);
        text-align: center;
    }
    .small-muted { color: var(--muted); font-size:0.9rem }
    """
    st.markdown(f"<style>{theme_vars} {animated_title_css} {footer_css} {other_styles}</style>", unsafe_allow_html=True)

def add_footer(name: str):
    st.markdown(
        f'<div class="footer">Made with ❤️ by {name}</div>', 
        unsafe_allow_html=True
    )
def add_footer(name: str):
    st.markdown(
        f"""
        <div class="footer">
        | Advanced Student Performance Prediction System © 2025. All Rights Reserved. |
        <br>
        PITP Final Project - \"Developed by: {name}\"
        </div>
        """, 
        unsafe_allow_html=True
    )