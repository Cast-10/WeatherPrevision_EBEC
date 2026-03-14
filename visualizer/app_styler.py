import streamlit as st


class AppStyler:
    def __init__(self, title: str, subtitle: str = ""):
        self.title = title
        self.subtitle = subtitle

    def apply_styles(self):
        st.markdown("""
            <style>
                .stApp {
                    background: #eef3fb;
                }
            </style>
        """, unsafe_allow_html=True)

    def render_header(self):
        st.markdown(f"<h1 style='text-align:center; color:#16325c;'>{self.title}</h1>", unsafe_allow_html=True)
        if self.subtitle:
            st.markdown(f"<p style='text-align:center; color:#1f3552; font-size:16px; font-weight:500;'>{self.subtitle}</p>", unsafe_allow_html=True)
    def open_selector_box(self):
        st.markdown("<div>", unsafe_allow_html=True)

    def open_timeline_box(self, title: str = "Hourly weather information"):
        st.markdown(f"<h3 style='color:#0f172a; font-weight:700;'>{title}</h3>", unsafe_allow_html=True)

    def close_box(self):
        st.markdown("</div>", unsafe_allow_html=True)
