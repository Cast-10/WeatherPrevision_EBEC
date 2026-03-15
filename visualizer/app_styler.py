import streamlit as st


class AppStyler:
    def __init__(self, title: str, subtitle: str):
        self.title = title
        self.subtitle = subtitle

    def apply_styles(self):
        st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
    color: #0f172a;
}

.main > div {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

.hero-card {
    background: linear-gradient(135deg, #163d7a 0%, #244f94 100%);
    border-radius: 24px;
    padding: 2rem;
    box-shadow: 0 14px 32px rgba(15, 23, 42, 0.16);
    color: white;
    margin-bottom: 1.8rem;
}

.hero-title {
    font-size: 2.05rem;
    font-weight: 800;
    margin-bottom: 0.45rem;
    color: white;
    line-height: 1.1;
}

.hero-subtitle {
    font-size: 0.98rem;
    color: rgba(255, 255, 255, 0.86);
    margin-bottom: 1rem;
    line-height: 1.45;
}

.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.14);
    color: white;
    font-size: 0.76rem;
    font-weight: 700;
    padding: 0.42rem 0.82rem;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.14);
}

.section-kicker {
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #3b82f6;
    margin-bottom: 0.45rem;
}

.section-title {
    font-size: 1.8rem;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 0.5rem;
    line-height: 1.15;
}

.section-subtitle {
    font-size: 0.98rem;
    color: #475569;
    margin-bottom: 1.15rem;
    line-height: 1.5;
}

.box-card {
    background: rgba(255,255,255,0.96);
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 1.15rem;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    margin-bottom: 1rem;
}

.timeline-box {
    background: rgba(255,255,255,0.96);
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 1rem;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    margin-top: 1rem;
    margin-bottom: 1rem;
}

.timeline-title {
    font-size: 1.05rem;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 0.8rem;
}

.weather-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 1.2rem 1rem;
    text-align: center;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
    min-height: 118px;
}

.weather-card-label {
    font-size: 0.82rem;
    color: #64748b;
    font-weight: 700;
    margin-bottom: 0.65rem;
    line-height: 1.35;
}

.weather-card-value {
    font-size: 1.95rem;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.1;
}

.weather-card-unit {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 0.35rem;
}

.accident-card {
    background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%);
    border: 1px solid #fdba74;
    border-left: 6px solid #ea580c;
    border-radius: 18px;
    padding: 18px;
    margin-top: 18px;
    margin-bottom: 18px;
    box-shadow: 0 10px 22px rgba(234, 88, 12, 0.10);
}

.accident-badge {
    display: inline-block;
    background: #ea580c;
    color: white;
    font-size: 0.76rem;
    font-weight: 700;
    padding: 6px 12px;
    border-radius: 999px;
    margin-bottom: 10px;
}

.accident-title {
    font-size: 1.2rem;
    font-weight: 800;
    color: #9a3412;
    margin-bottom: 0.35rem;
    line-height: 1.2;
}

.accident-subtitle {
    font-size: 0.92rem;
    color: #7c2d12;
    line-height: 1.45;
    margin-bottom: 0;
}

.empty-state {
    background: rgba(255,255,255,0.9);
    border: 1px dashed #cbd5e1;
    border-radius: 20px;
    padding: 2rem 1.25rem;
    text-align: center;
    color: #475569;
    margin-top: 1rem;
}

.empty-state-title {
    font-size: 1.15rem;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 0.35rem;
}

.empty-state-text {
    font-size: 0.95rem;
    color: #64748b;
    line-height: 1.5;
}

div[data-baseweb="select"] > div {
    min-height: 46px;
    border-radius: 14px !important;
    border-color: #cbd5e1 !important;
    box-shadow: none !important;
}

div[data-baseweb="select"] > div:hover {
    border-color: #94a3b8 !important;
}

input, .stDateInput input {
    border-radius: 14px !important;
}

div[data-testid="stDateInput"] > div {
    border-radius: 14px !important;
}
</style>
""", unsafe_allow_html=True)

    def render_header(self):
        st.markdown(
            f"""
<div class="hero-card">
    <div class="hero-title">🌤️ {self.title}</div>
    <div class="hero-subtitle">{self.subtitle}</div>
    <div class="hero-badge">Portugal Weather Intelligence</div>
</div>
""",
            unsafe_allow_html=True
        )

    def render_section_intro(self):
        st.markdown(
            """
<div class="section-kicker">Weather Dashboard</div>
<div class="section-title">Explore weather conditions across Portugal</div>
<div class="section-subtitle">
    Select a district and a date to access daily summaries, hourly forecasts, and future weather insights.
</div>
""",
            unsafe_allow_html=True
        )

    def open_selector_box(self):
        st.markdown("<div class='box-card'>", unsafe_allow_html=True)

    def close_box(self):
        st.markdown("</div>", unsafe_allow_html=True)

    def open_timeline_box(self, title: str):
        st.markdown(
            f"""
<div class="timeline-box">
    <div class="timeline-title">{title}</div>
""",
            unsafe_allow_html=True
        )

    def render_empty_state(self):
        st.markdown(
            """
<div class="empty-state">
    <div class="empty-state-title">No data selected yet</div>
    <div class="empty-state-text">
        Choose a district and a date to explore weather summaries, hourly timelines, and future predictions.
    </div>
</div>
""",
            unsafe_allow_html=True
        )