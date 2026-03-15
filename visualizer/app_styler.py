import streamlit as st


class AppStyler:
    def __init__(self, title: str, subtitle: str = ""):
        # Save the texts shown in the header
        self.title = title
        self.subtitle = subtitle

    def apply_styles(self):
        # Inject the main custom styles into the Streamlit page
        st.markdown("""
            <style>
                /* Main page background */
                .stApp {
                    background: linear-gradient(180deg, #eef3fb 0%, #f5f8fc 100%);
                }

                /* Top hero section */
                .hero-box {
                    background: linear-gradient(135deg, #16325c 0%, #244c86 100%);
                    border-radius: 24px;
                    padding: 28px 32px;
                    margin-bottom: 28px;
                    box-shadow: 0 10px 30px rgba(22, 50, 92, 0.18);
                }

                /* Top row inside the hero section */
                .hero-top {
                    display: flex;
                    align-items: center;
                    gap: 16px;
                    margin-bottom: 10px;
                }

                /* Weather icon in the header */
                .hero-icon {
                    font-size: 40px;
                    line-height: 1;
                }

                /* Main app title */
                .hero-title {
                    color: white;
                    font-size: 38px;
                    font-weight: 800;
                    margin: 0;
                }

                /* Subtitle below the main title */
                .hero-subtitle {
                    color: #dbe8ff;
                    font-size: 16px;
                    font-weight: 400;
                    margin-top: 6px;
                    margin-bottom: 0;
                }

                /* Small tag below the title area */
                .hero-tag {
                    display: inline-block;
                    margin-top: 16px;
                    padding: 8px 14px;
                    border-radius: 999px;
                    background: rgba(255, 255, 255, 0.16);
                    color: white;
                    font-size: 13px;
                    font-weight: 600;
                }

                /* Box around the selector section */
                .selector-box {
                    background: white;
                    border: 1px solid #d9e4f2;
                    border-radius: 18px;
                    padding: 18px 20px;
                    margin-bottom: 25px;
                    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
                }

                /* Box around the hourly timeline section */
                .timeline-box {
                    background: white;
                    border: 1px solid #d9e4f2;
                    border-radius: 22px;
                    padding: 20px;
                    margin-top: 20px;
                    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.07);
                }

                /* Section title inside white boxes */
                .section-title {
                    font-size: 22px;
                    font-weight: 700;
                    color: #0f172a;
                    margin-bottom: 12px;
                }
            </style>
        """, unsafe_allow_html=True)

    def render_header(self):
        # Show the top hero section
        st.markdown(f"""
            <div class="hero-box">
                <div class="hero-top">
                    <div class="hero-icon">🌦️</div>
                    <div>
                        <div class="hero-title">{self.title}</div>
                        <p class="hero-subtitle">{self.subtitle}</p>
                    </div>
                </div>
                <div class="hero-tag">Portugal Weather Intelligence</div>
            </div>
        """, unsafe_allow_html=True)

    def open_selector_box(self):
        # Open a styled white box for the selectors
        st.markdown('<div class="selector-box">', unsafe_allow_html=True)

    def open_timeline_box(self, title: str = "Hourly weather information"):
        # Open a styled white box for the hourly section
        st.markdown('<div class="timeline-box">', unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

    def close_box(self):
        # Close the previously opened box
        st.markdown('</div>', unsafe_allow_html=True)
