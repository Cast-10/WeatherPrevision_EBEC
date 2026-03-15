import streamlit as st


class MainPage:
    def __init__(self):
        # This class controls the homepage presentation
        pass

    def render_intro(self):
        # Main introduction block
        with st.container(border=True):
            st.markdown(
                "<p style='color:#2f5b9a; font-size:13px; font-weight:700; letter-spacing:0.08em; margin-bottom:6px;'>WEATHER DASHBOARD</p>",
                unsafe_allow_html=True
            )
            st.markdown(
                "<h2 style='color:#0f172a; margin-top:0; margin-bottom:10px;'>Explore weather conditions across Portugal</h2>",
                unsafe_allow_html=True
            )
            st.markdown(
                "<p style='color:#334155; font-size:16px; line-height:1.7; margin-bottom:0;'>"
                "Select a district and a date to access daily summaries, hourly forecasts, "
                "and future weather insights in a clean and intuitive interface."
                "</p>",
                unsafe_allow_html=True
            )

        st.write("")

        # Three highlight cards
        col1, col2, col3 = st.columns(3)

        with col1:
            with st.container(border=True):
                st.markdown("<div style='font-size:28px;'>📍</div>", unsafe_allow_html=True)
                st.markdown(
                    "<p style='color:#0f172a; font-size:18px; font-weight:700; margin-bottom:6px;'>District-based view</p>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    "<p style='color:#475569; font-size:14px; line-height:1.6; margin-bottom:0;'>"
                    "Explore weather information by district across Portugal."
                    "</p>",
                    unsafe_allow_html=True
                )

        with col2:
            with st.container(border=True):
                st.markdown("<div style='font-size:28px;'>🕒</div>", unsafe_allow_html=True)
                st.markdown(
                    "<p style='color:#0f172a; font-size:18px; font-weight:700; margin-bottom:6px;'>Hourly forecast</p>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    "<p style='color:#475569; font-size:14px; line-height:1.6; margin-bottom:0;'>"
                    "View the evolution of weather conditions throughout the day."
                    "</p>",
                    unsafe_allow_html=True
                )

        with col3:
            with st.container(border=True):
                st.markdown("<div style='font-size:28px;'>🔮</div>", unsafe_allow_html=True)
                st.markdown(
                    "<p style='color:#0f172a; font-size:18px; font-weight:700; margin-bottom:6px;'>Future predictions</p>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    "<p style='color:#475569; font-size:14px; line-height:1.6; margin-bottom:0;'>"
                    "Extend beyond historical data and support forecast exploration."
                    "</p>",
                    unsafe_allow_html=True
                )

    def render_empty_state(self):
        # Empty state shown before the user selects anything
        with st.container(border=True):
            st.markdown(
                "<div style='text-align:center; font-size:42px; margin-bottom:8px;'>🌦️</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                "<h3 style='text-align:center; color:#0f172a; margin-bottom:10px;'>Start your weather exploration</h3>",
                unsafe_allow_html=True
            )
            st.markdown(
                "<p style='text-align:center; color:#475569; font-size:15px; line-height:1.7; margin-bottom:0;'>"
                "Choose a district and a date above to unlock the daily summary and the hourly forecast view."
                "</p>",
                unsafe_allow_html=True
            )
