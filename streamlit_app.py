import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import BytesIO
from PIL import Image
import sys
import os
from instagram_analysis import run_full_analysis_and_ideas, analyze_profile, analyze_posts, analyze_comments

# Plot Functions
def plot_donut_pie(data):
    fig, ax = plt.subplots(figsize=(4, 4))
    labels = list(data.keys())
    values = list(data.values())
    colors = ['#26A69A', '#9CCC65', '#FF7043']  # Teal, Lime, Coral
    wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%', colors=colors,
                                      textprops={'color': 'white'}, wedgeprops={'width': 0.4})
    ax.set_title("Sentiment Breakdown", fontsize=14, color='white')
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    return fig

def plot_engagement_pie(data):
    fig, ax = plt.subplots(figsize=(4, 4))
    labels = list(data.keys())
    values = list(data.values())
    colors = ['#26A69A', '#9CCC65', '#FF7043']  # Teal, Lime, Coral
    ax.pie(values, labels=labels, autopct='%1.1f%%', colors=colors, textprops={'color': 'white'})
    ax.set_title("Engagement Quality", fontsize=14, color='white')
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    return fig

# Streamlit Layout
st.set_page_config(layout="wide", page_title="Influencer D2C Dashboard")
st.markdown("""
    <style>
    .main {background-color: #12141c; color: white; font-family: 'Segoe UI', sans-serif;}
    .block-container {padding: 1rem;}
    h1, h2 {color: white; text-align: center; margin-bottom: 0.5rem;}
    .d2c-heading {font-size: 1.5rem; color: white; text-align: left; margin-bottom: 0.5rem;}
    .metric {margin-bottom: 0.4rem; font-size: 1rem;}
    .profile-pic {
        width: 130px;
        height: 130px;
        border-radius: 50%;
        border: 2px solid #E0E0E0;
        object-fit: cover;
        margin: 0 auto;
        display: block;
    }
    .stMetric {background-color: #1c1f26; border-radius: 8px; padding: 10px; margin-bottom: 0.5rem;}
    .stMetric label {font-size: 1.2rem; color: white;}
    .stMetric div[role="metric-value"] {font-size: 1.8rem; color: #26A69A; font-weight: bold; white-space: normal; word-wrap: break-word;}
    .best-post-time div[role="metric-value"] {font-size: 1.4rem; color: #26A69A; font-weight: bold; white-space: normal; word-wrap: break-word;}
    .stDataFrame {font-size: 1.2rem; color: white;}
    .stDataFrame table {background-color: #1c1f26; color: white;}
    .stDataFrame th, .stDataFrame td {color: white; border-color: #E0E0E0; padding: 10px; height: 40px;}
    .d2c-container {background-color: #1c1f26; padding: 1rem; border-radius: 8px; margin-top: 1rem;}
    hr {border: 0; height: 1px; background: #E0E0E0; margin: 1rem 0;}
    </style>
""", unsafe_allow_html=True)

st.title("✨ Influencer D2C Dashboard - House of X")

# Input and UI
username = st.text_input("Enter Instagram Username", value="")
apify_token = "apify_api_EtCTiSakLFyWlWYSjna8IX4yO6dEz73Kcsph"  # Hardcoded for simplicity
if st.button("Generate Dashboard"):
    if not username.strip():
        st.error("Please enter a valid Instagram username.")
    else:
        with st.spinner("Analyzing Instagram data..."):
            try:
                # Run the analysis script
                from io import StringIO
                sys.stdout = StringIO()
                run_full_analysis_and_ideas(username, apify_token)
                output = sys.stdout.getvalue()
                sys.stdout = sys.__stdout__

                # Check if CSV files were generated
                required_files = [
                    f"{username}_profile.csv",
                    f"{username}_posts.csv",
                    f"{username}_comments.csv",
                    f"{username}_d2c_ideas.csv"
                ]
                for file in required_files:
                    if not os.path.exists(file):
                        raise FileNotFoundError(f"Missing required file: {file}")

                # Process profile data
                profile_summary, followers = analyze_profile(f"{username}_profile.csv")
                profile_data = {
                    "Username": str(profile_summary.loc["Username", "Value"]),
                    "Full Name": str(profile_summary.loc["Full Name", "Value"]),
                    "Biography": str(profile_summary.loc["Biography", "Value"]),
                    "Followers": f"{int(profile_summary.loc['Followers', 'Value']):,}",
                    "Following": f"{int(profile_summary.loc['Following', 'Value']):,}",
                    "Post Count": f"{int(profile_summary.loc['Post Count', 'Value']):,}",
                    "Joined Recently": str(profile_summary.loc["Joined Recently", "Value"]),
                    "Engagement Potential": f"{float(profile_summary.loc['Engagement Potential', 'Value']):.1f}",
                    "Creator Type": str(profile_summary.loc["Creator Type", "Value"]),
                    "Trust Score": str(profile_summary.loc["Trust Score", "Value"]),
                    "Profile Picture URL": str(profile_summary.loc["Profile Picture URL", "Value"])
                }

                # Process post data
                post_summary = analyze_posts(f"{username}_posts.csv", followers)
                # Abbreviate Best Post Time for display
                best_post_time = post_summary["Best Post Time"]
                time_mappings = {
                    "Morning (6am–11am)": "Morning 6-11",
                    "Midday (11am–3pm)": "Midday 11-3",
                    "Afternoon (3pm–6pm)": "Afternoon 3-6",
                    "Evening (6pm–9pm)": "Evening 6-9",
                    "Night (9pm–12am)": "Night 9-12",
                    "Late Night (12am–6am)": "Late Night 12-6"
                }
                for full, short in time_mappings.items():
                    best_post_time = best_post_time.replace(full, short)
                post_data = {
                    "Engagement Rate": post_summary["Engagement Rate"],
                    "Content Vibe": post_summary["Content Vibe"],
                    "Content Type": post_summary["Content Type"],
                    "Best Post Time": best_post_time
                }

                # Process comment data
                comment_summary = analyze_comments(f"{username}_comments.csv")
                comment_data = {
                    "Sentiment Breakdown": comment_summary["Sentiment Breakdown"],
                    "Engagement Quality": comment_summary["Engagement Quality"]
                }

                # Process D2C ideas
                ideas_df = pd.read_csv(f"{username}_d2c_ideas.csv")
                idea_data = {
                    "Category": str(ideas_df["Category"].iloc[0]),
                    "Brand Name": str(ideas_df["Brand Name"].iloc[0]),
                    "Description": str(ideas_df["Description"].iloc[0]),
                    "Products": ideas_df["Product Ideas"].iloc[0].split("\n"),
                    "Revenue": str(ideas_df["Estimated Revenue"].iloc[0])
                }

                # Display Dashboard
                # Load profile image
                try:
                    response = requests.get(profile_data["Profile Picture URL"])
                    response.raise_for_status()
                    profile_image = Image.open(BytesIO(response.content))
                except:
                    profile_image = None
                    st.warning("Profile image not available. Using placeholder.")

                # Create three columns with adjusted widths
                col1, col2, col3 = st.columns([1, 2, 1])

                with col1:
                    st.markdown("<h2>PROFILE ANALYSIS</h2>", unsafe_allow_html=True)
                    if profile_image:
                        st.image(profile_image, width=130, use_column_width=False)
                    else:
                        st.markdown('<div class="profile-pic" style="background-color: #26A69A;"></div>', unsafe_allow_html=True)
                    st.markdown(f"<h2 style='text-align:center;'>{profile_data['Username']}</h2>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align:center;'>{profile_data['Biography']}</p>", unsafe_allow_html=True)
                    profile_table = {
                        "Attribute": ["Full Name", "Followers", "Following", "Post Count", "Joined Recently",
                                      "Engagement Potential", "Creator Type", "Trust Score"],
                        "Value": [profile_data[key] for key in ["Full Name", "Followers", "Following", "Post Count",
                                                                "Joined Recently", "Engagement Potential", "Creator Type", "Trust Score"]]
                    }
                    df_profile = pd.DataFrame(profile_table)
                    st.dataframe(df_profile, use_container_width=True, hide_index=True)

                with col2:
                    st.markdown("<h2>POST INSIGHTS</h2>", unsafe_allow_html=True)
                    st.metric(label="Engagement Rate", value=post_data["Engagement Rate"])
                    st.metric(label="Content Vibe", value=post_data["Content Vibe"])
                    st.metric(label="Content Type", value=post_data["Content Type"])
                    st.markdown('<div class="stMetric best-post-time">', unsafe_allow_html=True)
                    st.metric(label="Best Post Time", value=post_data["Best Post Time"])
                    st.markdown('</div>', unsafe_allow_html=True)

                with col3:
                    st.markdown("<h2>COMMENT INSIGHTS</h2>", unsafe_allow_html=True)
                    st.markdown("<div class='d2c-heading'>Sentiment Breakdown</div>", unsafe_allow_html=True)
                    st.pyplot(plot_donut_pie(comment_data["Sentiment Breakdown"]))
                    st.markdown("<div class='d2c-heading'>Engagement Quality</div>", unsafe_allow_html=True)
                    st.pyplot(plot_engagement_pie(comment_data["Engagement Quality"]))

                # D2C Business Idea Section
                st.markdown("---")
                st.markdown("<h2>💡 D2C Business Idea for House of X</h2>", unsafe_allow_html=True)
                with st.container():
                    st.markdown('<div class="d2c-container">', unsafe_allow_html=True)
                    st.markdown(f"<div class='d2c-heading'>Category</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric'>{idea_data['Category']}</div>", unsafe_allow_html=True)
                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.markdown(f"<div class='d2c-heading'>Brand Name</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric'>{idea_data['Brand Name']}</div>", unsafe_allow_html=True)
                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.markdown(f"<div class='d2c-heading'>Description</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric'>{idea_data['Description']}</div>", unsafe_allow_html=True)
                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.markdown(f"<div class='d2c-heading'>Product Ideas</div>", unsafe_allow_html=True)
                    for p in idea_data["Products"]:
                        st.markdown(f"<div class='metric'>• {p}</div>", unsafe_allow_html=True)
                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.markdown(f"<div class='d2c-heading'>Estimated Revenue</div>", unsafe_allow_html=True)
                    st.metric(label="Estimated Revenue", value=idea_data["Revenue"])
                    st.markdown('</div>', unsafe_allow_html=True)

            except FileNotFoundError as e:
                st.error(f"Data generation failed: {str(e)}. Ensure the username is valid and the account is public.")
            except ValueError as e:
                st.error(f"Analysis error: {str(e)}")
                if "private" in str(e).lower():
                    st.warning("The account may be private. Please use a public Instagram account.")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}. Please try again or check the logs.")
                st.write(f"Debug output: {output}")

