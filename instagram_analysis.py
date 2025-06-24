from apify_client import ApifyClient
import pandas as pd
from textblob import TextBlob
from collections import Counter
import re
import nltk
from transformers import pipeline
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import requests
import json
import time
import os
from dotenv import load_dotenv

# NLTK setup (run once: nltk.download('punkt'))
# nltk.download('punkt')

# Load environment variables
load_dotenv()

# Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyADQnHd6_TvwVkTAYAadaflLTLePBevD68")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

# Zero-shot classifier for content type
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
CONTENT_LABELS = [
    "Comedy", "Fashion", "Tech", "Fitness", "Education", "Luxury",
    "Food", "Parenting", "Motivation", "Travel", "Beauty", "Lifestyle"
]

def map_hour_to_slot(hour):
    if 6 <= hour < 11: return "Morning (6am–11am)"
    elif 11 <= hour < 15: return "Midday (11am–3pm)"
    elif 15 <= hour < 18: return "Afternoon (3pm–6pm)"
    elif 18 <= hour < 21: return "Evening (6pm–9pm)"
    elif 21 <= hour < 24: return "Night (9pm–12am)"
    else: return "Late Night (12am–6am)"

# 1️⃣ Scrape Instagram data using Apify
def scrape_instagram_all(username, apify_token):
    client = ApifyClient(apify_token)
    base_url = f"https://www.instagram.com/{username}/"

    # Profile
    profile_input = {
        "directUrls": [base_url], "resultsType": "details",
        "resultsLimit": 1, "addParentData": False
    }
    profile_run = client.actor("shu8hvrXbJbY3Eb9W").call(run_input=profile_input)
    profile_items = list(client.dataset(profile_run["defaultDatasetId"]).iterate_items())
    if not profile_items:
        raise ValueError("❌ Profile data not found or account may be private.")
    profile_data = profile_items[0]
    pd.DataFrame([profile_data]).to_csv(f"{username}_profile.csv", index=False)

    # Posts & Comments
    posts_input = {
        "directUrls": [base_url], "resultsType": "posts",
        "resultsLimit": 10, "addParentData": False
    }
    posts_run = client.actor("shu8hvrXbJbY3Eb9W").call(run_input=posts_input)
    post_items = list(client.dataset(posts_run["defaultDatasetId"]).iterate_items())

    posts_data, comments_data = [], []
    for post in post_items:
        posts_data.append({
            "Post URL": post.get("url"),
            "Caption": post.get("caption"),
            "Likes": post.get("likesCount"),
            "Comments": post.get("commentsCount"),
            "Timestamp": post.get("timestamp"),
            "Location": post.get("locationName"),
            "Hashtags": post.get("hashtags"),
            "Mentions": post.get("mentions")
        })
        for comment in post.get("latestComments", []):
            comments_data.append({
                "Post URL": post.get("url"),
                "Comment Username": comment.get("ownerUsername"),
                "Comment": comment.get("text"),
                "Likes on Comment": comment.get("likesCount"),
                "Comment Timestamp": comment.get("timestamp")
            })
    pd.DataFrame(posts_data).to_csv(f"{username}_posts.csv", index=False)
    pd.DataFrame(comments_data).to_csv(f"{username}_comments.csv", index=False)

# 2️⃣ Analyze profile
def analyze_profile(profile_csv):
    df = pd.read_csv(profile_csv)
    df["Engagement Potential"] = df["followersCount"] / df["followsCount"]
    df["Creator Type"] = df["joinedRecently"].apply(lambda x: "New" if x else "Established")
    df["Trust Score"] = df["verified"].astype(int) + df["isBusinessAccount"].astype(int)
    summary = df[[
        "username", "fullName", "biography", "followersCount", "followsCount",
        "postsCount", "joinedRecently", "Engagement Potential", "Creator Type",
        "Trust Score", "profilePicUrlHD"
    ]]
    summary = summary.rename(columns={
        "username": "Username",
        "fullName": "Full Name",
        "biography": "Biography",
        "followersCount": "Followers",
        "followsCount": "Following",
        "postsCount": "Post Count",
        "joinedRecently": "Joined Recently",
        "profilePicUrlHD": "Profile Picture URL"
    })
    return summary.T.rename(columns={0: "Value"}), int(df["followersCount"].iloc[0])

# 3️⃣ Analyze posts
def analyze_posts(posts_csv, followers_count):
    df = pd.read_csv(posts_csv)
    df["Engagement Rate (%)"] = ((df["Likes"] + df["Comments"]) / followers_count) * 100
    avg_engagement = round(df["Engagement Rate (%)"].mean(), 2)
    df["Caption"] = df["Caption"].fillna("")
    df["Polarity"] = df["Caption"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df["Tone"] = df["Polarity"].apply(lambda p: "Positive" if p > 0.2 else "Negative" if p < -0.2 else "Neutral")
    tone = df["Tone"].value_counts().idxmax()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
    df["Post Hour"] = df["Timestamp"].dt.hour
    df["Time Slot"] = df["Post Hour"].apply(map_hour_to_slot)
    best_times = ", ".join(df["Time Slot"].value_counts().head(2).index.tolist())
    captions = df["Caption"].tolist()
    label_scores = {label: 0.0 for label in CONTENT_LABELS}
    for caption in captions:
        if caption.strip() == "": continue
        result = classifier(caption, CONTENT_LABELS)
        for label, score in zip(result["labels"], result["scores"]):
            label_scores[label] += score
    content_type = max(label_scores.items(), key=lambda x: x[1])[0]
    return {
        "Engagement Rate": f"{avg_engagement}%",
        "Content Vibe": tone,
        "Content Type": content_type,
        "Best Post Time": best_times
    }

# 4️⃣ Analyze comments
def analyze_comments(comments_csv):
    df = pd.read_csv(comments_csv)
    df["Comment"] = df["Comment"].fillna("").astype(str)
    df["Sentiment"] = df["Comment"].apply(lambda t: "Positive" if TextBlob(t).sentiment.polarity > 0.2 else "Negative" if TextBlob(t).sentiment.polarity < -0.2 else "Neutral")
    sentiment = df["Sentiment"].value_counts(normalize=True).mul(100).round(2).to_dict()
    brands = dict(Counter(re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', " ".join(df["Comment"]))).most_common(10))
    df["Engagement Quality"] = df["Comment"].apply(lambda c: "High Quality" if len(c) >= 40 else "Medium Quality" if len(c) >= 15 else "Low Quality")
    quality = df["Engagement Quality"].value_counts(normalize=True).mul(100).round(2).to_dict()
    return {
        "Sentiment Breakdown": sentiment,
        "Top Brand Mentions": brands,
        "Engagement Quality": quality
    }

# 5️⃣ Gemini API for idea generation
def generate_ideas(category, profile_data, max_retries=3):
    bio = profile_data.get("Biography", "")
    content_type = profile_data.get("Content Type", "")
    followers = profile_data.get("Followers", 0)
    prompt = (
        f"You are a creative assistant for House of X, a platform that empowers creators to launch D2C brands. "
        f"Given a creator with a predicted D2C category '{category}', biography '{bio}', "
        f"content type '{content_type}', and {followers:,} followers, suggest a D2C brand idea. "
        f"Return a JSON object with: 'brand_name' (a catchy name), 'description' (a 1-2 sentence brand description), "
        f"'product_ideas' (a list of 3 specific product names as strings, e.g., ['Product1', 'Product2', 'Product3']), "
        f"and 'estimated_revenue' (in $M, based on category and follower scale). "
        f"Ensure the brand and products align with House of X's focus on Gen Z and millennial audiences in Beauty, Fashion, Food, Fitness, or Lifestyle categories."
    )
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 300
        }
    }
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
                headers=headers,
                json=payload
            )
            if response.status_code == 200:
                result = response.json()
                content = result["candidates"][0]["content"]["parts"][0]["text"]
                # Clean JSON if wrapped in markdown
                content = content.strip('```json\n').strip('```')
                ideas = json.loads(content)
                # Handle product_ideas if it's a list of dictionaries
                product_ideas = ideas["product_ideas"]
                if product_ideas and isinstance(product_ideas[0], dict):
                    product_ideas = [item.get("name", item.get("product", str(item))) for item in product_ideas]
                return {
                    "brand_name": ideas["brand_name"],
                    "description": ideas["description"],
                    "product_ideas": product_ideas,
                    "estimated_revenue": ideas["estimated_revenue"]
                }
            elif response.status_code == 429:
                wait_time = 2 ** attempt
                print(f"Rate limit hit, retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                print(f"Gemini API error: {response.status_code} - {response.text}")
                break
        except Exception as e:
            print(f"Request error: {e}")
            if attempt == max_retries - 1:
                break
            time.sleep(2 ** attempt)
    print("Falling back to default response due to Gemini API failure")
    return {
        "brand_name": f"Fallback{category}Brand",
        "description": f"A {category} brand inspired by the creator’s persona",
        "product_ideas": ["Product1", "Product2", "Product3"],
        "estimated_revenue": "$20M"
    }

# 6️⃣ Load and train category prediction model
def train_category_model():
    data = pd.read_csv("creator_brands.csv")
    X_text = data["Description"]
    X_numeric = data[["Year", "Revenue"]].replace(r"[\$M]", "", regex=True).astype(float).fillna(0)
    y = data["Category"]
    tfidf = TfidfVectorizer(max_features=500, stop_words="english")
    X_text_tfidf = tfidf.fit_transform(X_text)
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X_numeric)
    X = np.hstack([X_text_tfidf.toarray(), X_numeric_scaled])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Model Accuracy: {accuracy:.2f}")
    return model, tfidf, scaler

# 7️⃣ Predict category and generate ideas
def predict_and_generate_ideas(creator_data, model, tfidf, scaler):
    text_features = [creator_data["Biography"] + " " + creator_data["Content Vibe"] + " " + creator_data["Content Type"]]
    text_tfidf = tfidf.transform(text_features)
    numeric_features = np.array([[
        2026,  # Future launch year
        creator_data["Followers"] / 1e6  # Proxy revenue potential
    ]])
    numeric_scaled = scaler.transform(numeric_features)
    combined_features = np.hstack([text_tfidf.toarray(), numeric_scaled])
    predicted_category = model.predict(combined_features)[0]
    confidence_scores = model.predict_proba(combined_features)[0]
    top_3_categories = pd.Series(confidence_scores, index=model.classes_).nlargest(3).to_dict()
    ideas = generate_ideas(predicted_category, creator_data)
    return {
        "Predicted Category": predicted_category,
        "Top-3 Category Probabilities": top_3_categories,
        "Brand Name": ideas["brand_name"],
        "Description": ideas["description"],
        "Product Ideas": ideas["product_ideas"],
        "Estimated Revenue": ideas["estimated_revenue"]
    }

# 8️⃣ Run full pipeline
def run_full_analysis_and_ideas(username, apify_token):
    print(f"🚀 Starting full analysis and D2C idea generation for @{username}")

    # Scrape and analyze Instagram data
    scrape_instagram_all(username, apify_token)
    profile_summary, followers = analyze_profile(f"{username}_profile.csv")
    post_summary = analyze_posts(f"{username}_posts.csv", followers)
    comment_summary = analyze_comments(f"{username}_comments.csv")

    # Prepare creator data for model
    creator_data = {
        "Username": profile_summary.loc["Username", "Value"],
        "Full Name": profile_summary.loc["Full Name", "Value"],
        "Biography": profile_summary.loc["Biography", "Value"],
        "Followers": followers,
        "Following": int(profile_summary.loc["Following", "Value"]),
        "Post Count": int(profile_summary.loc["Post Count", "Value"]),
        "Joined Recently": profile_summary.loc["Joined Recently", "Value"],
        "Engagement Potential": float(profile_summary.loc["Engagement Potential", "Value"]),
        "Creator Type": profile_summary.loc["Creator Type", "Value"],
        "Trust Score": int(profile_summary.loc["Trust Score", "Value"]),
        "Engagement Rate": float(post_summary["Engagement Rate"].strip("%")),
        "Content Vibe": post_summary["Content Vibe"],
        "Content Type": post_summary["Content Type"],
        "Best Post Time": post_summary["Best Post Time"],
        "Sentiment Breakdown": comment_summary["Sentiment Breakdown"],
        "Top Brand Mentions": comment_summary["Top Brand Mentions"],
        "Engagement Quality": comment_summary["Engagement Quality"]
    }

    # Train category prediction model
    model, tfidf, scaler = train_category_model()

    # Predict category and generate ideas
    ideas_output = predict_and_generate_ideas(creator_data, model, tfidf, scaler)

    # Display results
    print("\n📌 Profile Summary:\n")
    print(profile_summary.to_string())

    print("\n📸 Post Insights:\n")
    for k, v in post_summary.items():
        print(f"{k}: {v}")

    print("\n💬 Comment Insights:\n")
    for k, v in comment_summary.items():
        print(f"{k}: {v}")

    print("\n💡 D2C Business Ideas for House of X:\n")
    print(f"Predicted Category: {ideas_output['Predicted Category']}")
    print(f"Top-3 Category Probabilities: {ideas_output['Top-3 Category Probabilities']}")
    print(f"Brand Name: {ideas_output['Brand Name']}")
    print(f"Description: {ideas_output['Description']}")
    print(f"Product Ideas: {', '.join(ideas_output['Product Ideas'])}")
    print(f"Estimated Revenue: {ideas_output['Estimated Revenue']}")

    # Save ideas to CSV
    output_data = pd.DataFrame({
        "Creator Name": [creator_data["Full Name"]],
        "Brand Name": [ideas_output["Brand Name"]],
        "Category": [ideas_output["Predicted Category"]],
        "Description": [ideas_output["Description"]],
        "Product Ideas": ["\n".join(ideas_output['Product Ideas'])],
        "Year": [2026],
        "Estimated Revenue": [ideas_output["Estimated Revenue"]]
    })
    output_data.to_csv(f"{username}_d2c_ideas.csv", index=False)
    print(f"\n📄 D2C ideas saved to {username}_d2c_ideas.csv")

if __name__ == "__main__":
    apify_token = "apify_api_EtCTiSakLFyWlWYSjna8IX4yO6dEz73Kcsph"
    username = input("Enter Instagram username: ").strip()
    run_full_analysis_and_ideas(username, apify_token)