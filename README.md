# 📊 House of X Influencer D2C Brand Recommendation Dashboard

An end-to-end data analysis project and Streamlit dashboard that scrapes Instagram creator data using Apify, analyzes profile, post, and comment insights, and generates personalized D2C brand ideas for influencers using Gemini API and ML models.

---

## ✨ Project Highlights
- 🔍 **Instagram Scraping**: Uses Apify to extract influencer profile, post, and comment data.
- 📈 **Data Analysis**: Analyzes engagement rate, tone, content type, sentiment, and brand mentions.
- 🧠 **ML Category Prediction**: Predicts D2C brand category using RandomForest on brand dataset.
- 🤖 **Gemini-Powered Idea Generator**: Creates creative D2C brand names, products, and revenue potential.
- 📊 **Streamlit Dashboard**: Beautiful UI with charts, profile display, and business idea report.

---

## 🚀 Demo  Video 

> 📎 [Check out the full demo and explanation on LinkedIn](https://www.linkedin.com/posts/vansh-suneja-32b0042a9_houseofx-creatoreconomy-opentowork-activity-7343579337641639936-Y_qT?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEo3rNMBHJiGSY1xSNOdeEbxgGX_uQoA6uA)

---

## 🛠 Tech Stack
- Python
- Streamlit
- Apify
- Google Gemini API
- Scikit-learn, Pandas, Seaborn
- Transformers (Zero-shot classification)

---

## 📂 File Structure
```
├── streamlit_app.py              # Final dashboard UI script
├── instagram_analysis.py        # Data collection and ML analysis pipeline
├── creator_brands.csv           # Training data for category prediction
├── .env                         # Contains GEMINI_API_KEY
├── requirements.txt             # All dependencies
├── /images                      # Screenshots for README
```

---

## 💻 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/houseofx-instagram-analysis
cd houseofx-instagram-analysis
```

### 2. Set up environment
```bash
pip install -r requirements.txt
```

### 3. Add your credentials
Create a `.env` file:
```bash
GEMINI_API_KEY=your_gemini_api_key
```

### 4. Run Streamlit app
```bash
streamlit run streamlit_app.py
```


---

## 🤝 Collaboration & Opportunities
I'm actively looking for internship/full-time roles in data, product, or analytics. If you're a founder, hiring manager, or just someone who loves solving problems using data—let's connect!

> 📩 Feel free to reach out on [LinkedIn](https://www.linkedin.com/in/vansh-suneja-32b0042a9/) or raise issues/PRs on this repo!

---

## 📜 License
This project is open-source and free to use for educational purposes.

---

## ✅ TODO
- [ ] Add Gradio alternative UI
- [ ] Deploy on Hugging Face Spaces
- [ ] Add support for multiple creators
- [ ] Improve visualizations with Plotly
