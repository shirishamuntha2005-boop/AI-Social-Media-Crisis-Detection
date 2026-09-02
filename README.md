# 🚨 AI-Powered Social Media Crisis Detection & Early Warning System

## Real-Time Crisis Monitoring, Severity Analysis, Risk Assessment & Early Warning Using NLP and Transformer Models

An AI-powered system for detecting and analyzing crisis-related information from social media posts using **Natural Language Processing (NLP)** and **Transformer-based Deep Learning models**.

The system classifies social media posts as **Informative** or **Not Informative**, estimates crisis severity, generates priority alerts, groups related crisis posts into events, calculates crisis risk, analyzes trends, and provides an interactive **real-time Streamlit dashboard** for early warning and decision support.

---

## 📌 Project Overview

During natural disasters and emergency situations, social media platforms generate large volumes of posts containing potentially valuable crisis information.

However, manually monitoring thousands of posts is difficult because:

- Social media data is generated continuously.
- Crisis-related information is mixed with irrelevant content.
- Emergency information can spread very quickly.
- Manual analysis is time-consuming.
- Different posts may describe the same crisis event.
- Emergency responders need timely alerts and risk information.

This project addresses these challenges by developing an **AI-powered social media crisis detection and early warning system**.

The proposed system combines:

- Natural Language Processing
- Machine Learning
- Transformer-based NLP
- Real-time processing
- Crisis severity detection
- Alert prioritization
- Crisis event grouping
- Risk assessment
- Trend analysis
- Early warning visualization

---

# 🎯 Objectives

The main objectives of this project are:

1. Detect crisis-related information from social media posts.
2. Classify posts as informative or not informative.
3. Use NLP techniques to clean and preprocess social media text.
4. Compare traditional machine learning models with transformer models.
5. Implement a RoBERTa-based crisis classification system.
6. Estimate the severity of detected crisis information.
7. Generate automated crisis alerts.
8. Group related posts into crisis events.
9. Calculate an overall crisis risk score.
10. Analyze crisis trends over time.
11. Provide early warning information through an interactive dashboard.
12. Allow users to test individual social media posts using the trained model.

---

# 🧠 System Architecture

```text
                 SOCIAL MEDIA POSTS
                         │
                         ▼
              ┌─────────────────────┐
              │ Data Collection     │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Text Cleaning &    │
              │ NLP Preprocessing  │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ RoBERTa Crisis     │
              │ Classification     │
              └──────────┬──────────┘
                         │
                ┌────────┴────────┐
                ▼                 ▼
          Informative       Not Informative
                │
                ▼
       ┌─────────────────────┐
       │ Severity Detection  │
       └──────────┬──────────┘
                  │
                  ▼
       ┌─────────────────────┐
       │ Alert Generation    │
       └──────────┬──────────┘
                  │
                  ▼
       ┌─────────────────────┐
       │ Crisis Event        │
       │ Grouping            │
       └──────────┬──────────┘
                  │
                  ▼
       ┌─────────────────────┐
       │ Risk Assessment     │
       └──────────┬──────────┘
                  │
                  ▼
       ┌─────────────────────┐
       │ Trend Analysis      │
       └──────────┬──────────┘
                  │
                  ▼
       ┌─────────────────────┐
       │ Early Warning       │
       │ Dashboard           │
       └─────────────────────┘

