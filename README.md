\# 🚨 AI Social Media Crisis Detection \& Early Warning System



An AI-powered Natural Language Processing (NLP) system that analyzes social media text and identifies whether a post contains informative crisis-related information.



\## 📌 Project Overview



During natural disasters and emergency situations, social media platforms contain large amounts of real-time information.



However, not every social media post provides useful crisis-related information.



This project uses Natural Language Processing and Machine Learning to automatically classify social media posts into:



\- INFORMATIVE

\- NOT INFORMATIVE



The project includes traditional Machine Learning models as well as a fine-tuned DistilBERT model and a Streamlit web application for real-time prediction.



\## 🎯 Objectives



\- Detect informative crisis-related social media posts.

\- Classify posts automatically using NLP.

\- Compare traditional Machine Learning approaches with Transformer-based models.

\- Provide a simple web interface for users.

\- Display prediction results and confidence scores.

\- Build a foundation for an AI-powered crisis early warning system.



\## 📊 Dataset



The project uses the CrisisMMD v2.0 dataset.



CrisisMMD is a multimodal disaster-related social media dataset containing information from crisis events.



The project focuses on the textual information for NLP-based crisis classification.



\### Dataset Processing



The data processing pipeline includes:



1\. Dataset collection

2\. Dataset combination

3\. Data cleaning

4\. Text preprocessing

5\. Exploratory Data Analysis

6\. TF-IDF feature extraction

7\. Machine Learning model training

8\. DistilBERT fine-tuning

9\. Model evaluation

10\. Streamlit deployment



\## 🤖 Models Used



\### Traditional Machine Learning



The project experimented with:



\- Logistic Regression

\- Random Forest

\- TF-IDF Word Unigrams

\- TF-IDF Word Bigrams

\- TF-IDF Word Trigrams

\- TF-IDF Character Features



\### Transformer Model



A DistilBERT-based text classification model was fine-tuned for crisis-related text classification.



The trained model is stored locally in:



`data/processed/distilbert\_crisis\_model`



\## 🧠 DistilBERT Model Files



The trained model directory contains:



\- `config.json`

\- `model.safetensors`

\- `tokenizer.json`

\- `tokenizer\_config.json`

\- `distilbert\_metrics.json`

\- `training\_history.json`



\## 📈 Model Results



The traditional Machine Learning experiments produced the following results:



| Model | Accuracy | F1 Score |

|---|---:|---:|

| Logistic Regression - Word Bigram | 78.83% | 79.16% |

| Logistic Regression - Word Trigram | 79.17% | 79.41% |

| Logistic Regression - Character TF-IDF | 78.99% | 79.52% |

| Random Forest - 200 Trees | 79.45% | 79.45% |

| Random Forest - 300 Trees | 79.51% | 79.51% |

| Random Forest - MaxFeaturesLog2 | 80.20% | 80.15% |

| Random Forest - No Balance | 80.23% | 78.67% |



\### Example Prediction



Example input:



"Flood water has entered several houses and people need immediate rescue."



Example output from the Streamlit application:



\- Prediction: \*\*INFORMATIVE\*\*

\- Class ID: \*\*1\*\*

\- Confidence: \*\*99.69%\*\*



> Note: Prediction confidence is specific to an individual input and should not be interpreted as the overall model accuracy.



\## 🖥️ Streamlit Application



The project includes a Streamlit web application.



The application allows users to:



1\. Enter social media text.

2\. Select example crisis-related posts.

3\. Analyze the text.

4\. Receive a classification.

5\. View the confidence score.

6\. View prediction details.



\## 🛠️ Technologies Used



\### Programming Language



\- Python



\### Machine Learning \& NLP



\- PyTorch

\- Hugging Face Transformers

\- DistilBERT

\- Scikit-learn

\- TF-IDF

\- Pandas

\- NumPy



\### Visualization \& Application



\- Streamlit

\- Matplotlib

\- Seaborn

\- Plotly



\### Development Tools



\- VS Code

\- Jupyter Notebook

\- Git

\- GitHub



\## 📁 Project Structure



```text

AI-Social-Media-Crisis-Detection/

│

├── app.py

├── README.md

├── .gitignore

│

├── data/

│   ├── raw/

│   └── processed/

│       ├── crisis\_mmd\_master.csv

│       ├── crisis\_mmd\_cleaned.csv

│       ├── crisis\_mmd\_nlp.csv

│       └── distilbert\_crisis\_model/

│

├── notebooks/

│

├── outputs/

│

├── src/

│   ├── combine\_datasets.py

│   ├── clean\_dataset.py

│   ├── eda.py

│   ├── nlp\_preprocessing.py

│   ├── tfidf\_features.py

│   └── train\_distilbert.py

│

└── venv/

