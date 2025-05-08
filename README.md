# NLP-QoS
Submitted by: Tanmaya Sarkar
Reg no: RA2212704010032


This project explores how user sentiment and engagement data can be used to dynamically optimize Quality of Service (QoS) for live game streaming platforms such as Twitch or YouTube Gaming.

We use BERT-based sentiment analysis and a Deep Neural Network (DNN) classifier to predict viewer satisfaction from real YouTube comment data. The final output can guide real-time decisions like adjusting bitrate, buffering strategies, or notifying streamers.

The dataset used is UScomments.csv, which includes real YouTube comment text data. We use these comments as proxies for user feedback in a streaming environment.

| Component                       | Tool/Library                      | Purpose                                                        |
| ------------------------------- | --------------------------------- | -------------------------------------------------------------- |
| **Sentiment Analysis**          | `transformers` (HuggingFace BERT) | Classify user comments as `POSITIVE` or `NEGATIVE`             |
| **Satisfaction Classification** | `TensorFlow` + `Keras` DNN        | Predict user satisfaction from features like comment length    |
| **Data Handling**               | `pandas`, `numpy`                 | Data cleaning and preprocessing                                |
| **Visualization**               | `matplotlib`, `sklearn`           | Training performance and confusion matrix                      |
| **QoS Decision Modeling**       | Rule-based simulation             | Suggests stream optimization based on satisfaction predictions |


1. Sentiment Extraction with BERT
We use the HuggingFace pipeline to apply a pre-trained BERT model (distilbert-base-uncased-finetuned-sst-2-english) to each comment. This gives us a sentiment label (POSITIVE or NEGATIVE), simulating viewer feedback in real time.

2. DNN-Based Satisfaction Classification
We train a lightweight deep neural network on features derived from the comments (e.g., comment length). The DNN learns to predict a satisfaction score, which abstracts viewer engagement.

3. QoS Simulation
Based on the satisfaction prediction, we simulate real-time Quality of Service actions. For example:

NEGATIVE → OPTIMIZE_STREAM (reduce bitrate, increase buffer)

POSITIVE → MAINTAIN_STREAM (normal streaming)

This models how platforms might dynamically adapt to viewer behavior.

Output Visualizations:
Training Accuracy and Loss Curves over 10 epochs

Confusion Matrix for satisfaction prediction

Optional: Add ROC/PR curves if needed






