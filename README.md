# Movie Recommendation System Using Collaborative Filtering

## Project Overview
Modern streaming platforms face a major personalization challenge: users are overwhelmed by choice. When viewers spend more time scrolling than watching, engagement drops — and churn increases.

This project builds a movie recommendation system using the MovieLens (ml-latest-small) dataset to predict what a user is likely to enjoy next, based purely on past rating behavior. Rather than relying on movie metadata or genres alone, the system learns patterns in user preferences through Collaborative Filtering, mimicking how real-world platforms such as Netflix and Amazon personalize content.

The final model delivers Top-5 personalized movie recommendations for any existing user and demonstrates a complete end-to-end machine learning workflow: from data exploration and modeling to evaluation and deployment-ready functions.

---

## Business Problem
Users are abandoning the streaming platform because:
- Too many options cause decision fatigue
- Recommendations feel generic and unhelpful
- Users spend excessive time browsing instead of watching

**Business Objective:**  
Reduce scrolling time and increase user engagement by delivering accurate, personalized movie recommendations.

---

## Project Goal
Build a recommendation system that:
- Predicts user ratings for unseen movies
- Recommends the Top 5 most relevant movies
- Achieves a low prediction error (RMSE < 0.90)
- Is scalable and interpretable as a proof of concept

---

## Dataset
Source: MovieLens (Small Dataset) – GroupLens Research Lab  
https://grouplens.org/datasets/movielens/

### Files Used
- `movies.csv`: Movie IDs, titles, and genres  
- `ratings.csv`: User IDs, movie IDs, and explicit ratings  

### Dataset Characteristics
- 610 unique users  
- 9,742 movies  
- 100,836 ratings  
- Highly **sparse user–item matrix** (most users rate only a small fraction of movies)

---

## Tools & Technologies
- Python
- pandas, numpy – data handling
- matplotlib, seaborn – visualization
- scikit-surprise – recommender system algorithms
- GridSearchCV (Surprise) – hyperparameter tuning

---

## Methodology

### 1. Data Preparation & EDA
- Loaded and merged `movies` and `ratings` datasets
- Removed unnecessary columns (timestamps)
- Verified no missing or duplicate values
- Analyzed:
  - Rating distribution (user positivity bias)
  - User activity levels
  - Movie popularity skew

**Key Insight:**  
The dataset is sparse and long-tailed — a small number of movies receive most ratings, while many movies are rated very few times. This validates the use of collaborative filtering.

---

### 2. Baseline Model: KNNBaseline
A user-based collaborative filtering approach that:
- Measures similarity using **Pearson correlation**
- Accounts for:
  - User bias (strict vs generous raters)
  - Item bias (universally liked or disliked movies)

**Performance:**
- RMSE ≈ 0.88
- Strong baseline that confirms collaborative filtering viability

---

### 3. Matrix Factorization with SVD
Implemented Singular Value Decomposition (SVD) to:
- Learn latent factors representing hidden movie preferences
- Decompose the ratings matrix into user and item embeddings
- Scale better than neighborhood-based methods

**Performance:**
- RMSE ≈ 0.87
- Outperformed KNNBaseline

---

### 4. Hyperparameter Tuning
Used `GridSearchCV` to optimize:
- `n_factors` – number of latent features
- `reg_all` – regularization strength
- `lr_all` – learning rate

**Best Parameters:**
- `n_factors = 100`
- `reg_all = 0.1`

**Final Model Performance:**
- **RMSE ≈ 0.868**
- MAE minimized as well

This met and exceeded the project’s target accuracy.

---

### 5. Recommendation Deployment
Implemented production-style functions to:
- Filter out movies already watched by a user
- Predict ratings for all unseen movies
- Return the Top 5 recommendations
- Provide genre-aware discovery suggestions
- Handle niche users with limited rating history

These functions directly translate the trained model into a usable recommendation engine.

---

## Evaluation Metrics
- RMSE (Root Mean Squared Error) — primary metric  
- MAE (Mean Absolute Error) — supporting metric  

**Why RMSE?**
- Ratings are continuous and ordinal
- RMSE penalizes large prediction errors more heavily
- Well-suited for recommender systems with explicit feedback

---

## Results Summary

| Model | RMSE |
|------|------|
| KNNBaseline | ~0.88 |
| SVD (Default) | ~0.87 |
| SVD (Tuned) | ~0.868 |

---

## Key Insights
- Matrix factorization consistently outperforms neighborhood-based models
- Accounting for user and item bias significantly improves prediction quality
- Sparse datasets are well-handled by latent factor models
- Hyperparameter tuning provides measurable performance gains
- Recommendation logic must filter watched content to remain useful

---

## Limitations
- Cold-start users (new users with no ratings) are not handled optimally
- Rankings are evaluated using regression metrics rather than ranking-based metrics
- Content metadata is not directly incorporated into predictions

---

## Future Improvements
- Add content-based filtering to address cold-start problems
- Implement SVD++ using implicit feedback
- Introduce ranking metrics such as Precision@K and Recall@K
- Deploy the model via an API or interactive web interface

---

## Conclusion
This project demonstrates a complete collaborative filtering recommendation system built using industry-relevant techniques and evaluation strategies. Through systematic experimentation, tuning, and interpretation, an SVD-based model was developed that delivers accurate and personalized movie recommendations.

The final system directly addresses the business problem of user disengagement and showcases a strong practical understanding of recommendation systems, model evaluation, and deployment-ready machine learning workflows.