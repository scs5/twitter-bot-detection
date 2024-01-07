# Twitter Bot Detection

### Milestones
- [X] **Clean training data**
  - Handle NaN values, encode categorical features, and remove useless features (e.g. account ID)
- [X] **Engineer features**
  - Extract additional textual features as described in [1], including
    - Character length of bio
    - Word length of bio
    - Character length of screen name
    - POS frequency in bio
    - Sentiment score of bio
    - Number of emojis in bio
- [X] **Generate baseline results**
  - Train and evaluate a default Random Forest classifier
  - Accuracy ~ 86.6%
- [X] **Tune models**
  - Experiment with different models (e.g. XGBoost) + hyperparameter tuning
  - Accuracy ~ 89.7%
- [ ] **Create twitter profile scraper**
  - Due to the high cost of the Twitter API, a custom profile scraper will be needed
- [ ] **Create web app for predictions**
  - Build a web app which generates the probability an account is a bot given the username

### Training Data
[Kaggle Dataset](https://www.kaggle.com/code/davidmartngutirrez/bots-accounts-eda/input?select=twitter_human_bots_dataset.csv)

### Citations
[1] Varol, Onur, et al. "[Online human-bot interactions: Detection, estimation, and characterization.](https://ojs.aaai.org/index.php/ICWSM/article/view/14871)" Proceedings of the international AAAI conference on web and social media. Vol. 11. No. 1. 2017.

[2] Yang, Kai-Cheng, Emilio Ferrara, and Filippo Menczer. "[Botometer 101: Social bot practicum for computational social scientists.](https://arxiv.org/abs/2201.01608)" Journal of Computational Social Science 5.2 (2022): 1511-1528.
