# Tweet Sentiment Classifier <img src="https://techcrunch.com/wp-content/uploads/2019/07/twitter-logo-sketch-wide.png?w=1390&crop=1" style="zoom: 33%;" />

![](https://img.shields.io/badge/Code-Python-blue?style=plastic&logo=python&logoColor=yellow) ![](https://img.shields.io/badge/Framework-Pytorch-red?style=plastic&logo=pytorch&logoColor=red) ![](https://img.shields.io/badge/version-v.1.0-green?style=plastic) 



### Description

This project is about the application of RNN with LSTM to classify tweets into Positive and Negative Sentiments.

The objective is that a user will enter a tweet and the model will predict whether the tweet is of positive or negative sentiment.



### Dataset Source

The dataset that was used for the model training can be found on this [Kaggle](https://www.kaggle.com/kazanova/sentiment140) link. 

The dataset is composed of 1.6 million tweets which are extracted through twitter api. Being so vast, it was impossible for me to train a model on the complete dataset, thereby I randomly sample 100,000 tweets out of the 1.6 million tweets and trained my model on those samples.



### Model Metrics

My model’s status after training on a small subset of the complete dataset are as follows:

```json
{
	"validation-loss" : 0.42073412984609604,
	"validation-accuracy" : 0.7948208512931034,
	"validation-f1_score" : 0.7804794881272281
}
```



### Try the API

You can test my model’s prediction first-hand, using this cURL in your Postman or Insomnia.

```
curl --request GET \
  --url 'https://twittersentimentanalysis13.herokuapp.com/predict?text=Forever%20fighting%20%23insomnia.%20HALP!%20%0A%0AHow%20do%20you%20force%20yourself%20to%20sleep%3F%0A'
```



### Acknowledgements:

1. https://www.kaggle.com/michawilkosz/twitter-sentiment-analysis-using-tensorflow
2. https://github.com/lucifermorningstar1305/news_classifier

