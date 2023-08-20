# Naive_Bayes

## Problem Statement

Prepare a classification model using Naive Bayes for salary data

# ➳ Naive_Bayes

A Naive Bayes classifier assumes that the effect of a particular feature in a class is independent of other features and is based on Bayes’ theorem. Bayes’ theorem is a mathematical equation used in probability and statistics to calculate conditional probability. In other words, you can use this theorem to calculate the probability of an event with functions like the Gaussian Probability Density function based on its association with another event.

The simple formula of the Bayes theorem is:

![Implementing Naive Bayes Classification](https://github.com/yagniksorathiya/Naive_Bayes/assets/129974278/b5a4a4cf-209b-45fe-acc6-a72e0849113f)

Where P(A) and P(B) are two independent events and (B) is not equal to zero.

+ P(A | B): is the conditional probability of an event A occurring given that B is true.

+ P( B | A): is the conditional probability of an event B occurring given that A is true.

+ P(A) and P(B):  are the probabilities of A and B occurring independently of one another (the marginal probability).

## ↳ Why is it called Naive Bayes?

The Naive Bayes algorithm is comprised of two words Naive and Bayes, Which can be described as:

+ **Naive:** It is called Naïve because it assumes that the occurrence of a certain feature is independent of the occurrence of other features. Such as if the fruit is identified on the bases of color, shape, and taste, then red, spherical, and sweet fruit is recognized as an apple. Hence each feature individually contributes to identify that it is an apple without depending on each other.

+ **Bayes:** It is called Bayes because it depends on the principle of Bayes' Theorem.


## ↳ What is Naive Bayes Classification?

The Naive Bayes classification algorithm is a probabilistic classifier and belongs to Supervised Learning. It is based on probability models that incorporate strong independence assumptions. The independence assumptions of the Naive Bayes models often do not impact reality. Therefore they are considered naive.

Another assumption made by the Naive Bayes classifier is that all the predictors have an equal effect on the outcome. The Naive Bayes classification has the following different types:

+ The **Multinomial Naive Bayes** method is a common Bayesian learning approach in natural language processing. Using the Bayes theorem, the program estimates the tag of a text, such as an email or a newspaper piece. It assesses the likelihood of each tag of multinomial Naive Bayes for a given sample and returns the tag with the highest possibility.

+ The **Bernoulli Naive Bayes** is a part of the family of Naive Bayes. It only takes binary values. Multiple features may exist, but each is assumed to be a binary-valued (Bernoulli, boolean) variable. Therefore, this class requires samples to be represented as binary-valued feature vectors.

+ The **Gaussian Naive Bayes** is a variant of Naive Bayes that follows Gaussian normal distribution and supports continuous data. To build a simple model using Gaussian Naive Bayes, we assume the data is characterized by a Gaussian distribution with no covariance (independent dimensions) between the parameters. This model may fit by applying the Bayes theorem to calculate the mean and standard deviation of the points within each label.

![Naive Bayes Classification](https://github.com/yagniksorathiya/Naive_Bayes/assets/129974278/f51dc1aa-8e04-461b-b0ed-7b8450ac74c2)

The Naive Bayes classifier makes two fundamental assumptions on the observations.

+ The **target classes** are independent of each other. Consider a rainy day with strong winds and high humidity. A Naive classifier would treat these two features, wind and humidity, as independent parameters. That is to say, each feature would impose its probabilities on the outcome, such as rain in this case.

+ Prior probabilities for the target classes are **equal**. That is, before calculating the posterior probability of each class, the classifier will assign each target class the same prior probability.

## ↳ When to use the Naive Bayes Classifier?

**Naive Bayes classifiers tend to perform especially well in any of the following situations:**
+ When the naive assumptions match the data.

+ For very well-separated categories, when model complexity is less important.

+ And for very high-dimensional data, when model complexity is again less important.

The last two points appear unrelated, but they are related to each other. As a dataset’s dimension grows, it becomes less likely that any two points will be discovered nearby. This means that clusters in high dimensions tend to be more separated than clusters in low dimensions,

**The Naive Bayes classifier has the following advantages.**

+ Naive Bayes classification is extremely fast for training and prediction especially using logistic regression.

+ It provides straightforward probabilistic prediction.

+ Naive Bayes has a very low computation cost.

+ It can efficiently work on a large dataset.

+ It performs well in the case of discrete response variables compared to continuous variables.

+ It can be used with multiple class prediction problems.

+ It also performs well in the case of text analytics problems.

+ When the assumption of independence holds, a Naive Bayes classifier performs better than other models like **Logistic Regression**.

## ↳ Real-life applications using Naive Bayes Classification

The Naive Bayes algorithm offers plenty of advantages to its users. That’s why it has a lot of applications in various industries, including Health, Technology, Environment, etc. Here are a few of the applications of the Naive Bayes classification:

+ It is used in **text classification**. For example, News on the web is rapidly growing, and each news site has its different layout and categorization for grouping news. To achieve better classification results, we apply the naive Bayes classifier to classify news content based on news code.

+ Another application of Naive Bayes classification is **Spam filtering**. It typically uses a bag of words features to identify spam e-mails. Naive Bayes classifiers work by correlating the use of tokens (typically words or sometimes other things), with spam and non-spam e-mails and then using Bayes’ theorem to calculate the probability that an email is or is not spam.

+ One of the advantages of the Naive Bayes Classifier is that it takes all the available information or data point to explain the decision. When dealing with medical data, the Naïve Bayes classifier considers evidence from many attributes to make the final prediction of the class label and provides transparent explanations of its decisions. That is why it has many applications in the health sector as well.

+ Weather prediction has been a challenging problem in the meteorological department for years. Even after technological and scientific advancements, the accuracy in weather prediction has never been sufficient. However, Naive Bayes classifiers give high-accuracy results when predicting weather conditions.

+ Its assumption of feature independence, and its effectiveness in solving multi-class problems, make it perfect for performing **Sentiment Analysis**. Sentiment Analysis identifies a target group’s positive or negative sentiments.

+ Collaborative Filtering and the Naive Bayes algorithm work together to build **recommendation systems**. These systems use data mining and Machine Learning to predict whether the user would like a particular resource. 

    
