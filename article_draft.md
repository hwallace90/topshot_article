# A Deeper Look at OTM True Value

My name is Harrison, data scientist and designer of the OTM True Value model. In this article I want to give a more in depth breakdown of the True Value model: how it was designed, some of the key assumptions and risks, and give a look into model performance.

To attempt to build the most in-depth NBA Top Shot valuation tool available, we decided on building a [Gradient Boosting Machine](https://en.wikipedia.org/wiki/Gradient_boosting) model, usually referred to as a GBM. From my past experience, the other top contender would be some type of [linear regression model](https://en.wikipedia.org/wiki/Linear_regression).

## Why a GBM

GBMs bring many benefits in comparison to more traditional statistical modeling methods, such as linear models. First, in a linear model, the relationships between the predictors and the outcome variable have to be more explicitly defined. For example, if you were simply trying to predict sales price as a function of serial number without any transformations applied to the data, this would be assumped to be a linear straight line upward. We know the market does not behave that way. Bigger serial numbers have lower prices, and the increase in price as serial gets closer to 1 starts slow, but as we get closer and closer to 1, the price curve increases steeply, with very low serials usually worth huge multiples of the least valuable junk serial numbers. To model a more complex relationship between serial and price using a linear model, we would have to perform a data transformation to the serial variable, and there are many possibilites on how to transform it. 

Now, what happens when we have another variable to add into the model, like low ask. Well, lower low asks (say $3), have much steeper price curves than much higher low asks (say $100). There is an [interaction](https://en.wikipedia.org/wiki/Interaction_(statistics)) happening between serial and low ask. That has to be explicitly defined in the model. Now, let's say there's 10 more variables we would like to add to the model. Gets complex quick, right? In GBMs we don't have to explicitly define all these relationships between variables. The algorithm is able to predict highly complex non-linear relationships between all the variables and the outcome variable. This is a huge benefit! This gives the creator freedom to iterate and test models in a more efficient manner. Of course, there's no free lunch, and GBMs have plenty of other things to define and optimize, and tradeoffs to watch out for, but I'll leave anyone more interested to seek out one of the many GBM tutorials across the internet.

Second, and most importantly, GBMs are one of the best machine learning algorithms out there for social science prediction problems. They used in important industries like finance and banking, where I used to work, to make decisions totaling billions of dollars. They consistently crush in many types of [Kaggle competitions](https://www.kaggle.com/competitions), where data scientists compete against each other to build the best model for a particular problem. When data sets get big and problems get complex, GBMs can squeeze a lot of juice and make better predictions.

## What is a GBM?

To explain what a GBM is, first you need to understand a little bit about its simpler cousin, the [decision tree](https://en.wikipedia.org/wiki/Decision_tree). A decision tree essentially looks like a flow chart, where at each decision node, a variable is used to split the node into 2 different branches, and those 2 different branches each have a different predictions. A common tutorial dataset used is the passengers on the sinking of the Titanic. What if we want to predict who survived and died on the Titanic? Because of the limited lifeboat capacity, surviving the sinking of the Titanic was not an equal opportunity affair, as women and children got first priority. So how might this be represented by a decision tree? Here's one example.

![image](https://user-images.githubusercontent.com/10187424/159057374-f21192da-b783-4933-bae2-b8bd829a41b6.png)
([source](https://commons.wikimedia.org/wiki/File:CART_tree_titanic_survivors.png), credit to Stephen Milborrow)

If the sex is female, predict that they survived, as 73% of females in the dataset survived. If they're male, and age 10 or older, predict death, as only 17% survived. If they're male, 9 or younger, and had 3 or more siblings (sibsp in the image), predict death, as only 5% survived, otherwise predict survive, as 89% survived.

A single decision tree can be great for basic problems, they're simple and highly interpretable. For complex problems, a GBM is a high performing extension of the decision tree concept. A GBM is an [ensemble](https://en.wikipedia.org/wiki/Ensemble_learning) of many decision trees fit (solved with math using an algorithm) iteratively. A tree will be fit, math will be performed to see how badly each prediction was missed (called a "pseudo residual"), and then another tree will be fit on the new pseudo residuals, in effect trying to correct for the areas where the last tree missed. This can be performed hundreds or thousands of times, reducing the error of our predictions, until the performance on a holdout dataset no longer improves.

## What's in the Model?

The model is made up of 18 predictor variables. They can be broadly classified in 3 categories:
* Static characteristics about the moment. Some of them are basic metadata, some of them are new derived variables.
* Constantly changing information about how the moment is performing on the marketplace.
* Macro level variables describing the overall state of the market.

The first two are fairly straight forward, but the third deserves some comment. The purpose is two-fold. First, they enable us to gain better information from times where the market looked far different than it currently does. For example, prices were much higher March through May 2021 than they are anytime after that. There's still millions of transaction there, and they have valuable information. Market-level variables make it easier to still extract some value from that data, even if prices are unrecognizable to right now. Second, the hope is that as market conditions continue to shift in the future, that the model is better able to quickly respond to changing conditions, and make better predictions about a moment's value. 

## How Well Does It Perform?

So how well does thew model perform? There's a few different ways to evaluate the model. The first test is understanding if the model is accurately describing the past, using observations that were held out and the model never directly saw. We always keep a 10-20% holdout sample to both understand when the model begins to get too complex (hyperlink: https://en.wikipedia.org/wiki/Overfitting) and more trees are no longer helping predictions, and also to see how well the model fits on these held out observations. The model fits extremely well on this hold out dataset. It's able to predict well across all the important dimensions, like serial and low ask.

(insert $ plots for serial and low ask here on hold out validation data, need to run this weeks model on my comp and do some edits to make the plots more professional and not give away how we do jersey numbers, but current examples I can just show)

![image](https://user-images.githubusercontent.com/10187424/159057829-b7a65615-3dfd-400e-a231-6473d44cc70d.png)

![image](https://user-images.githubusercontent.com/10187424/159057877-827f3945-d755-4ff9-9614-a3d1f5e0f13d.png)

The second test is to see how well the model predicts the future. For any well constructed social science machine learning model, the performance on future, unseen observations is always going to be worse than held out observations that occurred in the time period you built the model on. The reason for this is simple, in a market driven by consumer sentiments driving outcomes, consumer sentiments are liable to shift over time. We see the same thing occuring in the NBA Top Shot markets. Over time, price curves are flattening. Everything else equal, the most valuable serials are selling for lower multiples of low ask. This leads to a bit of overprediction as the model adjusts. 

Models are not magic, in the end they are essentially mathematical averages of what's happened in the past. You can influence what the past looks like through data transformations, loss functions, observation weighting, and other tools at your disposal, but there are always tradeoffs.  The task then becomes trying to make the model adjust as quickly as possible, while still balancing the valuable information that came in the past. This also brings up an important point on how the model should be used. It should not be used as a indication of where things will move in the future, instead it is simply a representation of what we think the moment should be worth, based on information we know about the moment, market, and past transactions that occurred under those circumstances.

With all that in mind, we are very pleased with the performance of the model for predicting the future, it slopes over the most important variables well, despite some overprediction in the most valuable serials.

(insert same plots as above for 2 week test holdout, with the same caveats)

![image](https://user-images.githubusercontent.com/10187424/159057982-1ce4a697-4afa-43ec-8655-741d64ee2d7c.png)

![image](https://user-images.githubusercontent.com/10187424/159058056-51fc2ce2-5d57-4838-b417-70e71b1a71dc.png)

Finally, does the model find the most valuable players and moments and differentiate them from the rest of the crowd? We are pleased in the model's ability to discriminate what are the players and moments that people are willing to pay above and beyond for.

(going to try and create a new plot to show this, have an idea in my head for it)

## A Couple Key Assumptions in the Model

Quickly, I wanted to comment on a few key assumptions in the model that affect its use.

* As you go from serial #1 to higher serial numbers, prices decrease non-linearly and monotonically, with exceptions for serials matching a player's jersey number, and the final serial of the mint.

You'll never see serial number 22 be suddenly worth less than serial number 23, assuming no jersey number match.

* Prices have a direct, unbreakable relationship to the listed low ask.

Currently, the model is based directly around low ask, the multiples of low ask that every moment sells for. As a result, if the current low ask is completely ridiculous, the predictions are going to be equally ridiculous. This might be a concern for some very low volume, low mint count moments. 

* The model does not explicitely account for the various types of challenges that occur that frequenty make more specific moments much more valuable.

There are other ways the model might pick up on this, but we do not speficially tell the model that a moment is involved in a challenge.

Finally, one key tradeoff in GBMs are I want to point out. 

* GBMs are unable to extrapolate outside of the bounds of data we've seen in the past

All models are weak for [extrapolation](https://en.wikipedia.org/wiki/Extrapolation). It's a tough problem to solve, as trends you see in past data are never guranteed to continue when moving into absolutely new territory. GBMs have a specific weakness here, because they do not even attempt to extrapolate, they will assume that it will be equal to the previous highest or lowest bounds seen previously. This is an inherent feature of using decision trees, as all they do at each decision node is split data at a specific point and then assign predictions on each side. What does that mean for the purpose of the OTM True Value model? It means when things happen that we've never seen before, for example when 60k mint count cards came out for the first time, and $2 low asks were seen for the first time for any extended period of time, both of those are outside of the bounds of past data, and there will be an adjustment period.
