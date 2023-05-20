# ML Introduction
                                     Machine Learning

Machine learning is a branch of computer science and artificial intelligence that deals with data, various machine learning models and algorithms that imitate how humans learn. It is way to make our model learn to observe patterns on it’s own with some given set of input and output. 
Machine learning can be divided into two categories:
1.	Supervised machine learning and
2.	Unsupervised machine learning
Lets talk about supervised machine leaning and then how it is used to train a model.
Supervised machine learning:  in this type of ML we provide the model or train the model with some input(X) as well as the known output (y). Then we will train our model on the basis of these inputs and outputs and provide them with new set of training data and try to get the predicted output. 
To understand how this entire system works. We need to dive deeper into the important components of machine learning. For affluence let’s divide the components into parts and go through them one by one:
1.	Data sets
2.	Learning algorithm (linear regression, logistic regression etc)
3.	To use ways to get more accurate predicted outputs that match without original output. 
4.	To add all the above steps and get our model ready.
Data sets
This is a collection of various types of data stored in digital format. This is the key component of machine learning project which acts as an example to teach machine learning algorithms how to make predictions. So before jumping into learning algorithms we must always collect the data related to the project we are working on.
Learning algorithms
It is a method by which the AI system conducts its task, generally predicting output values from given input data. There are different algorithms we use for different types of problems (regression and classification). We just need to know how our output types are and with that we can make sure which of the algorithms will do well.
Now after selecting an algorithm our next priority should be how to make it predict more accurate and faster output. 
This is where cost function comes into play. We don’t want our predictions to be very much different from our desired output. And that is why we calculate the cost function which tells the difference between our actual output and predicted output. We need to make sure that the cost function is always less because a good model should not have much difference between its actual output and predicted output.
The cost function use mean square error to calculate the difference between the original output and the predicted output. It is the squared difference between the true value and the prediction. 
There are two parameters that control the value of cost function: w and b
We need to find the specific value for w and b so that our regression line fits perfect to all the data. And to minimize these parameters J(w,b) we’ll Gradient Descent. 
What Gradient Descent does it will start with some parameter w and b and keep changing it till the updated values don’t change anymore or it reaches the local minima(a point where function value is smaller than at nearby points).
There many types of gradient descent for various problems: 
1.	Batch gradient descent: all training data is taken into consideration 
2.	Stochastic gradient descent: single random training data is taken into consideration along with its derivative. 
3.	Mini-batch gradient descent: a fixed batch of fixed number of training data which is less than the actual data set is taken into consideration. 
Now after discussing the key points of optimizing a model we need to trace our steps back and get to train our model. 
As we might have huge data sets. We will use one half to train our model and the other half to test the model. 
For selecting the algorithms we can take two approach based on our output (for most of the time) :-
1.	If we have to predict the values of a variable are dependent on the values of other variable. Or if our output is a numerical variable(ex :- predicting house prices , the weight of a person related to their height etc)
2.	If it is a classification problem. Or if we expect binary output labels.(ex :- if the tumor is malignant or benign ) 
For the first type of problem we use linear regression. For linear regression, f  w,b(X) = wx+b this function is making predictions for value of y using a streamline function of x. This will fit a line that tries to touch most of the data points. And if we extend this line and add a new data point we can get make predictions of that new data point. 
After choosing the linear regression model we calculate the cost function and gradient descent respectively and try to optimize the model.
Now we cannot provide the raw data and features to the model. Cause if we do so the performance of the model will reduce or perhaps give wrong prediction. For that we need to do some altering and then give it to the model. 
First let’s talk about feature scaling. Real-world data sets often contains features that are varying in degrees of magnitude, range and units. Therefore, in order for the model to interpret these features on the same scale, we need to perform feature scaling. There are many ways to scale features but one of the most used one is z-score normalization.
Z-score value is to understand how far the data point is from the mean. The measures the standard deviations below or above the mean. Converting a normal distribution into z- score normalization allows to calculate the probability of certain values occurring and to compare different data sets. It is calculated as the value minus its mean and dividing the result with the standard deviation. 
Also, gradient descent works faster after feature scaling as after normalization it will be easier to reach the local minima.  
Now that we have discussed what to do with the one with many predicted outputs let’s talk about what algorithm to use when it’s a classification problem or the output is a binary label. In linear regression we saw what function was predictions but for a classification problem linear regression won’t be suitable. In this algorithm we’ll use the sigmoid function. 
Logistic regression is used when there is a classification problem because it provides a discreet output unlike linear regression which provides continues output. The sigmoid function i.e. g(z) = 1/1+e(-z) where 0<g(z) <1. 
It function depends on how large or small is z. if z is large positive then g(z) which is the sigmoid function of z is going to be very close to one. And if z is a large negative number then g(z) becomes 1 over a giant number which is why g(z) is very close to 0. 
Moving on to the logistic regression algorithm, her f w,b (X) = g(w.x + b) (where w.x + b = z) and this gives us to the overall algorithm :-  f w,b (X) = 1/1+e-(w.x + b). and this gives the probability that the prediction is 1 or 0. For example if f w,b (X) = 0.7 this means that there’s a 70% chance that the tumor is malignant or 1. 
So, f w,b(X) = P(y = 1 | x ; w , b) the “;” says that w and b are the parameters that affect this computation of what is the probability of  y= 1 given the input feature is X.
After knowing the algorithm, we want to know how it computes the output. What the above function does is, it draws a decision boundary that partitions the underlying vector space into two sets, one for each class.  
The decision boundary is computed by the expression z = w.x + b = 0. For example: 
if f w,b(X) = g(z) = g(w1x1 + w2x2 + b) then the decision boundary for this one will be
 z = x1+x2-3 = 0
= > x1+x2 = 3
Which means if z > 3 then f w,b(X) will be 1 and if z < 3 then then f w,b(X) will be 0.
Now after this we want to optimize our model using cost function. The interesting part about using cost function in logistic regression is that it doesn’t use mean squared error for calculating the cost function because for logistic regression it’s not always convex and mean squared error gives us a non-convex line with many local minima. Instead it calculates the loss function that measures how well the model is doing or the error on one training example and takes the average of those to calculate for the entire training set. 
Note: we take the sum of all the loss function and not the product because we are using Mean Absolute Error / L1 Loss which is the average of absolute differences between the actual values and the predicted values.
So the cost function for logistic regression is :-  yi .log(f w,b (X)) + (1 − yi) · log(1 − f w,b (X) ) 
Gradient descent for logistic regression looks exactly similar as the gradient descent for linear regression but the key difference whereas in linear regression in f w,b(X) = w.x + b, in logistic regression f w,b = 1/ 1+ e-(z) where z = w.x + b.
Now that we know what models to use for what problems there are some complication that arises while doing all the above steps. The main one of them is the problem of overfitting and underfitting. 
Underfitting happens when the regression lines doesn’t fit all the data points which ends up not giving us good predictions for any new data points. In machine learning underfitting is also called high bias and these terms are used interchangeably more often. 
Overfitting happens when the regression line fits the data too well. Which means it tries to connect all the data points without caring about what will happen if a new data point is added. This is also called high variance and is used interchangeably in machine learning. 
To tackle the above problems, we use regularization to properly fit our model onto our test set. This is also a way to optimize our model as we are making it function more efficiently. 
The idea of regularization is that if there are smaller values for parameters there’s a bit likely having a simple model which therefore is less prone to overfitting. If we don’t know what features to penalize with the help of regularization it will penalize all the wj parameters. The algorithm also tries to keep the parameters wj small, which will tend to reduce overfitting. The value of lambda that you choose, specifies the relative importance or the relative trade off or how you balance these two goals. And so, what you want is some value of lambda that is in between that more appropriately balances these first and second terms of trading off, minimizing the mean squared error and keeping the parameters small. And when the value of lambda is not too small and not too large, but just right, then hopefully you end up able to fit a 4th order polynomial, keeping all of these features. 
We can apply regularization to both linear and logistic regression for optimizing the models.
Now earlier we discussed about normalization which was transporting the features on a same scale. How this is different form regularization as in regularization we are adjusting the wj parameter? 
Here’s where the difference arises in both the optimization: normalization is used to scale the features so that our data adjust to a similar scale of units or values whereas what regularization does is when the model needs to identity more important features ignore the noise (random variation not really related to classification) it executes some control on the parameters by rewarding simpler fitting functions over complex ones. Because if you give your model free rein to minimize the error on the given data, you can suffer from overfitting: the model insists on predicting the data set exactly, including those random variations. 
This is how we need to do both normalization and regularization for optimizing our model. 







