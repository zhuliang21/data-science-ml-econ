# Practice Exam Review

**Qestion 1**: Unsupervised learning is when we have our algorithm learn inherent pattterns in the data. Which is an example of unsupervised learning?

Note: The difference between supervised learning and unsupervised learning is that in supervised learning, you have **labeled output** and you are trying to predict it based on other factors; in unsupervised learning, you have **no labeled output** and you are trying to find patterns in the data.

Option A: You have a huge volume of variables in your data, and you run an algorithm to decrease the number of variables by finding combinations of variables that explain a lot.

Explanation: This is an example of unsupervised learning because the algorithm is finding patterns in the data without any labeled output.

Option B: You are trying to model how fast different computers run by regressing speed on different factors.

Explanation: This is an example of supervised learning because you have labeled output (the speed of the computers) and you are trying to predict it based on other factors by regression.

Option C: You are trying to predict weather a candidate will win the presidential election based on data about previous winners.

Explanation: This is an example of supervised learning because you have labeled output (whether the candidate won or not in the previous elections) and you are trying to predict it based on other factors.

---

**Qestion 2**: Which of the following types of errors goes along with regression?

Note: There are two type of modeling: Regression and Classification.

- Regression is when you predict a **continuous** value, e.g. the price of a house, the temperature, etc; 
- Classification is when you predict a **discrete** value, e.g. whether an email is spam or not, whether a customer will buy a product or not, etc.

Option A: You predict a customer would purchase, and they do not purchase.

Explanation: This is an example of classification error because you are predicting a discrete value (whether the customer will purchase or not).

Option B: You predict a couple will get divorced, and they do not get divorced.

Explanation: This is an example of classification error because you are predicting a discrete value (whether the couple will get divorced or not).

Option C: You predict that sales would increase by 15%, and they actually increase by 3%

Explanation: This is an example of regression error because you are predicting a continuous value (the percentage increase in sales).

---

**Qestion 3**: Gradient descent is an algorithm for finding values of parameters $\theta_0$ and $\theta_1$ that minimize the cost function $J(\theta_0, \theta_1)$ by the following update rule:

$$\theta_1^\text{new} = \theta_1 - \alpha \frac{\partial J(\theta_0, \theta_1)}{\partial \theta_1}$$

$$\theta_0^\text{new} = \theta_0 - \alpha \frac{\partial J(\theta_0, \theta_1)}{\partial \theta_0}$$

Let's say right now we have

$$\frac{\partial J(\theta_0, \theta_1)}{\partial \theta_1} = \frac{\partial J(\theta_0, \theta_1)}{\partial \theta_0} = 0$$

What will the algorithm do?

Option A: increase parameters

Option B: decrease parameters

Option C: Remain constant (no change)

Explanation: The algorithm will update the parameters by the update rule. Now if the $\frac{\partial J(\theta_0, \theta_1)}{\partial \theta_1} = 0$, then $ \alpha \frac{\partial J(\theta_0, \theta_1)}{\partial \theta_1} = 0$, so the update rule will subtract 0 from the current value of $\theta_1$, so $\theta_1$ will remain constant. The same applies to $\theta_0$. The reason is that the gradient is 0, so the by Gradient Descent, we are at a local minimum, so the parameters will not change. Option C is correct.

---

**Qestion 4**: Gradient descent is an algorithm for finding values of parameters $\theta_0$ and $\theta_1$ that minimize the cost function $J(\theta_0, \theta_1)$ by the following update rule:

$$\theta_1^\text{new} = \theta_1 - \alpha \frac{\partial J(\theta_0, \theta_1)}{\partial \theta_1}$$

$$\theta_0^\text{new} = \theta_0 - \alpha \frac{\partial J(\theta_0, \theta_1)}{\partial \theta_0}$$

Let's say right now we have

$$\frac{\partial J(\theta_0, \theta_1)}{\partial \theta_1} = \frac{\partial J(\theta_0, \theta_1)}{\partial \theta_0} = 0$$

Which of the following is a **possible** explanation (select all that apply):

Option A: The algorithm has converged to a global minimum

Option B: The algorithm has converged to a local maximum

Option C: The algorithm has converged to a local minimum

Option D: The function is flat everywhere

Option E: The function is decreasing everywhere

Explanation: $\frac{\partial J(\theta_0, \theta_1)}{\partial \theta_1} = \frac{\partial J(\theta_0, \theta_1)}{\partial \theta_0} = 0$ means that the gradient at the **current** point is 0. There are two situations:

- The function is flat everywhere, so the gradient is 0 everywhere. This means that the algorithm will not converge to a local minimum or maximum, because there is no minimum or maximum. Option D is a possible explanation.
- The function has a varying gradient, and at this point the tangent line is horizontal, so this point could be a local minimum or maximum. This local minimum or maximum could be a global minimum or maximum, depending on the function. So option A, B, C are all possible explanations.

Option E is not a possible explanation because if the function is decreasing everywhere, then the gradient will never be 0 (it is always negative).

---

**Question 5**:  If a function is decreasing everywhere, which of the following will occur hwen running gradient descent to minimize the function?

Option A: The algorithm will converge to a local minimum

Option B: The algorithm will take too long to converge

Option C: The algorithm will converge to a global minimum

Option D: The algorithm will not converge

Explanation: If the function is decreasing everywhere, then the gradient $\frac{\partial J(\theta)}{\partial \theta}$  will be negative everywhere (never reach 0), so in the updating rule, the updating part $\alpha \frac{\partial J(\theta)}{\partial \theta}$ will be positive, so the algorithm will keep increasing the parameters, and it will never stop because the gradient is never 0. So the answer is D.

Another way to think about it is that if the function is decreasing everywhere, then the function does not have a minimum, so the algorithm will never converge.

---

**Qestion 6**: Newton's mathod will always converge faster than gradient descent.

Option A: True
Option B: False

Explanation: False. Newton's method is a second-order optimization algorithm that uses the Hessian matrix to find the minimum of a function, which can lead to faster convergence than gradient descent.
However, the Hessian matrix itself can be computationally expensive to calculate, especially for high-dimensional problems.

So we have two time variants of the same algorithm: the steps of the algorithm and the time to compute each step. Newton's method is faster in terms of the number of steps, but each step is more expensive. So in some cases, Newton's method can be slower than gradient descent.

---

**Qestion 7**: Which of the following is an effect of making the learning rate alpha too high in gradient descent?

Option A: Will coverge too fast

Option B: May take too long to converge

Option C: May overshoot optimum and thus not converge

Explanation: If the learning rate is too high, the algorithm may overshoot (meaning that the new value of the parameter is too far away from the old value), and thus not converge. We can use the case in the last week TA session to domonstrate this.

![Saraj Rival’s noteboo](https://www.makerluis.com/content/images/size/w2400/2023/11/Gradient_parabola_step_sizes.jpeg)

When the learning rate is small (less than 1), the algorithm will converge to the minimum step by step. When the learning rate is too high (equal to ), it skips the symmetric point which have the same absolute value of the gradient, and thus it will bounce back and forth between two points, and thus it will never converge. However, as long as the learning rate is less than 1, the algorithm will converge.

---