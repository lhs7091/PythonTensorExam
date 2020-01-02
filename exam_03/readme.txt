Gradient descent algorithm
 - Minimize cost function
 - Gradient descent is used many minimization problems
 - For a given cost function, cost(W,b), it will find W, b to minimize cost
 - It can be applied to more general function: cost(w1, w2, ….)

How it works??
Start with initial guesses
 - Start at 0.0 (or any other value)
 - Keeping changing W and b a little bit to try and reduce cost(W, b)
Each time you change the parameters, you select the gradient which reduces cost(W,b) the most possible
Repeat
Do so until you converge to a local minimum
Has an interesting property
 - Where you start can determine which minimum you end up

Formal definition -> 미분을 이용한다.

Learning_rate??? the value of alpha
기울기 : gradient


