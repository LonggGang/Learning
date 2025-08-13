#### Example of continuous state space applications

Many robotic control applications, including the lunar lander application that you work on in the practice lab, have continuous state spaces.


**Discrete Set of States**

| Example (Mars Rover):     | State 1 | State 2 | State 3 | State 4 (Rover Starts) | State 5 | State 6 |
| ----------- | ----------- | ------- | ------- | ------- | -------| ------|

Discrete set of states: Mars Rover could only be in one of six possible positions.

But most robots can be in more than one of six or any discrete number of positions, instead, they can be in any of a very large number of continuous value positions. For example, if the Mars rover could be anywhere on a line, so its position was indicated by a number ranging from 0-6 kilometers where any number in between is valid. That would be an example of a continuous state space, because the position would be represented by a number such as that is 2.7 kilometers along or 4.8 kilometers or any other number between zero and six.

**Continuous States**

Example:

To control a self-drivng car to drive smoothly, then this car might include a few numbers such as:
* X,Y position
* Orientatioon
* Speeds in X,Y direction
* How quickly it's turning

Denote:

$$ 
S = \begin{bmatrix} x \\ y \\ \dot{x} \\ \dot{y} \\ \dot{\theta} \end{bmatrix} 
$$

Where:
* $\dot{x}$ : How quickly is x-coordinate changing
* $\dot{y}$ : How quickly is y-coordinate changing
* $\dot{\theta}$ : How quickly is the angle of the car changing
* $S$ : States - The vector of a list of 6 numbers that is input to a policy,and the job of a policy is look at these 12 numbers and decide what's an appropriate action to take in the car.

> So any continuous state reinforcement learning problem or a continuous state Markov decision process, continuously MTP. The state of the problem isn't just one of a small number of possible discrete values, like a number from 1-6. Instead, it's a vector of numbers, any of which could take any of a large number of values. ~ Andrew Ng

#### Lunar Lander

**Overview:**
The Lunar Lander is a simulated environment (often used in reinforcement learning) where your job is to safely land a spacecraft on the Moon’s surface between two flags.

**Actions:**
At each time step, you can choose one of four actions:

1. **Nothing** – no thrust, gravity pulls you down.
2. **Left** – fire left thruster (pushes lander right).
3. **Main** – fire main engine (pushes lander upward).
4. **Right** – fire right thruster (pushes lander left).

**State Variables:**
The state includes:

* $x, y$: position
* $\dot{x}, \dot{y}$: velocity in horizontal and vertical directions
* $\theta$: angle (tilt)
* $\dot{\theta}$: angular velocity
* $l, r$: binary values indicating if the left or right leg is touching the ground.

State : 

$$
S = \begin{bmatrix} x \\ y \\ \dot{x} \\ \dot{y} \\ \theta \\ \dot{\theta} \\ l \\ r \end{bmatrix} 
$$ 

**Reward Function:**

* Landing on the pad: **+100 to +140** (depends on accuracy)
* Moving closer to the pad: **positive reward**
* Moving away: **negative reward**
* Crashing: **−100**
* Soft landing: **+100**
* Each leg grounded: **+10**
* Main engine use: **−0.3** per step
* Side thruster use: **−0.03** per step

This reward function is carefully designed to encourage desirable behaviors (landing smoothly, saving fuel) and discourage bad ones (crashing, wasting fuel).

**Goal:**
Learn a policy $\pi(s)$ that chooses the best action in each state to maximize the **sum of discounted rewards**:

$$
\text{Return} = R_1 + \gamma R_2 + \gamma^2 R_3 + \dots
$$



with $\gamma = 0.985$ and 

$$ 
S = \begin{bmatrix} x \\ y \\ \dot{x} \\ \dot{y} \\ \theta \\ \dot{\theta} \\ l \\ r \end{bmatrix} 
$$

If we can find such a policy, the lander can reliably land between the flags.

#### Learning the state-value function
Let's see how we can use reinforcement learning to control the lunar lander or for other reinforcement learning problems. The key idea is that we're going to train a neural network to compute or to approximate the state action value function Q of SA, and that in turn will let us pick good actions.
***Deep Reinforcement Learning***

The heart of the learning algorithm is we're going to train a neural network that inputs the current state and the current action and computes or approximates Q of SA.

**Lunar Lander Example**

Input (X): 
* State :
$$ 
S = \begin{bmatrix} x \\ y \\ \dot{x} \\ \dot{y} \\ \theta \\ \dot{\theta} \\ l \\ r \end{bmatrix} 
$$ 
* Action: 4 possible action
  * Nothing
  * Left
  * Main
  * Right
We can encode any of those four actions using a one-hot feature vector. $\rightarrow$ four binary number.

$\rightarrow$ Neural Network Input : $\vec{x}$ = [ S , Action] (12 numbers)

Model :

We'll then take these 12 numbers and feed them to a neural network with, say, 64 units in the first hidden layer, 64 units in the second hidden layer, and then a single output in the output layer.

Output (Y):

$ Q(S,a) $ for all four actions. Whichever of these has the highest value, we would pick the corresponding action A.

Problem: But how do you get a training set with values for X and Y that you can then train a neural network on? 

***Bellman Equation***

**Recall**

* In the Lunar Lander simulator, if we don’t yet have a good policy, we can take **random actions**:

  * Fire left thruster
  * Fire right thruster
  * Fire main engine
  * Do nothing
* Each interaction with the environment produces:

  * **State (S)**: current position, velocity, angle, etc.
  * **Action (A)**: chosen move (one of four possible actions)
  * **Reward R(S)**: reward for being in that state
  * **Next State (S')**: new state after taking action

These four elements **(S, A, R(S), S')** are called a **tuple** in Python.


**Turning Tuples into Training Data**

* Suppose we have many tuples:
  (S1, A1, R(S1), S'1), (S2, A2, R(S2), S'2), … up to (S10000, A10000, …).
* Each tuple produces **one training example**:

  * **Input (X)** = (S, A)

    * S: 8 numbers (state features of the lunar lander)
    * A: 4 numbers (one-hot encoded action)
  * **Target Output (Y)** = computed using **Bellman Equation**:

$$
Y = R(S) + \gamma \max_{A'} Q(S', A')
$$

Where:

* $R(S)$ = reward for the current state
* $\gamma$ = discount factor
* $S'$ = next state after taking A
* $A'$ = all possible next actions
* $Q(S', A')$ = estimated future return from taking action $A'$ in state $S'$


**Role of Q(S', A')**

* Initially, $Q$ is **unknown**.
* We **start with a random guess** for $Q$.
* As the algorithm runs, Q-values get updated and improve over time.


**Building the Training Set**

* For each tuple:

  1. **X** = (state vector, one-hot action vector)
  2. **Y** = reward + discounted max Q-value of next state
* Example: If
  $R(S_1) = 1.5$, $\gamma = 0.9$, and
  $\max_{A'} Q(S'_1, A') = 12.2$, then:

$$
Y_1 = 1.5 + 0.9 \times 12.2 = 12.48
$$

* Repeat for all experiences to get a dataset of (X, Y) pairs.



**Training the Neural Network**

* Input: 12 features (8 state + 4 action)
* Output: single number (predicted Q-value for that state-action pair)
* Loss function: Mean Squared Error between predicted Q-value and computed Y.

***Learning Algorithm (Deep Q Network)***

Let's put it all together into a single learning algorithm.

Initialize neural network randomly as guess of $Q(s,a)$

Repeat \{
  * Take actions in the lunar lander. Get $(s,a,R(s), s')$
  * Store 10,000 most recent $(s,a,R(s), s')$ - Replay Buffer
  * Train neural network: 
    * Create training set of 10,000 examples using $ x = (s,a)$ and $ y = R(s) + y \max\limits_{a'}Q(s',a')$
    * Train $Q_{new}$ such that $Q_{new}(s,a) \approx y$
  * Set $Q= Q_{new}$
\}

> Many of the ideas in this algorithm are due to Min et al. And it turns out that if you run this algorithm where you start with a really random guess of the Q function, then use Bellman's equations to repeatedly try to improve the estimates of the Q function. Then by doing this over and over, taking lots of actions, training a model that will improve your guess for the Q function. And so for the next model you train, you now have a slightly better estimate of what is the Q function. And then the next model you train will be even better. And when you update Q equals Q new, then for the next time you train a model, Q of S prime A prime will be an even better estimate. ~ Andrew Ng

#### Algorithm refinement: Improved neural network architecture


**Previous Neural Network Setup**

* Original approach:

  * Input: **12 numbers** (8 for state + 4 for one-hot action)
  * Output: **Q(s, a)** for that specific state-action pair
* Problem:

  * To choose the best action from a state, we must **run inference 4 times** (once for each action) to find the maximum Q-value.


**Improved Architecture**

* New design:

  * **Input:** 8 numbers (state features of the lunar lander)
  * **Hidden layers:** Two layers, each with 64 units
  * **Output layer:** 4 units, each representing Q(s, a) for one possible action:

    1. Do nothing → Q(s, nothing)
    2. Fire left thruster → Q(s, left)
    3. Fire main engine → Q(s, main)
    4. Fire right thruster → Q(s, right)
* **Advantage:**

  * A single forward pass computes Q-values for **all actions** from state s.
  * We can instantly pick the action with the largest Q-value.

**Efficiency in Bellman Equation**

* Recall Bellman update:

$$
Q(s, a) \leftarrow R(s) + \gamma \max_{a'} Q(s', a')
$$

* This architecture speeds up:

  * The "$max_{a'}$" part, because the network outputs all $Q(s', a')$ in one run.
  * No need to run the network multiple times for the same next state.

**Conclusion:**
By switching to a network that takes **only the state as input** and outputs all action values at once, we reduce computation time and make both action selection and Bellman updates much faster.

#### Algorithm refinement: $\epsilon$ - greedy policy

**Problem**

* While still learning $Q(s,a)$, the agent must choose action in the Lunar Lander 
* If the agent only picks that currently maximizes $Q(s,a)$:
  * It might never try certain actions due to poor initials estimates.
  * This could prevent learning about actions that are actually useful.

***$\epsilon$-Greedy Policy***

How it works:

* With probability $1-\epsilon  \rightarrow$  choose the action that maximizesQ(s,a) (greedy step / exploitation).
* With probability $\epsilon  \rightarrow$  choose a random action (exploration step).

Example: 
$\epsilon = 0.05$ means 95% greedy, 5% random

**Purpose:**
  * Prevents the network from getting "stuck" with false beliefs about actions.
  * Allows learning about less-tried actions.

Exploration vs. Exploitation:

**Exploration:** Trying actions that may not be optimal, to gain information.

**Exploitation:** Using current knowledge to choose the best-known action.

RL often discusses the exploration - exploitation trade-off: balancing the two for effective learning.

***Tricks ( Epsilon Decay)***
Strategy
* Start with a high $\epsilon$ (e.g., 1.0 $\rightarrow$  always random at first).
* Gradually decrease to a small value (e.g., 0.01) over time.

$\rightarrow$ This ensures:
  * Early stage : broad exploration.
  * Later stage : mostly exploitation with occasional exploration.

**Notes**

* Hyperparameter tuning is critical in RL:
  * In supervised learning, poor parameters slow learning (e.g., 3x longer).
  * In RL, poor parameters can make learning 10x - 100x slower or fail entirely.

**Key takeaways**

> Epsilon-greedy policy ensures the agent explores enough to learn effectively while still using its knowledge to maximize reward. Adjusting $\epsilon$ over time is key to balancing exploration and exploitation in reinforcement learning.


#### Algorithm refinement: Mini-batch and soft updates

##### Mini Batches 

**Motivations**

* In supervised learning, using all training examples for every gradient descent step is slow for large datasets. (Batch Gradient Descent)
* Mini-batch gradient descent uses only a subset of data (e.g., 1,000 examples) per step.
* This makes each iteration faster, though updates are noisier.

**In RL (Replay Buffer):**

* Replay buffer may store 10,000 recent tuples (S, A, R(S), S').
* Instead of using all of them each time: Sample a mini-batch (e.g., 1,000) to train the network.
* Benefit: faster iterations, overall faster learning
* Drawback: slightly noisier updates, but usually worth it.

##### Soft updates

**Problems:** 

* Original algorithm: after training a new Q-network ($Q_{new}$), directly replace $Q$ with $Q_{new}$.

* If $Q_{new}$ is worse (due to noise), performance can drop suddenly.

**Soft updates approach**

Instead of full replacement:

$$
W \leftarrow \tau \cdot W_{new} + (1-\tau) \cdot W
\\
B \leftarrow \tau \cdot B_{new} + (1-\tau) \cdot B

$$

Where:

* $\tau$ is small (e.g., 0.01).
* $W,B$ are parameters of the old network.
* $W_{new},B_{new}$ are from the newly trained network.


Effect:
* Gradual change to Q-function.
* Reduces risk of instability or oscillation.
* Makes convergence more reliable.

***Takeways***

> Mini-batches make training **faster** by sampling small subsets of experiences, while soft updates ensure **stable** learning by blending new network parameters into the old ones gradually. These improvements help RL algorithms converge more smoothly, especially in challenging environments like the Lunar Lander.

#### The state of reinforcement learning

Understanding Reinforcement Learning

* Reinforcement learning has gained significant attention, but much of the research is based on simulated environments rather than real-world applications.
* Implementing reinforcement learning in real robots is often more challenging than in simulations or video games.

Applications and Comparisons

* There are fewer practical applications of reinforcement learning compared to supervised and unsupervised learning methods.
* In practical scenarios, supervised and unsupervised learning techniques are more commonly used than reinforcement learning.

Future Potential

* Despite its current limitations, reinforcement learning remains a crucial area of research with significant potential for future applications.
* Understanding reinforcement learning can enhance your ability to develop effective machine learning systems, especially in robotic control applications.

