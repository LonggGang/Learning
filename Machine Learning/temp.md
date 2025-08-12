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

