---
layout: post
title: Neural Operators
date: 2021-06-12
description: Project page for neural operators
---
Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar
- Papers: [FNO](https://arxiv.org/abs/2010.08895), [GNO](https://arxiv.org/abs/2010.08895), [MGNO](https://arxiv.org/abs/2010.08895).
- Code: [FNO](https://github.com/zongyi-li/fourier_neural_operator), [GNO](https://github.com/zongyi-li/graph-pde).
- Blog posts: [FNO](https://zongyi-li.github.io/blog/2020/fourier-pde/), [GNO](https://zongyi-li.github.io/blog/2020/graph-pde).
- Media cover: [MIT Tech Review](https://www.technologyreview.com/2020/10/30/1011435/ai-fourier-neural-network-cracks-navier-stokes-and-partial-differential-equations/),
[Quanta Magezine](https://www.quantamagazine.org/new-neural-networks-solve-hardest-equations-faster-than-ever-20210419/),
[Towards Data Science](https://towardsdatascience.com/ai-has-unlocked-a-key-scientific-hurdle-in-predicting-our-world-5343b4ed136e),
[Medium](https://medium.com/swlh/artificial-intelligence-can-now-solve-a-mathematical-problem-that-can-make-researchers-life-easier-9602c869128).
- Videos: [Anima's cover](https://www.youtube.com/watch?v=Bd4KvlmGbY4), 
[Zongyi's cover](https://www.youtube.com/watch?v=0Ve9xwNJO2o),
[Yannic's cover](https://www.youtube.com/watch?v=IaS72aHrJKE).


> The neural operator is a framework for operator learning, 
>epecially for learning the solution operators raising in partial differential equations.
>In this project page we will showcase several examples of neural operators.




## 1. Burgers Equation
The 1-d Burgers' equation is a non-linear PDE with various applications 
including modeling the one-dimensional flow of a viscous fluid. It takes the form


$$ \partial_t u(x,t) + \partial_x ( u^2(x,t)/2) = \nu \partial_{xx} u(x,t), \qquad x \in (0,1), t \in (0,1] $$

$$ u(x,0) = u_0(x), \qquad \qquad \:\: x \in (0,1) $$

with periodic boundary conditions where $$u_0 \in L^2((0,1);\R)$$ 
is the initial condition and $$\nu = 0.1, 0.01, 0.001$$ is the viscosity coefficient. 
We aim to learn the operator mapping the initial condition to the solution 
at time one, defined by $$u_0 \mapsto u(1)$$.

![Burgers equation](/assets/img/Burgers.png){: width="700px"}
Input functions (left) and output functions (right) for Burgers equation.

![Benchmarks](/assets/img/fourier_error.png){: width="700px"}
As shown in the Figure, FNO has one order better accuracy on Burgers equation and Darcy flow 
compared to other supervised learning methods.

## 2. Darcy Flow

We consider the steady-state of the 2-d Darcy Flow equation 
on the unit box which is the second order, linear, elliptic PDE

$$ - \nabla \cdot (a(x) \nabla u(x)) = f(x) \qquad x \in (0,1)^2 $$

$$ u(x) = 0 \qquad \quad \:\:x \in \partial (0,1)^2 $$

with a Dirichlet boundary where $$a \in L^\infty((0,1)^2;\R_+)$$  is the diffusion coefficient 
and $$f \in L^2((0,1)^2;\R)$$ is the forcing function. 
This PDE has numerous applications including modeling the pressure of the subsurface flow, 
the deformation of linearly elastic materials, and the electric potential in conductive materials. 
We are interested in learning the operator mapping the diffusion coefficient to the solution, 
 defined by $$a \mapsto u$$. 


![Darcy Equation](/assets/img/Darcy.png){: width="700px"}
The neural operator are resolution invariants. It is capable of the zero-shot super-ressolution.
It is trained on 16x16 resolution and evaluated on 241x241 resolution, while still achieving a very small error rate.



## 3. Navier-Stokes Equation

We consider the 2-d Navier-Stokes equation for a viscous, 
incompressible fluid in vorticity form on the unit torus:

$$ \partial_t w(x,t) + u(x,t) \cdot \nabla w(x,t) = \nu \Delta w(x,t) + f(x), \qquad x \in (0,1)^2, t \in (0,T]  $$

$$ \nabla \cdot u(x,t) = 0, \qquad \qquad  x \in (0,1)^2, t \in [0,T]  $$

$$ w(x,0) = w_0(x), \qquad \qquad \qquad  x \in (0,1)^2  $$

where $$u$$ is the velocity field, 
$$w = \nabla \times u$$ is the vorticity, 
$$w_0 $$ is the initial vorticity,  
$$\nu$$ is the viscosity coefficient, 
and $$f $$ is the forcing function. 
We are interested in learning the operator mapping the vorticity up to time 10 
to the vorticity up to some later time $$T > 10$$, 
defined by $$w|_{(0,1)^2 \times [0,10]} \mapsto w|_{(0,1)^2 \times (10,T]}$$. 
We experiment with the viscosities 
$$\nu = 1\mathrm{e}{-3}, 1\mathrm{e}{-4}, 1\mathrm{e}{-5}$$,
decreasing the final time $$T$$ as the dynamic becomes chaotic.


![Navier-Skotes Equation](/assets/img/ns_sr_v1e-4_labelled.gif){: width="700px"}
FNO achieves better accuracy compared to CNN-based methods. 
Further, it is capable of the zero-shot super-ressolution.
It is trained on 64x64x20 resolution and evaluated on 256x256x80 resolution, in both space and time.



## 4. Bayesian Inverse Problem

In this experiment, we use a function space Markov chain Monte Carlo (MCMC) method 
to draw samples from the posterior distribution of the initial vorticity 
in Navier-Stokes given sparse, noisy observations at time $$T=50$$. 
We compare the Fourier neural operator acting as a surrogate model 
with the traditional solvers used to generate our train-test data (both run on GPU).
We generate 25,000 samples from the posterior (with a 5,000 sample burn-in period), 
requiring 30,000 evaluations of the forward operator.

![Bayesian inverse problem](/assets/img/fourier_bayesian.png){: width="700px"}

The top left panel shows the true initial vorticity while the bottom left panel shows 
the true observed vorticity at $$T=50$$ with black dots indicating 
the locations of the observation points placed on a $$7 \times 7$$ grid. 
The top middle panel shows the posterior mean of the initial vorticity 
given the noisy observations estimated with MCMC using the traditional solver, 
while the top right panel shows the same thing but using FNO as a surrogate model. 
The bottom middle and right panels show the vorticity at $$T=50$$ 
when the respective approximate posterior means are used as initial conditions. 



## 5. Karamuto-Sivashinsky Equation
We consider the following one-dimensional Kuramoto-Sivashinsky equation, which is a common example of chaotic systems.

$$ \frac{\partial u}{\partial t} = -u \frac{\partial u}{\partial x} - \frac{\partial^2 u}{\partial x^2}  - \frac{\partial^4 u}{\partial x^4}, \qquad \text{on } [0,L] \times (0, \infty) $$

$$ u(\cdot, 0) = u_0, \qquad \qquad \qquad \text{on } [0, L] $$

where $$L = 32\pi \text{ or } 64 \pi$$ and the spatial domain $$ [0,L]$$
is equipped with periodic boundary conditions. 
In this case, the system is Markovian, and we want to learn the time-evolution operator
$$ u(t) \mapsto u(t+h)$$.

![KS trajectory](/assets/img/KS-trajectory.png){: width="700px"}
In general, FNO is capable to capture the exact trajectory of the chaotic system for a longer period, 
as it has a smaller per-step error.

![KS statistics](/assets/img/KS-stats.png){: width="700px"}
Chaotic systems are intrinsically instable. 
Smaller errors will accumulate and make the simulation diverge from the truth.
Even if FNO is capable to captrue the true trajectory for a period, 
it eventually diverges from the truth.
But neural operator still preserves the same orbit (attractor) of the system 
and its statistical properties.

## 6. Kolmogorov Flows

The Kolmogorov Flows is a special form of the 2-d Navier-Stokes equation.
It has a ergodic but chaotic states.

$$ \partial_t w(x,t) + u(x,t) \cdot \nabla w(x,t) = \nu \Delta w(x,t) + f(x), \qquad x \in (0,1)^2, t \in (0,T]  $$
The system is Markovian. Again, we want to learn the time-evolution operator
$$ w(t) \mapsto w(t+h)$$.

<!---
![KF vorticity](/assets/img/KF-vorticity.png){: width="700px"}
Five snapshots of the Kolmogorov Flows (vorticity) with Reynolds number 40.
--->

![KF statistics](/assets/img/KF-stats.png){: width="700px"}
![KF attractor](/assets/img/KF-Attractor.png){: width="700px"}


## 7. Plasticity
![Plasticity](/assets/img/plasticity.gif){: width="700px"}

## 8. Physics-informed Neural Operator (PINO)

![Physics-informed Neural Operator](/assets/img/pino-re500.gif){: width="700px"}

