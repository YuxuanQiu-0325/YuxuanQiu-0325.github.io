---
layout: post
title: "A Unified Dynamical Perspective on Generative Modeling"
---

# Foundations of Stochastic Dynamics

## Discrete Random Walk

We consider a one-dimensional discrete random walk model: a particle
moves a fixed step length $`\Delta x`$ to the right with a probability
of $`0.5`$ or to the left by the same step length with a probability of
$`0.5`$, within each time step $`\Delta t`$. Let the random variable
$`X_i`$ denote the displacement in the $`i`$-th step.

#### Single-Step Statistical Properties

The core statistical properties for any single-step displacement $`X_i`$
are as follows:

- **Expectation** The mean displacement for each step is $`0`$, which
  indicates that the random motion is unbiased.
  ``` math
  \mathbb{E}[X_i] = (+\Delta x) \times 0.5 + (-\Delta x) \times 0.5 = 0
  ```

- **Variance** The variance reflects the uncertainty or magnitude of
  fluctuation for each step’s displacement.
  ``` math
  \mathrm{Var}(X_i) = \mathbb{E}[X_i^2] - (\mathbb{E}[X_i])^2 = (\Delta x)^2 \times 0.5 + (-\Delta x)^2 \times 0.5 - 0^2 = (\Delta x)^2
  ```

#### Multi-Step Statistical Properties

After $`n`$ steps, the total displacement of the particle, $`S_n`$, is
the sum of the individual independent displacements, i.e.,
$`S_n = \sum_{i=1}^n X_i`$.

- **Expectation** Regardless of the number of steps taken, the expected
  position (i.e., the average position) of the particle remains at the
  origin.
  ``` math
  \mathbb{E}[S_n] = \sum_{i=1}^n \mathbb{E}[X_i] = 0
  ```

- **Variance** The variance of the total displacement is proportional to
  the number of steps, $`n`$. This implies that the more steps are
  taken, the larger the potential range of the particle’s movement
  becomes.
  ``` math
  \mathrm{Var}(S_n) = \sum_{i=1}^n \mathrm{Var}(X_i) = n (\Delta x)^2
  ```

## Transition to Continuous Brownian Motion

To derive the continuous-time limit from the discrete random walk, we
construct a stochastic process $`S(t)`$ and examine its convergence
behavior as the time step $`\Delta t \to 0`$ and the space step
$`\Delta x \to 0`$. Let the total time $`t = n \cdot \Delta t`$ be held
as a finite constant, which implies that as $`\Delta t \to 0`$, the
number of steps $`n \to \infty`$. On this timescale, the variance of the
total displacement $`S_n`$ can be rewritten as:
``` math
\mathrm{Var}(S_n) = n(\Delta x)^2 = \frac{t}{\Delta t}(\Delta x)^2 = t \frac{(\Delta x)^2}{\Delta t}
```
For the stochastic process to converge to a non-trivial limit—that is, a
process that neither collapses to zero nor diverges to infinity—the
variance of the total displacement must converge to a non-zero, finite
value. This requires the ratio $`\frac{(\Delta x)^2}{\Delta t}`$ to be a
constant in the limit as $`\Delta t \to 0`$. We accordingly define this
constant as the square of the diffusion coefficient, denoted as
$`\sigma^2`$:
``` math
\lim_{\Delta t \to 0} \frac{(\Delta x)^2}{\Delta t} \equiv \sigma^2 \implies \Delta x \approx \sigma\sqrt{\Delta t}
```

Here, $`\sigma`$ is known as the **diffusion coefficient**. It
quantifies the growth rate of the particle’s spatial range (i.e., its
uncertainty) per unit of time. Based on this scaling relationship, we
can determine the key statistical properties of the continuous process:
``` math
\mathbb{E}[S(t)] = \lim_{n \to \infty} \mathbb{E}[S_n] = 0,\quad\mathrm{Var}(S(t)) = \sigma^2 t    
\label{eq:1}
```

According to the functional **Central Limit Theorem**, as
$`n \to \infty`$, the discrete random walk process $`S_n`$ converges in
distribution to a continuous stochastic process known as the Wiener
Process, or standard Brownian motion. Therefore, at any given time
$`t`$, the probability distribution of the particle’s position is a
normal distribution with a mean of $`0`$ and a variance of
$`\sigma^2 t`$:
``` math
S(t) \sim \mathcal{N}(0, \sigma^2 t)
```

## Derivation of the Macroscopic Diffusion Equation

Previously, we described the random behavior of a single particle over
long timescales, where its position probability distribution $`S_n`$
converges to a normal distribution $`\mathcal{N}(0, \sigma^2 t)`$. We
now shift from the microscopic, single-particle random perspective to a
macroscopic, deterministic view of a particle ensemble, aiming to derive
the partial differential equation (PDE) that describes the
spatiotemporal evolution of the particle number density $`\rho(x, t)`$.

We assume the evolution of the particle cloud is a Markovian process,
whose temporal evolution can be described by the Chapman-Kolmogorov
equation. At time $`t + \Delta t`$, the particle density at position
$`x`$ is the sum of contributions from all particles that were at other
positions $`x-z`$ at time $`t`$ and arrived at $`x`$ after a single
displacement step $`z`$:
``` math
\rho(x, t + \Delta t) = \int_{-\infty}^{\infty} \rho(x - z, t)\, \phi(z)\, \mathrm{d}z
\label{eq:chapman_kolmogorov}
```
where $`\phi(z)`$ is the stationary probability density function (PDF)
of a single-step jump displacement $`z`$, which does not vary with time
or space.

To derive a differential equation from this integral equation, we
perform a Taylor series expansion on both sides in the limit as
$`\Delta t \to 0`$. The left side of the equation,
$`\rho(x, t + \Delta t)`$, is expanded with respect to time $`t`$,
keeping terms up to the first order:
``` math
\rho(x, t + \Delta t) \approx \rho(x, t) + \frac{\partial \rho(x, t)}{\partial t} \Delta t
\label{eq:taylor_time}
```
The term $`\rho(x - z, t)`$ inside the integral on the right side is
expanded with respect to the spatial displacement $`z`$, keeping terms
up to the second order due to the infinitesimal nature of $`z`$:
``` math
\rho(x - z, t) \approx \rho(x, t) - z \frac{\partial \rho(x, t)}{\partial x} + \frac{z^2}{2} \frac{\partial^2 \rho(x, t)}{\partial x^2}
\label{eq:taylor_space}
```

Substituting Eq. <a href="#eq:taylor_space" data-reference-type="eqref"
data-reference="eq:taylor_space">[eq:taylor_space]</a> into the right
side of Eq. <a href="#eq:chapman_kolmogorov" data-reference-type="eqref"
data-reference="eq:chapman_kolmogorov">[eq:chapman_kolmogorov]</a>, we
obtain:
``` math
\int_{-\infty}^{\infty} \left[ \rho(x, t) - z \frac{\partial \rho}{\partial x} + \frac{z^2}{2} \frac{\partial^2 \rho}{\partial x^2} \right] \phi(z)\, \mathrm{d}z
= \rho \int \phi(z)\mathrm{d}z - \frac{\partial \rho}{\partial x} \int z\phi(z)\mathrm{d}z + \frac{1}{2}\frac{\partial^2 \rho}{\partial x^2} \int z^2\phi(z)\mathrm{d}z
```
We utilize the statistical properties of $`\phi(z)`$:

- Probability normalization:
  $`\int_{-\infty}^{\infty} \phi(z) \mathrm{d}z = 1`$

- Zero mean (assuming symmetric jumps):
  $`\mathbb{E}[z] = \int_{-\infty}^{\infty} z \phi(z) \mathrm{d}z = 0`$

- Finite second moment (variance):
  $`\mathbb{E}[z^2] = \int_{-\infty}^{\infty} z^2 \phi(z) \mathrm{d}z = \sigma_z^2`$

Thus, the right side of the integral
equation <a href="#eq:chapman_kolmogorov" data-reference-type="eqref"
data-reference="eq:chapman_kolmogorov">[eq:chapman_kolmogorov]</a> can
be approximated as
$`\rho(x, t) + \frac{1}{2} \frac{\partial^2 \rho}{\partial x^2} \mathbb{E}[z^2]`$.
Combining the time expansion from
Eq. <a href="#eq:taylor_time" data-reference-type="eqref"
data-reference="eq:taylor_time">[eq:taylor_time]</a> with the result
above, we have:
``` math
\rho(x, t) + \frac{\partial \rho}{\partial t} \Delta t \approx \rho(x, t) + \frac{1}{2} \frac{\partial^2 \rho}{\partial x^2} \mathbb{E}[z^2]
```
Canceling $`\rho(x, t)`$ and dividing by $`\Delta t`$, we arrive at the
difference quotient form:
``` math
\frac{\partial \rho(x, t)}{\partial t} \approx \frac{\mathbb{E}[z^2]}{2\Delta t} \frac{\partial^2 \rho(x, t)}{\partial x^2}
```
In the continuous limit as $`\Delta t \to 0`$, we define the macroscopic
**Diffusion Coefficient** $`D`$:
``` math
D \equiv \lim_{\Delta t \to 0} \frac{\mathbb{E}[z^2]}{2\Delta t} 
\label{eq:5}
```
Finally, we obtain the famous partial differential equation describing
the evolution of particle density—the **Diffusion Equation**, also known
as Fick’s second law:
``` math
\frac{\partial \rho(x, t)}{\partial t} = D \frac{\partial^2 \rho(x, t)}{\partial x^2}
```
This is a deterministic equation that describes the average macroscopic
effect of the collective behavior of a large number of independent
random-walking particles. Therefore, Eulerian perspective that describes
the particle cloud’s density via the diffusion equation, and the
microscopic, Lagrangian perspective that tracks a single particle’s
trajectory are unified descriptions of the same underlying process:

For a single particle, the variance of a single-step jump is
$`\sigma^2 \Delta t`$ from Eq.<a href="#eq:1" data-reference-type="ref"
data-reference="eq:1">[eq:1]</a>. Substituting this into the definition
of the diffusion coefficient from
Eq.<a href="#eq:5" data-reference-type="ref"
data-reference="eq:5">[eq:5]</a>:
``` math
D = \frac{(\Delta x)^2}{2\Delta t} = \frac{\sigma^2 \Delta t}{2\Delta t} = \frac{\sigma^2}{2}
```

If all particles are initially concentrated at the origin
($`\rho(x, 0) = \delta(x)`$), the solution to the diffusion equation at
time $`t`$ is
``` math
\rho(x, t) = \frac{1}{\sqrt{4\pi D t}} \exp\left(-\frac{x^2}{4Dt}\right)
= \frac{1}{\sqrt{2\pi \sigma^2 t}} \exp\left(-\frac{x^2}{2\sigma^2 t}\right)
```

This is precisely a normal distribution $`\mathcal{N}(0,\, \sigma^2 t)`$
with a mean of $`0`$ and a variance of $`\sigma^2 t`$, which is in
perfect agreement with the limiting distribution we derived from the
single-particle process. Therefore, the smooth diffusion of particle
density on a macroscopic scale is, in essence, the statistical
manifestation of the independent, random motion of a large number of
particles.

## Stochastic Differential Equations and the Fokker-Planck Equation

The preceding discussion has focused on the simplest form of Brownian
motion, characterized by the absence of external forces and a constant
diffusion coefficient. In a more general framework, the trajectory of a
single particle simultaneously influenced by deterministic forces and
random fluctuations can be modeled by a **Stochastic Differential
Equation (SDE)**. Its Itô form is typically written as:
``` math
\label{eq:sde_general_en}
\mathrm{d}X_t = \mu(X_t, t)\mathrm{d}t + \sigma(X_t, t)\mathrm{d}W_t
```
The terms in this equation represent different physical effects:

- The **drift term**, $`\mu(X_t, t)\mathrm{d}t`$, describes the
  deterministic part of the particle’s motion. It represents the average
  infinitesimal displacement caused by a systematic force field.

- The **diffusion term**, $`\sigma(X_t, t)\mathrm{d}W_t`$, models the
  random component arising from stochastic phenomena (such as collisions
  with surrounding molecules). The intensity of this random motion is
  controlled by the diffusion coefficient $`\sigma(X_t, t)`$, while
  $`\mathrm{d}W_t`$ is the infinitesimal increment of the Wiener
  process.

The SDE provides a Lagrangian description of a single particle’s
trajectory, while the corresponding Eulerian description, which
describes the evolution of the probability density function
$`\rho(x, t)`$ for an ensemble of such particles, is given by the
**Fokker-Planck Equation**:
``` math
\label{eq:fpe_general_en}
\frac{\partial \rho(x, t)}{\partial t} = -\frac{\partial}{\partial x} \left[ \mu(x, t)\rho(x, t) \right] + \frac{\partial^2}{\partial x^2} \left[ D(x, t)\rho(x, t) \right]
```
Here, the diffusion coefficient $`D(x, t)`$ is related to the SDE’s
diffusion term by the relation $`D(x, t) = \frac{1}{2}\sigma(x, t)^2`$.

In the special case of standard Brownian motion, the system is not
subject to external forces ($`\mu(x, t) = 0`$) and experiences random
fluctuations of constant intensity ($`\sigma(x, t) = \sigma`$, where
$`\sigma`$ is a constant). In this case, the SDE from
Eq. <a href="#eq:sde_general_en" data-reference-type="eqref"
data-reference="eq:sde_general_en">[eq:sde_general_en]</a> simplifies
to:
``` math
\mathrm{d}X_t = \sigma \mathrm{d}W_t
```
Correspondingly, the Fokker-Planck Equation from
Eq. <a href="#eq:fpe_general_en" data-reference-type="eqref"
data-reference="eq:fpe_general_en">[eq:fpe_general_en]</a> degenerates
into the standard diffusion equation we derived earlier:
``` math
\frac{\partial \rho}{\partial t} = D \frac{\partial^2 \rho}{\partial x^2}
```
where $`D = \sigma^2 / 2`$.

These two perspectives—the microscopic stochastic dynamics of a single
particle (SDE) and the macroscopic deterministic evolution of the
probability density for a particle ensemble (Fokker-Planck) are
mathematically equivalent descriptions of the same underlying physical
process. The smooth diffusion phenomenon observed on a macroscopic scale
is the statistical manifestation of countless independent random events
occurring on a microscopic scale.

## Limiting Case: Degeneration from SDE to ODE

The relationship between a Stochastic Differential Equation (SDE) and an
Ordinary Differential Equation (ODE) can be understood by examining the
limiting case where the stochastic term vanishes. The general form of an
SDE is:
``` math
\mathrm{d}X_t = \mu(X_t, t)\mathrm{d}t + \sigma(X_t, t)\mathrm{d}W_t
```
When the intensity of the stochastic term approaches zero, i.e.,
$`\sigma(X_t, t) \to 0`$, the random fluctuations are suppressed, and
the SDE degenerates into a purely deterministic ODE:
``` math
\mathrm{d}x_t = \mu(x_t, t)\mathrm{d}t \quad \implies \quad \frac{\mathrm{d}x}{\mathrm{d}t} = \mu(x, t)
```
In this limit, the particle’s trajectory is uniquely determined by its
initial conditions and the velocity field $`\mu(x, t)`$, containing no
randomness.

This transition is also directly reflected in the corresponding
evolution of the probability density. As $`\sigma(x, t) \to 0`$, the
diffusion coefficient $`D(x,t) = \sigma^2(x,t)/2`$ also approaches zero.
Consequently, the diffusion term in the Fokker-Planck equation vanishes,
and the equation simplifies to the **Liouville equation**, a form of the
continuity equation:
``` math
\frac{\partial \rho(x, t)}{\partial t} = -\frac{\partial}{\partial x}[\mu(x, t)\rho(x, t)]
\label{eq:ode_FPE}
```
This equation describes a process where the probability density is
transported purely by the deterministic drift field $`\mu(x, t)`$. It
signifies that the total probability is conserved, and the evolution of
the density arises solely from advection by the underlying deterministic
flow, without any additional diffusion or broadening.

# Diffusion Models

The core objective of a generative model is to learn an empirical data
distribution $`p_{\text{data}}(x)`$. This distribution is typically
high-dimensional and multimodal, and its probability density function is
analytically intractable, known only through a finite dataset
$`\{ x_i \}_{i=1}^N`$ sampled from it. However, the Manifold Hypothesis
posits that such data points $`x`$ are concentrated on a low-dimensional
manifold $`\mathcal{M}`$ embedded within the high-dimensional ambient
space. This assumption reframes the generative task from directly
estimating the intractable density $`p_{\text{data}}`$ to learning a
parameterized transformation $`f_\theta`$. This function is designed to
map a simple, easy-to-sample prior distribution $`p_{\text{init}}`$
(e.g., a standard normal distribution) onto the complex data
distribution residing on the manifold. The fundamental task can thus be
summarized as learning a mapping such that:
``` math
p_{\text{init}} \xrightarrow{f_\theta} p_{\text{data}}
```

## Itô’s Lemma and Euler-Maruyama Discretization

### Itô’s Lemma and the Itô Integral

**Itô’s Lemma** is a cornerstone of stochastic analysis, providing a
version of the chain rule for the calculus of stochastic processes.
Consider a random variable $`X_t`$ described by the following Itô
process:
``` math
\mathrm{d}X_t = \mu(X_t, t)\mathrm{d}t + \sigma(X_t, t)\mathrm{d}W_t \label{eq:sde_ito_revised}
```
Let $`f(t, x)`$ be a twice continuously differentiable scalar function
of time $`t`$ and state $`x`$. Itô’s Lemma gives the total differential
of $`f(t, X_t)`$ as:
``` math
\mathrm{d}f(t, X_t) = \left( \frac{\partial f}{\partial t} + \mu(X_t, t)\frac{\partial f}{\partial x} + \frac{1}{2}\sigma(X_t, t)^2 \frac{\partial^2 f}{\partial x^2} \right)\mathrm{d}t + \sigma(X_t, t)\frac{\partial f}{\partial x}\mathrm{d}W_t
\label{eq:ito_lemma_full_revised}
```
Compared to the chain rule of classical calculus, Itô’s Lemma includes
an additional second-derivative term. This term arises from the non-zero
quadratic variation of the Wiener process
($`(\mathrm{d}W_t)^2 = \mathrm{d}t`$) and is crucial for accurately
describing the evolution of stochastic processes.

The solution to the SDE in
Eq. <a href="#eq:sde_ito_revised" data-reference-type="eqref"
data-reference="eq:sde_ito_revised">[eq:sde_ito_revised]</a> requires
integrating both sides. The integral of the drift term,
$`\mu(X_t, t)\mathrm{d}t`$, is a standard Riemann integral. However, due
to the nowhere-differentiable nature of the Wiener process, the integral
of the diffusion term, $`\sigma(X_t, t)\mathrm{d}W_t`$, must be defined
using a special formulation, namely the **Itô Integral** (for a
derivation, see
Appendix <a href="#appendix:Ito" data-reference-type="ref"
data-reference="appendix:Ito">5</a>). The complete integral form of the
SDE is written as:
``` math
X_T = X_0 + \int_0^T \mu(X_t, t)\mathrm{d}t + \int_0^T \sigma(X_t, t)\mathrm{d}W_t
\label{eq:sde_integral_form}
```

### Discretizing the Itô Integral: The Euler-Maruyama Method

Analytically solving the integral form of an SDE,
Eq. <a href="#eq:sde_integral_form" data-reference-type="eqref"
data-reference="eq:sde_integral_form">[eq:sde_integral_form]</a>, is
often very difficult. Therefore, we resort to numerical methods to
approximate its solution, the most fundamental of which is the
Euler-Maruyama Method. The core idea of this method is to discretize the
continuous integration process into a series of small, computable steps.

We discretize the time axis into points $`t_i = i \cdot h`$ with a step
size of $`h`$. Within any given time step $`[t_i, t_{i+1}]`$, we
approximate the integrals from
Eq. <a href="#eq:sde_integral_form" data-reference-type="eqref"
data-reference="eq:sde_integral_form">[eq:sde_integral_form]</a> as
follows:

- **Drift Term Integral Approximation**:
  $`\int_{t_i}^{t_{i+1}} \mu(X_t, t)\mathrm{d}t \approx \mu(X_i, t_i)h`$

- **Diffusion Term Integral Approximation**:
  $`\int_{t_i}^{t_{i+1}} \sigma(X_t, t)\mathrm{d}W_t \approx \sigma(X_i, t_i)(W_{t_{i+1}} - W_{t_i})`$

According to the properties of the Wiener process, the increment
$`\Delta W_i = W_{t_{i+1}} - W_{t_i}`$ is a random variable drawn from a
normal distribution $`\mathcal{N}(0, h)`$. We can generate it as
$`\sqrt{h} \cdot \varepsilon_i`$, where
$`\varepsilon_i \sim \mathcal{N}(0, \mathbf{I})`$ is a standard normal
random variable.

Combining these approximations, we arrive at the iterative update rule
for the Euler-Maruyama method:
``` math
X_{i+1} = X_i + \mu(X_i, t_i)h + \sigma(X_i, t_i)\sqrt{h} \cdot \varepsilon_i
\label{eq:euler_maruyama_revised}
```

In the generative process of a diffusion model, the diffusion
coefficient $`\sigma(t)`$ is typically a predefined schedule. The core
modeling task is to learn a parameterized drift function
$`\mu_\theta(x, t)`$ that, during the reverse-time process, can
accurately guide a simple noise distribution back to the target data
distribution. Once an accurate **$`\mu_\theta(x, t)`$** is obtained, one
can generate data samples $`x_T`$ from random noise
$`x_0 \sim \mathcal{N}(0, \mathbf{I})`$ by simulating this discretized
SDE process.

<div class="algorithm">

<div class="algorithmic">

Learned drift $`\mu_\theta(x,t)`$, diffusion schedule $`\sigma(t)`$,
number of steps $`N`$, total time $`T`$ $`h \gets T / N`$ Sample initial
state $`x_0 \sim \mathcal{N}(0, \mathbf{I})`$ $`t_i \gets i \cdot h`$
Sample $`\varepsilon_i \sim \mathcal{N}(0, \mathbf{I})`$
$`x_{i-1} \gets x_i + \mu_\theta(x_i, t_i)h + \sigma(t_i)\sqrt{h} \cdot \varepsilon_i`$
$`x_N`$

</div>

</div>

## How to Learn $`\mu_{\theta}(x,t)`$?

### From Simple to Complex, or Complex to Simple?

The core challenge for generative models is that directly modeling the
complex data distribution $`p_{\text{data}}`$ from a simple distribution
is analytically intractable. Diffusion models propose an indirect
modeling strategy: first, define a forward diffusion process that
gradually transforms the data distribution into a simple prior
distribution (e.g., Gaussian noise), and second, learn the reverse
generative process.

#### The Forward Process: Analytically Tractable Diffusion Dynamics

The forward process from data to noise is mathematically straightforward
to define. It is typically defined by a Stochastic Differential Equation
(SDE) with predefined parameters:
``` math
\mathrm{d}X_t = f(X_t, t)\mathrm{d}t + g(t)\mathrm{d}W_t
```
In this equation, the drift function $`f(X_t, t)`$ and the diffusion
function $`g(t)`$ together define an analytically tractable stochastic
process that gradually eliminates the structural information in the data
manifold. Since the dynamics of this process are predetermined, its
transition kernel $`p_t(x_t|x_0)`$ at any time $`t`$ is also
analytically known.

#### The Reverse Process: Generative Dynamics to be Learned

The reverse process is responsible for the generation task: sampling
from the prior distribution and recovering the target data. This process
can also be described by a reverse-time SDE. However, unlike the forward
process, the drift term of this reverse SDE is not predefined. Instead,
it depends on the score function, $`\nabla_x \log p_t(x)`$, of the
marginal probability density $`p_t(x)`$ of the forward process at each
time $`t`$. Because $`p_t(x)`$ is itself intractable, the reverse drift
term is unknown and must be learned via a parameterized model. This
constitutes the core learning problem of diffusion models.

### Score Matching

For a forward diffusion process described by the following SDE:
``` math
\mathrm{d}X_t = f(X_t, t) \mathrm{d}t + g(t) \mathrm{d}W_t
\label{eq:forward_sde}
```
According to the theory of stochastic processes (Anderson 1982), there
exists a corresponding reverse process that traces time backward from
$`T`$ to $`0`$. This process is described by the following
**reverse-time SDE** (see
Appendix <a href="#appendix:SDE" data-reference-type="ref"
data-reference="appendix:SDE">6</a> for details):
``` math
\mathrm{d}x = \left[f(x, t) - g(t)^2 \nabla_x \log p_t(x)\right] \mathrm{d}t + g(t) \mathrm{d}\bar{W}_t
\label{eq:reverse_sde}
```
Here, $`\mathrm{d}\bar{W}_t`$ represents the infinitesimal increment of
a standard Wiener process flowing backward in time, and $`\mathrm{d}t`$
represents an infinitesimal negative time step.

According to the analytical form of the reverse SDE,
Eq. <a href="#eq:reverse_sde" data-reference-type="eqref"
data-reference="eq:reverse_sde">[eq:reverse_sde]</a>, the drift term of
the generative process is jointly determined by the predefined functions
$`f(x,t)`$, $`g(t)`$, and the key unknown term: the **score function**,
$`\nabla_x \log p_t(x)`$. Therefore, the task of learning the generative
model is transformed into the problem of how to effectively estimate
this score function.

Directly computing the marginal score function $`\nabla_x \log p_t(x)`$
is analytically intractable. The fundamental reason is that the marginal
probability density $`p_t(x)`$ is obtained by integrating (i.e.,
marginalizing) the conditional probability density $`p_t(x|x_0)`$ over
the entire unknown true data distribution $`p_{\text{data}}(x_0)`$:
``` math
p_t(x) = \int p_t(x_t|x_0)p_{\text{data}}(x_0)\mathrm{d}x_0
```
This integral depends on $`p_{\text{data}}`$, which makes it impossible
to compute directly. However, a key insight is that although the
marginal score function is intractable, the **conditional score
function** $`\nabla_x \log p_t(x_t|x_0)`$ is analytically known. This is
because the forward noising process $`p_t(x|x_0)`$ is determined by the
SDE that we predefined.

Fortunately, a precise mathematical relationship exists between the
conditional score function $`\nabla_x \log p_t(x_t|x_0)`$ and the
marginal score function $`\nabla_x \log p_t(x)`$. We can derive this
relationship by starting from the definition of the marginal score
function and applying Bayes’ theorem:
``` math
\begin{aligned}
\nabla_{x_t}\log p_t(x_t) &= \frac{\nabla_{x_t}p_t(x_t)}{p_t(x_t)} \nonumber \\
&= \frac{1}{p_t(x_t)} \nabla_{x_t} \int p_t(x_t|x_0) p_{\text{data}}(x_0) \mathrm{d}x_0 \nonumber \\
&= \frac{1}{p_t(x_t)} \int \nabla_{x_t} p_t(x_t|x_0) p_{\text{data}}(x_0) \mathrm{d}x_0 \quad (\text{using the Leibniz integral rule}) \nonumber \\
&= \frac{1}{p_t(x_t)} \int \left( \nabla_{x_t} \log p_t(x_t|x_0) \right) p_t(x_t|x_0) p_{\text{data}}(x_0) \,\mathrm{d}x_0 \quad ( \nabla p = p \nabla \log p) \nonumber \\
&= \int \left( \nabla_{x_t} \log p_t(x_t|x_0) \right) \frac{p_t(x_t|x_0) \, p_{\text{data}}(x_0)}{p_t(x_t)} \, \mathrm{d}x_0 \nonumber \\
&= \int \left( \nabla_{x_t} \log p_t(x_t|x_0) \right) p_t(x_0|x_t) \, \mathrm{d}x_0 \quad (\text{by Bayes' theorem}) \nonumber \\
&= \mathbb{E}_{p_t(x_0|x_t)} \left[ \nabla_{x_t} \log p_t(x_t|x_0) \right] \label{eq:score_identity_revised}
\end{aligned}
```
Eq. <a href="#eq:score_identity_revised" data-reference-type="eqref"
data-reference="eq:score_identity_revised">[eq:score_identity_revised]</a>
shows that the marginal score function is the expectation of the
conditional score function with respect to the posterior probability
distribution $`p_t(x_0|x)`$. This important relationship transforms the
problem of estimating a complex, intractable marginal score function
into one of estimating a simpler, tractable conditional score function.
Therefore, we can train a parameterized network $`s_\theta(x, t)`$ to
approximate the conditional score function $`\nabla_x \log p_t(x|x_0)`$
over all times $`t`$ and data points $`x_0`$:

``` math
L_{\text{score}}(\theta) = \left\| s_\theta(x_t, t) - \mathbb{E}_{x_0 \sim p_t(x_0|x_t)} \left[ \nabla_{x_t} \log p_t(x_t|x_0) \right]
\right\|^2
```

According to **Optimal Estimation Theory**(See Appendix
<a href="#appendix:optimal_estimation" data-reference-type="ref"
data-reference="appendix:optimal_estimation">7</a>), for an optimization
problem measured by mean squared error, the conditional expectation is
the optimal estimator. In this context, the marginal score function
$`\nabla_{x_t}\log p_t(x_t)`$ is precisely the optimal mean squared
error estimate of the conditional score
$`\nabla_{x_t} \log p_t(x_t|x_0)`$, given the noisy sample $`x_t`$. This
conclusion reveals a critical equivalence: a loss function designed to
fit the intractable marginal score (i.e.,
$`\mathbb{E}_{x_0 \sim p_t(x_0|x_t)} \left[ \nabla_{x_t} \log p_t(x_t|x_0) \right]`$)
and a loss function designed to fit the tractable conditional score
(i.e., $`\nabla_{x_t} \log p_t(x_t|x_0)`$) share the exact same optimal
solution.

Therefore, we can instead minimize the latter, which is the objective
function of **Denoising Score Matching (DSM)**. This method trains the
score network $`s_\theta(x_t, t)`$ by directly fitting the analytically
known conditional score:

``` math
\min_{\theta} \, \mathbb{E}_{t \sim \mathcal{U}(0,T)} \, \mathbb{E}_{x_0 \sim p_{\text{data}}} \, \mathbb{E}_{x_t \sim p_t(\cdot|x_0)} \left[ \left\| s_\theta(x_t, t) - \nabla_{x_t} \log p_t(x_t | x_0) \right\|^2 \right]
\label{eq:dsm_loss}
```

## A Unified Dynamical Perspective on Diffusion Models: VP-SDE, VE-SDE, and Langevin Dynamics

Modern score-based generative models are unified under a continuous-time
framework defined by Stochastic Differential Equations (SDEs). The core
of this framework is the construction of a diffusion process that
reversibly transforms a complex data distribution into a simple prior
distribution (typically Gaussian noise). From the unified perspective of
SDEs, this section analyzes three principal designs for the diffusion
path: the **Variance Preserving SDE (VP-SDE)** and the **Variance
Exploding SDE (VE-SDE)**. It also explores their connection to classical
**Langevin Dynamics**, aiming to reveal their commonalities and
differences in terms of modeling paradigms, sampling strategies, and the
application of score functions.

### VP-SDE: Variance Preserving Diffusion Modeling

The design of the VP-SDE aims to ensure that during the diffusion
process, the variance of the marginal data distribution remains within a
finite range. It is inspired by DDPM (Denoising Diffusion Probabilistic
Models) (Ho, Jain, and Abbeel 2020).

#### Forward Process

The VP-SDE is defined by a linear SDE of the following form:
``` math
\mathrm{d}X_t = -\frac{1}{2} \beta(t) X_t \, \mathrm{d}t + \sqrt{\beta(t)} \, \mathrm{d}W_t, \quad t \in [0, T]
\label{eq:vp_sde_forward}
```
where $`\beta(t) > 0`$ is a predefined, monotonically increasing noise
schedule function. This linear SDE has an analytical solution, and its
transition kernel $`p_t(x_t|x_0)`$, given initial data $`x_0`$, is a
Gaussian distribution:
``` math
p_t(x_t|x_0) = \mathcal{N}\left(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) \mathbf{I}\right)
```
where
$`\bar{\alpha}_t = \exp\left(-\int_0^t \beta(s) \mathrm{d}s\right)`$.
This formulation ensures that as $`t`$ goes from $`0 \to T`$, the
influence of the initial data $`x_0`$ gradually diminishes while the
influence of the noise gradually increases, such that $`p_T(x_T)`$
ultimately approximates a standard normal distribution,
$`\mathcal{N}(0, \mathbf{I})`$.

#### Reverse Process

According to the theory of stochastic processes by Anderson (1982), the
reverse-time SDE corresponding to
Eq. <a href="#eq:vp_sde_forward" data-reference-type="eqref"
data-reference="eq:vp_sde_forward">[eq:vp_sde_forward]</a> exists and
has the following form:
``` math
\mathrm{d}X_t = \left[ -\frac{1}{2} \beta(t) X_t - \beta(t) \nabla_{x_t} \log p_t(x_t) \right] \mathrm{d}t + \sqrt{\beta(t)} \, \mathrm{d}\bar{W}_t
\label{eq:vp_sde_reverse}
```
where $`\mathrm{d}t`$ is an infinitesimal negative time step, and
$`\mathrm{d}\bar{W}_t`$ is a reverse-time Wiener process. The drift term
of this reverse process is composed of two parts: the drift term from
the forward process, $`f(x,t) = -\frac{1}{2}\beta(t)x_t`$, and a
correction term based on the score function,
$`-g(t)^2 \nabla_{x_t} \log p_t(x_t)`$. The core of the generative
process is to learn a parameterized network $`s_\theta(x_t, t)`$ to
accurately estimate the true score function.

#### Discretized Sampling

To generate samples from this model, we need to numerically discretize
the reverse SDE from
Eq. <a href="#eq:vp_sde_reverse" data-reference-type="eqref"
data-reference="eq:vp_sde_reverse">[eq:vp_sde_reverse]</a>. This is
typically done by starting at $`t=T`$ and stepping backward to $`t=0`$.
Applying the Euler-Maruyama method, a single update step from time
$`t_i`$ to $`t_{i-1}`$ can be expressed as:
``` math
x_{i-1} = x_i - \left[ -\frac{1}{2}\beta(t_i)x_i - \beta(t_i)s_\theta(x_i, t_i) \right]\Delta t + \sqrt{\beta(t_i)\Delta t} \cdot \varepsilon_i
```
where $`\Delta t = t_i - t_{i-1} > 0`$ is a small positive time step,
and $`\varepsilon_i \sim \mathcal{N}(0, \mathbf{I})`$. After
rearranging, this sampling format corresponds to the predictor step of
many Predictor-Corrector samplers.

### VE-SDE: Variance Exploding Diffusion Modeling

The design of the VE-SDE (Variance Exploding SDE) originates from early
score-based generative models. It is characterized by the fact that
during the diffusion process, the variance of the data distribution
continuously increases, eventually covering the entire space.

#### Forward Process

The VE-SDE is an SDE with no drift term, defined as follows:
``` math
\mathrm{d}X_t = \sigma(t) \, \mathrm{d}W_t, \quad t \in [0, T]
```
where $`\sigma(t)`$ is a diffusion coefficient function that increases
with time. The transition kernel $`p_t(x_t|x_0)`$ of this SDE is also a
Gaussian distribution, but its mean remains unchanged:
``` math
p_t(x_t|x_0) = \mathcal{N}\left(x_t; x_0, \left[\int_0^t \sigma^2(s)\mathrm{d}s\right] \mathbf{I}\right)
```
As $`t \to T`$, the variance term
$`\int_0^t \sigma^2(s)\mathrm{d}s \to \infty`$, hence the name "Variance
Exploding."

#### Reverse Process

The corresponding reverse-time SDE does not contain a structural drift
term and has the form:
``` math
\mathrm{d}X_t = - \sigma(t)^2 \nabla_{x_t} \log p_t(x_t) \, \mathrm{d}t + \sigma(t) \, \mathrm{d}\bar{W}_t
\label{eq:ve_sde_reverse}
```
The generative process likewise depends on an accurate estimation of the
score function, $`\nabla_{x_t} \log p_t(x_t)`$.

#### Discretized Sampling

Applying the Euler-Maruyama method to
Eq. <a href="#eq:ve_sde_reverse" data-reference-type="eqref"
data-reference="eq:ve_sde_reverse">[eq:ve_sde_reverse]</a> yields the
following discrete sampling update rule:
``` math
x_{i} = x_{i+1} + \sigma(t_{i+1})^2 s_\theta(x_{i+1}, t_{i+1}) |\Delta t| + \sqrt{\sigma(t_{i+1})^2 |\Delta t|} \cdot \varepsilon_i
```
where $`s_\theta`$ is the trained score network and
$`\Delta t = t_i - t_{i+1} > 0`$.

### Langevin Dynamics as a Unifying Perspective

Although the reverse processes of VP-SDE and VE-SDE differ in form, they
can both be viewed as generalizations of classical Langevin Dynamics
under different settings.

#### Classical Langevin Dynamics

Langevin dynamics originates from statistical physics and is used to
sample from a given probability density function $`p(x)`$. Its SDE form
is:
``` math
\mathrm{d}X_t = \frac{1}{2}\nabla_x \log p(x) \, \mathrm{d}t + \mathrm{d}W_t
```
The stationary distribution of this process is the target distribution
$`p(x)`$. Its discrete form, the Langevin sampling algorithm, iterates
as follows:
``` math
x_{k+1} = x_k + \frac{\eta}{2} \nabla_x \log p(x_k) + \sqrt{\eta} \cdot \varepsilon_k
```
In score-based generative models, iteratively applying Langevin sampling
steps at different noise scales constitutes the Annealed Langevin
Sampling algorithm.

#### Structural Mapping and Unification

We can uniformly represent the reverse processes of both VP-SDE and
VE-SDE as a form of generalized Langevin dynamics. A generalized
overdamped Langevin SDE can be written as:
``` math
\mathrm{d}X_t = -\nabla_x U(x_t, t) \, \mathrm{d}t + \sqrt{2D(t)} \, \mathrm{d}W_t
```
where $`U(x,t)`$ is a time-dependent potential energy function, and
$`D(t)`$ is a time-varying diffusion constant. For a probability density
$`p_t(x)`$, the relationship between the potential and the density is
given by $`\nabla_x \log p_t(x) = -\frac{1}{2D(t)}\nabla_x U(x, t)`$.

#### Equivalence of VE-SDE and Langevin Dynamics

The reverse process of the VE-SDE,
Eq. <a href="#eq:ve_sde_reverse" data-reference-type="eqref"
data-reference="eq:ve_sde_reverse">[eq:ve_sde_reverse]</a>,
``` math
\mathrm{d}X_t = - \sigma(t)^2 \nabla_{x_t} \log p_t(x_t) \, \mathrm{d}t + \sigma(t) \, \mathrm{d}\bar{W}_t
```
can be seen as a time-scaled Langevin dynamics. Its drift term is guided
entirely by the score function and contains no additional structural
drift.

#### Relationship between VP-SDE and Langevin Dynamics

The reverse process of the VP-SDE,
Eq. <a href="#eq:vp_sde_reverse" data-reference-type="eqref"
data-reference="eq:vp_sde_reverse">[eq:vp_sde_reverse]</a>,
``` math
\mathrm{d}X_t = \left[ -\frac{1}{2} \beta(t) X_t - \beta(t) \nabla_{x_t} \log p_t(x_t) \right] \mathrm{d}t + \sqrt{\beta(t)} \, \mathrm{d}\bar{W}_t
```
can be decomposed into two parts: a structural drift term,
$`f(x,t) = -\frac{1}{2}\beta(t)x_t`$, and a score-guided term.
Therefore, the VP-SDE can be regarded as a generalized Langevin dynamics
with an additional structural drift.

<div id="tab:sde_comparison">

| **Model** | **Structural Drift $`f(x,t)`$** | **Diffusion Coefficient $`g(t)`$** | **Score-Guided Term** |
|:---|:--:|:--:|:--:|
| Langevin Dynamics | $`0`$ | $`\sqrt{2D}`$ | $`\nabla_x \log p(x)`$ |
| VE-SDE (Reverse) | $`0`$ | $`\sigma(t)`$ | $`- \sigma(t)^2 \nabla_x \log p_t(x)`$ |
| VP-SDE (Reverse) | $`-\frac{1}{2} \beta(t) x`$ | $`\sqrt{\beta(t)}`$ | $`- \beta(t) \nabla_x \log p_t(x)`$ |

Structural Mapping of Model Reverse Processes to Langevin Dynamics

</div>

<span id="tab:sde_comparison" label="tab:sde_comparison"></span>

## Removing Randomness: Degeneration from SDE to the Probability Flow ODE

The numerical discretization of an SDE, such as the Euler-Maruyama
method, has the following iterative formula:
``` math
x_{i+1} = x_i + \mu(x_i, t_i)h + \sigma(t_i)\sqrt{h} \cdot \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, \mathbf{I})
```
The presence of the stochastic term
$`\sigma(t_i)\sqrt{h} \cdot \varepsilon_i`$ causes the sampling
trajectories of the SDE to exhibit random fluctuations. To ensure
numerical stability and accurately approximate the true solution, the
time step $`h`$ must be sufficiently small, which leads to a large
number of sampling steps and consequently, low generation efficiency.

To overcome this limitation, an equivalent deterministic process can be
sought, whose trajectories are described by an Ordinary Differential
Equation (ODE), yet which generates marginal probability density
functions, $`p_t(x)`$, identical to those of the original SDE. This
implies that although the individual trajectories of the ODE are
deterministic, if a large number of points are sampled from the entire
prior distribution, $`p_T(x)`$, and evolved according to the ODE, their
spatial distribution at any given time $`t`$ will be identical to the
particle ensemble distribution resulting from the SDE’s evolution. This
equivalent ODE is known as the **Probability Flow ODE**:

``` math
\begin{aligned}
\frac{\partial p(x, \tau)}{\partial \tau} 
&= \nabla_x \cdot [f(x, \tau) p(x, \tau)] - \frac{g(\tau)^2}{2} \nabla_x^2 p(x, \tau)\\
&= \nabla_x \cdot [f(x, \tau) p(x, \tau)]- \frac{g(\tau)^2}{2} \nabla_x \cdot (\nabla_x p(x, \tau))\\
&=\nabla_x \cdot \left[f(x, \tau) p(x, \tau) - \frac{g(\tau)^2}{2} \nabla_x p(x, \tau) \right]\\
&=\nabla_x \cdot \left[ f(x, \tau) p(x, \tau) - \frac{g(\tau)^2}{2} \left(p(x, \tau) \cdot \nabla_x \log p(x, \tau) \right) \right]\\
&=- \nabla_x \cdot \left[ \left(-f(x, \tau) + \frac{g(\tau)^2}{2} \nabla_x \log p(x, \tau) \right) p(x, \tau) \right]
\end{aligned}
```

This equation describes the probability density evolution of a
deterministic process. By comparing this to the form of the FPE without
a stochastic term, Eq. <a href="#eq:ode_FPE" data-reference-type="ref"
data-reference="eq:ode_FPE">[eq:ode_FPE]</a>, we can identify the
corresponding velocity field (or drift term) for the ODE as:
``` math
\mu(x,t) = f(x, t) - \frac{1}{2}g(t)^2 \nabla_x \log p(x, t)
```
Therefore, the Probability Flow ODE that shares the same marginal
probability densities as the forward SDE is:
``` math
\mathrm{d}x = \left[ f(x, t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x) \right] \mathrm{d}t
```
In generative modeling, we use a neural network $`s_\theta(x,t)`$ to
approximate the score function $`\nabla_x \log p_t(x)`$. By integrating
the above ODE backward in time (from $`T`$ to $`0`$), we can
deterministically generate data from noise.

<div class="algorithm">

<div class="algorithmic">

Learned drift $`\mu_\theta(x,t)`$, diffusion schedule $`\sigma(t)`$,
number of steps $`N`$, total time $`T`$ $`h \gets T / N`$ Sample initial
state $`x_0 \sim \mathcal{N}(0, \mathbf{I})`$ $`t_i \gets i \cdot h`$
Sample $`\varepsilon_i \sim \mathcal{N}(0, \mathbf{I})`$
$`x_{i-1} \gets x_i + \mu_\theta(x_i, t_i)h`$ $`x_N`$

</div>

</div>

The deterministic nature of the Probability Flow ODE enables efficient,
high-order numerical solvers (e.g., Runge-Kutta) that can generate
high-quality samples in very few steps, forming the basis for fast
algorithms like DDIM (Song, Meng, and Ermon 2020). This reveals a
profound SDE-ODE duality: although both processes share the same
macroscopic evolution of marginal densities$`p_t(x)`$, their microscopic
paths differ fundamentally. SDE trajectories are stochastic and
irregular, while ODE trajectories are smooth and unique, a contrast
visualized in Figure <a href="#fig:sde_vs_ode" data-reference-type="ref"
data-reference="fig:sde_vs_ode">1</a>.

<figure id="fig:sde_vs_ode">
<img src="figures/sde.png" style="width:70.0%" />
<figcaption> <strong>Comparison of SDE and Probability Flow ODE Paths.
(Song et al., 2021)</strong> This figure illustrates the forward process
from the data distribution <span
class="math inline"><em>p</em><sub>0</sub>(<em>x</em>)</span> to the
prior distribution <span
class="math inline"><em>p</em><sub><em>T</em></sub>(<em>x</em>)</span>,
as well as the reverse process of recovering data from the prior. The
background heatmap represents the spatiotemporal evolution of the
marginal probability density <span
class="math inline"><em>p</em><sub><em>t</em></sub>(<em>x</em>)</span>.
The colorful, jagged curves in the figure are random sample paths
generated by an SDE solver (e.g., the Euler-Maruyama method), while the
smooth white curves are the deterministic paths generated by the
corresponding Probability Flow ODE. The ODE paths evolve precisely along
the ’ridges’ or peak regions of the probability density. </figcaption>
</figure>

# Flow-Based Models

After an in-depth exploration of diffusion models, which are centered on
stochastic processes, this section shifts its focus to another powerful
generative modeling paradigm: **Flow-Based Models**. Unlike diffusion
models, which connect data and noise by simulating a stochastic process,
flow-based models aim to learn a deterministic and invertible
transformation. This transformation can precisely map a simple prior
distribution (e.g., a Gaussian) to a complex target data distribution.

This process can be conceptualized as learning a vector field defined by
an Ordinary Differential Equation (ODE). Guided by this vector field,
data points, like massless particles, smoothly ’flow’ from a simple
region of space to the complex region where the data manifold resides.
This chapter will begin with classical **Normalizing Flows**,
progressively reveal their inherent limitations, and show how they
naturally evolve into the ODE-based **Continuous Normalizing Flows**,
ultimately leading to a more training-efficient modern paradigm: **Flow
Matching**.

## The Classic Paradigm: Normalizing Flows

Normalizing Flows (NF) (Rezende and Mohamed 2015) are a foundational
work in the flow-based modeling paradigm. Their core mechanism is to map
a complex data distribution $`p_X(x)`$ to a simple base distribution
$`p_Z(z)`$ through a series of invertible transformations
$`f_1, f_2, \dots, f_M`$; a process that is both mathematically clear
and elegant.

Their mathematical foundation is the change of variables formula from
probability theory. For an invertible transformation $`z = f(x)`$, the
relationship between the probability densities is:
``` math
p_X(\boldsymbol{x}) = p_Z(f(\boldsymbol{x})) \left| \det\left( \frac{\partial f(\boldsymbol{x})}{\partial \boldsymbol{x}} \right) \right|
```
where $`\frac{\partial f(x)}{\partial x}`$ is the Jacobian matrix of the
transformation, and the absolute value of its determinant,
$`|\det(\cdot)|`$, measures the stretching or compression of an
infinitesimal volume element by the transformation. This factor is the
critical ’normalizing’ term that ensures the transformed probability
density function still integrates to 1.

Because the expressive power of a single transformation is limited, NFs
typically concatenate multiple simple, invertible transformations
$`f_i`$ to form a ’flow’. The calculation of its log-likelihood depends
on the sum of the log-determinants of the Jacobians:
``` math
\log p_X(x) = \log p_Z(f_M(\dots f_1(x)\dots)) + \sum_{i=1}^{M} \log \left| \det\left( \mathbf{J}_{f_i} \right) \right|
```

Behind the simplicity of this formula lie the core design constraints of
NF models. To ensure the model is computationally feasible, each
transformation layer $`f_i`$ must be carefully designed to
simultaneously satisfy two demanding conditions:

1.  **Invertibility**: The transformation must be bijective.

2.  **Efficient Jacobian Determinant**: The computation of the Jacobian
    determinant, $`|\det(\mathbf{J}_{f_i})|`$, must be efficient (e.g.,
    with a computational complexity of $`\mathcal{O}(D)`$ instead of
    $`\mathcal{O}(D^3)`$).

The second condition severely restricts the available neural network
architectures and has given rise to models such as RealNVP and Glow,
which use specially designed coupling layers. However, this structural
compromise, made for the sake of computational efficiency, undoubtedly
also limits the model’s expressive power.

## From Discrete to Continuous: Continuous Normalizing Flows

A solution to the aforementioned limitations lies in reconceptualizing
the nature of the transformation from a discrete hierarchy to a
continuous process. If we imagine an NF with an infinite number of
layers, where each layer applies only an infinitesimal transformation to
the data, then the limit of this process is a continuous-time flow
parameterized by time. This is the core idea behind **Continuous
Normalizing Flows (CNF)**.

In the CNF framework, the transformation is no longer defined by a
discrete sequence of functions $`\{f_i\}`$, but is instead described by
a continuous trajectory $`z(t)`$ governed by an Ordinary Differential
Equation (ODE):
``` math
\frac{\mathrm{d}z(t)}{\mathrm{d}t} = f(z(t), t, \theta), \quad \text{with initial condition } z(0) = x
```
From this perspective, the neural network $`f`$ no longer directly
defines a transformation, but instead defines the instantaneous velocity
vector field for a data point at any position $`z(t)`$ and time $`t`$.

This change in perspective brings a critical breakthrough for likelihood
computation. In the continuous limit, the sum of the log-determinants of
the Jacobians in the discrete NF log-likelihood formula becomes an
integral of the **trace** of the Jacobian matrix. According to the
instantaneous change of variables formula, the log-likelihood of a CNF
can be expressed as:
``` math
\log p_X(x) = \log p_Z(z(T)) - \int_0^T \text{Tr}\left(\frac{\partial f(z(t), t)}{\partial z(t)}\right) \mathrm{d}t
```
The shift from "determinant" to "trace" is significant because it
greatly relaxes the constraints on the neural network architecture (the
computational complexity of the trace is typically $`\mathcal{O}(D)`$).
However, this theoretical elegance shifts the computational burden to
the training phase. To evaluate the likelihood of any data point $`x`$,
two computationally expensive operations must be performed:

1.  Starting from $`z(0) = x`$, integrate the ODE fully using a
    numerical solver to obtain the final state $`z(T)`$.

2.  While solving the ODE, one must also compute and integrate the trace
    of the Jacobian along the entire trajectory.

These two steps make the training process for CNFs slow, prompting
researchers to seek new paradigms that can both retain the powerful
expressiveness of the ODE framework and achieve efficient training.

## A Modern Paradigm for Efficient Training: Flow Matching

The idea behind Flow Matching (FM) (Lipman et al. 2022) is highly
insightful: instead of indirectly optimizing the vector field by
computing the probability densities at the two ends of a trajectory, we
directly learn a vector field $`u_t^\theta(x)`$ to construct a
deterministic flow via an ODE that moves from a simple noise
distribution to a complex data distribution:
``` math
x_0\sim p_0, \quad \frac{dx_t}{dt}=u_t^\theta(x)
```

Assume there exists an ideal vector field $`u_t(x)`$ that can transport
the source distribution $`p_0`$ to the target data distribution $`p_1`$.
The objective of Flow Matching is to train a neural network
$`u_\theta(\boldsymbol{x}, t)`$ to directly regress this target field:
``` math
L(\theta) = \mathbb{E}[||u_t^{\theta}(x) - u_t^{\text{true}}(x)||^2]
```

However, the true marginal vector field $`u_t^{\text{true}}(x)`$ and the
marginal probabilities along the path, $`p_t^{\text{true}}(x)`$, are
both unknown.

### Conditional Flow Matching

We encountered a similar challenge in diffusion models: the marginal
score function $`\nabla_{x_t}\log p_t(x_t)`$ was intractable, so we
instead used the tractable conditional score function
$`\nabla_{x_t}\log p_t(x_t|x_0)`$ and proved the identity in
Eq. <a href="#eq:score_identity_revised" data-reference-type="eqref"
data-reference="eq:score_identity_revised">[eq:score_identity_revised]</a>.
Here, we will adopt the exact same philosophical approach: we derive the
marginal vector field $`u_t^{\text{true}}(x)`$ by constructing a
conditional vector field $`u_t^{\text{true}}(x|z)`$:

``` math
\begin{aligned}
u_t^{\text{target}}(x) 
&= \int u_t^{\text{target}}(x|z) p_t(z|x) \, \mathrm{d}z \\
&= \int u_t^{\text{target}}(x|z) \frac{p_t(x|z) p^{\text{data}}(z)}{p_t(x)} \, \mathrm{d}z\\
&=\mathbb{E}_{z \sim p_t(z|x)} \left[ u_t^{\text{target}}(x \mid z) \right]
\end{aligned}
```
By definition, for any probability flow generated by the vector field
$`u_t(x)`$, its probability density $`p_t(x)`$ must satisfy the
continuity equation:

``` math
\begin{aligned}
\frac{\partial p_t(x)}{\partial t} 
&= -\mathrm{div} \left( p_t(x) \cdot u_t^{\text{target}}(x) \right) \\
&= -\mathrm{div} \left( \int p_t(x|z) p_{\text{data}}(z) u_t^{\text{target}}(x|z) \, dz \right) \\
&= -\mathrm{div} \left( \int p_t(x|z) u_t^{\text{target}}(x|z) p_{\text{data}}(z) \, dz \right) \\
&= -\mathrm{div} \left( p_t(x) \int u_t^{\text{target}}(x|z) \frac{p_t(x|z) p_{\text{data}}(z)}{p_t(x)} \, dz \right) 
\end{aligned}
```
即
``` math
u_t^{\text{target}}(x) = \int u_t^{\text{target}}(x|z) \frac{p_t(x|z) p_{\text{data}}(z)}{p_t(x)} \, dz=\mathbb{E}_{z \sim p_t(z|x)} \left[ u_t^{\text{target}}(x|z) \right]\label{eq:33}
```

This is a classic posterior expectation form, which indicates that at a
given position $`x`$ and time $`t`$, the marginal vector field
$`u_t^{\text{target}}(x)`$ can be viewed as the expectation of the
conditional vector field $`u_t^{\text{target}}(x|z)`$ under the
posterior distribution $`p_t(z|x)`$. This relationship is highly
significant: it shows that although the true marginal vector field
$`u_t(x)`$ cannot be obtained explicitly, we can design conditional path
distributions $`p_t(x|z)`$ that are easy to sample from and compute.
Combined with training data $`z \sim p_{\text{data}}(z)`$, this allows
us to estimate this expectation via Monte Carlo methods during training,
thereby fitting the marginal vector field.

Furthermore, we note that this formula essentially originates from
Bayes’ theorem:
``` math
p_t(z \mid x) = \frac{p_t(x \mid z) \cdot p_{\text{data}}(z)}{p_t(x)}
```

Therefore, the expression for the marginal vector field can also be
interpreted as follows: the marginal vector field is a weighted average
of the conditional vector fields $`u_t(x|z)`$ defined by all possible
target states $`z`$, where the weights are proportional to the
likelihood that state $`z`$ generated the current point $`x`$.

Furthermore, based on the core identity from
Eq. <a href="#eq:33" data-reference-type="eqref"
data-reference="eq:33">[eq:33]</a>, a theoretically ideal objective
would be to train a parameterized vector field network
$`u_\theta(x, t)`$ to directly regress the true marginal vector field
$`u_t(x)`$. This can be formalized as minimizing the following loss
function:
``` math
\mathcal{L}_{\text{marginal}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}(0,1), x \sim p_t(x)} \left[ \left\| u_\theta(x, t) - u_t(x) \right\|^2 \right]
\label{eq:35}
```
However, this objective function is computationally intractable. The
fundamental reason is that the target,
$`u_t(x) = \mathbb{E}_{z \sim p_t(z|x)} [u_t(x|z)]`$, is an expectation
under the unknown posterior probability distribution $`p_t(z|x)`$, which
in turn depends on the unknown data prior $`p_{\text{data}}(z)`$ and the
intractable marginal distribution $`p_t(x)`$.

To construct a tractable learning objective, **Conditional Flow Matching
(CFM)** proposes to instead regress the more tractable **conditional
vector field** $`u_t(x|z)`$. Based on the principle of optimal
estimation (see
Appendix <a href="#appendix:optimal_estimation" data-reference-type="ref"
data-reference="appendix:optimal_estimation">7</a>), its loss function
is defined as:
``` math
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}(0,1), z \sim p_{\text{data}}(z), x \sim p_t(x|z)} \left[ \left\| u_\theta(x, t) - u_t(x|z) \right\|^2 \right]
\label{eq:36}
```
Alternatively, the equivalence of
Eq. <a href="#eq:35" data-reference-type="ref"
data-reference="eq:35">[eq:35]</a> and
Eq. <a href="#eq:36" data-reference-type="ref"
data-reference="eq:36">[eq:36]</a> can be proven from a gradient
optimization perspective, as shown in
Appendix <a href="#appendix:gradient" data-reference-type="ref"
data-reference="appendix:gradient">8</a>.

## Gaussian Probability Paths

To concretely illustrate the conditional vector field and its induced
probability paths, we construct a special case where the conditional
path $`p_t(x|x_1)`$ follows a Gaussian distribution:
``` math
p_t(x_t | x_1) = \mathcal{N}(x_t; \mu_t(x_1), \sigma_t(x_1)^2 \mathbf{I})
\label{eq:37}
```
where $`\mu_t(x_1)`$ and $`\sigma_t(x_1)`$ are the conditional mean and
standard deviation of the path at time $`t`$, respectively. A
conditional vector field $`u_t(x|x_1)`$ capable of generating this
Gaussian path has the following analytical form (for a proof, see
Appendix <a href="#appendix:conditional vector field" data-reference-type="ref"
data-reference="appendix:conditional vector field">9</a>):
``` math
u_t(x | x_1) = \frac{\dot{\sigma}_t(x_1)}{\sigma_t(x_1)} (x - \mu_t(x_1)) + \dot{\mu}_t(x_1)
\label{eq:general_gaussian_vf}
```

#### Linear Interpolation

We can choose a simple functional form for the mean and standard
deviation of this Gaussian path, namely, linear interpolation:
``` math
\begin{aligned}
\mu_t(x_1) &\triangleq tx_1 \\
\sigma_t(x_1) &\triangleq (1-t) + t\sigma_{\min}
\end{aligned}
```
Their corresponding time derivatives are $`\dot{\mu}_t(x_1) = x_1`$ and
$`\dot{\sigma}_t(x_1) = -1 + \sigma_{\min}`$. Substituting these terms
into Eq. <a href="#eq:general_gaussian_vf" data-reference-type="eqref"
data-reference="eq:general_gaussian_vf">[eq:general_gaussian_vf]</a>
yields the expression for the conditional vector field under this
specific linear interpolation path:
``` math
\begin{aligned}
u_t(x \mid x_1) 
&= \frac{-(1 - \sigma_{\min})}{1 - (1 - \sigma_{\min})t} (x - tx_1) + x_1 \\
&= \frac{1}{(1 - t) + t \sigma_{\min}} \left( - (1 - \sigma_{\min})(x - tx_1) + \left(1 - (1 - \sigma_{\min})t \right)x_1 \right) \\
&= \frac{1}{(1 - t) + t \sigma_{\min}} \left( - (1 - \sigma_{\min})x + x_1 \right) \\
&= \frac{x_1 - (1 - \sigma_{\min})x}{1 - (1 - \sigma_{\min})t}
\end{aligned}
```

To intuitively demonstrate the difference in paths generated by the
conditional vector field $`u_t(x|x_1)`$ and its corresponding marginal
vector field $`u_t(x)`$, we construct a Gaussian-to-ring transformation
task. Let the initial distribution be a Gaussian distribution centered
at the origin, $`p_0 = \mathcal{N}(\mathbf{0}, \sigma_0^2\mathbf{I})`$,
and the target distribution be a ring distribution centered at the
origin with radius $`R`$, denoted as $`p_1`$.

According to the core identity of Flow Matching, at any intermediate
time $`t`$ and any position $`x_t`$, the marginal vector field
$`u_t(x_t)`$ is the expectation of all possible conditional vector
fields $`u_t(x_t|x_1)`$ under the posterior probability
$`p_t(x_1|x_t)`$. However, since the target point $`x_1`$ is not
uniquely determined, we obtain the marginal vector field by weighting
all possible conditional vector fields $`u_t(x \mid x_1)`$ by the
Bayesian posterior $`p_t(x_1 \mid x)`$:
``` math
\begin{aligned}
u_t\left(\phi_t(x_0)\right) &= \mathbb{E}_{x_1 \sim p_t(x_1|x_t)} \left[ u_t\left( x_t \mid x_1 \right) \right]\\
&\approx \frac{1}{n} \sum_{i=1}^n u_t\left(x_t \mid x_1^{(i)}\right) 
\quad \text{with } x_1^{(i)} \sim p_{1|t}\left(x_1 \mid x_t\right).    
\end{aligned}
```

<figure id="fig:g2g_paths">
<img src="figures/1&amp;2.png" style="width:80.0%" />
<figcaption> <strong>Visual comparison of conditional and marginal paths
in the Gaussian-to-Ring example.</strong> <strong>(Left)</strong>
Conditional paths <span
class="math inline"><em>ϕ</em><sub><em>t</em></sub>(<em>x</em><sub>0</sub>|<em>x</em><sub>1</sub>)</span>
driven by the conditional vector field <span
class="math inline"><em>u</em><sub><em>t</em></sub>(<em>x</em>|<em>x</em><sub>1</sub>)</span>.
Each path corresponds to a linear interpolation between a starting point
<span class="math inline"><em>x</em><sub>0</sub></span> randomly sampled
from <span class="math inline"><em>p</em><sub>0</sub></span> and an
endpoint <span class="math inline"><em>x</em><sub>1</sub></span>
randomly sampled from <span
class="math inline"><em>p</em><sub>1</sub></span>. Due to the random
pairing, the paths exhibit significant crossing.
<strong>(Right)</strong> Marginal paths <span
class="math inline"><em>ϕ</em><sub><em>t</em></sub>(<em>x</em><sub>0</sub>)</span>
driven by the marginal vector field <span
class="math inline"><em>u</em><sub><em>t</em></sub>(<em>x</em>)</span>.
These paths are the result of averaging over all possible endpoints
<span class="math inline"><em>x</em><sub>1</sub></span>, weighted by
their posterior probabilities. The paths are smooth and do not
intersect, which is consistent with the uniqueness theorem for solutions
to Ordinary Differential Equations (ODEs). </figcaption>
</figure>

However, to reveal at a more microscopic level how the smoothness of the
marginal vector field $`u_t(x)`$ arises from statistical averaging, it
is necessary to examine its instantaneous dynamical properties. By its
definition, $`u_t(x) = \mathbb{E}_{p_t(x_1|x_t)}[u_t(x|x_1)]`$, the
properties of this vector field at each moment in time $`t`$ are
determined by the posterior distribution $`p_t(x_1|x_t)`$. Predictably,
as the particle’s state $`x_t`$ approaches the target distribution
$`p_1`$, the posterior distribution $`p_t(x_1|x_t)`$ will contract
significantly.
Figure <a href="#fig:cfm-dynamics" data-reference-type="ref"
data-reference="fig:cfm-dynamics">3</a> aims to clarify this phenomenon
of posterior contraction by visualizing the vector fields at two
different moments in time, revealing how it leads to a reduction in the
variance of the conditional vector fields and thereby enhances the
stability of the marginal vector field.

<figure id="fig:cfm-dynamics">
<img src="figures/3.png" style="width:50.0%" />
<figcaption> A comparison of the marginal vector field <span
class="math inline"><em>u</em><sub><em>t</em></sub>(<em>x</em>)</span>
and the conditional vector fields <span
class="math inline"><em>u</em><sub><em>t</em></sub>(<em>x</em> ∣ <em>x</em><sub>1</sub>)</span>
at different time steps, visualizing the effect of posterior
distribution contraction on the stability of the vector field.
</figcaption>
</figure>

## Coupling

An intrinsic challenge arises when training Conditional Flow Matching:
for small values of $`t`$, the conditional vector field $`u_t(x_t|x_1)`$
constructed by the default **independent coupling** strategy exhibits
high variance, leading to training instability.

The root cause is that independent coupling pairs a starting point
$`x_0 \sim p_0`$ with an endpoint $`x_1 \sim p_1`$ while completely
disregarding their geometric relationship. When the two distributions
are geometrically mismatched (e.g., a Gaussian to a ring distribution),
this strategy generates a multitude of chaotic and conflicting
conditional paths. This ultimately results in high variance in the loss
function’s gradient estimates, thereby impeding the training process.

Therefore, a natural idea is to design a more intelligent coupling
scheme to fundamentally reduce the intrinsic variance of the conditional
vector fields. This constitutes the core of this section. We will
explore the evolution from simple independent coupling to the
geometry-aware **Optimal Transport Coupling**, aiming to construct more
orderly and less conflicting conditional paths, thereby significantly
improving training stability and efficiency.

#### One-Sided Conditioning

An initial and more direct method for constructing the vector field is
to condition on data points $`x_1`$ sampled from the target distribution
and subsequently marginalize over this distribution. This paradigm is
known as one-sided conditioning.

The marginal probability path $`p_t(x_t)`$ is defined by marginalizing
the conditional probability path $`p_t(x_t | z)`$ over the latent
variable $`z=x_1`$, where $`x_1`$ is sampled from the data distribution
$`p_{\text{data}}(x_1)`$. This relationship can be expressed as:
``` math
\label{eq:one_sided_en}
p_t(x_t) = \int p_t(x_t | z) p_{\text{data}}(z) dz = \int p_t(x_t | x_1) p_{\text{data}}(x_1) dx_1
```
A typical example of such a conditional path is a time-varying Gaussian
distribution centered at the target point $`x_1`$. For instance:
``` math
p(x_t | x_1) = \mathcal{N}(x_t | x_1, (1-t)^2\mathbf{I})
```

#### Two-Sided Conditioning

A more general and powerful framework is to condition on a latent
variable $`z`$ that contains information about *both endpoints* of the
trajectory. This method is known as two-sided conditioning. This latent
variable is defined as the pair of points $`z = (x_1, x_0)`$.

Under this paradigm, the marginal probability path is defined by
marginalizing over the joint distribution, or coupling,
$`p_{\text{data}}(x_1, x_0)`$:
``` math
\label{eq:two_sided_en}
p_t(x_t) = \iint p_t(x_t | z) p_{\text{data}}(z) dz = \iint p_t(x_t | x_1, x_0) p_{\text{data}}(x_1, x_0) dx_1 dx_0
```

<figure id="fig:one_vs_two_sided_mol">
<div class="minipage">
<img src="figures/5.png" />
</div>
<div class="minipage">
<img src="figures/6.png" />
</div>
<figcaption> <strong>Comparison and Application Example of One-Sided vs.
Two-Sided Conditioning in CFM:</strong> <strong>(Left) One-Sided
Conditioning:</strong> The evolution of the path is guided entirely by
the final configuration <span
class="math inline"><em>x</em><sub>1</sub></span>, while its starting
point <span class="math inline"><em>x</em><sub>0</sub></span> is a
random noise distribution. The goal is to create an ordered structure
from chaos. <strong>(Right) Two-Sided Conditioning:</strong> The path is
a direct bridge connecting a specific initial state <span
class="math inline"><em>x</em><sub>0</sub></span> (e.g., an initial
guess based on the centroid) to the final state <span
class="math inline"><em>x</em><sub>1</sub></span>. Its stronger control
over the path’s geometry helps to achieve more stable training
dynamics.</figcaption>
</figure>

#### Optimal Transport Coupling

For a generative transformation between two probability distributions,
$`q_0(x_0)`$ and $`q_1(x_1)`$, the construction of the paths depends on
the choice of the joint distribution $`q(x_1, x_0)`$ between the
starting points $`x_0`$ and target points $`x_1`$. Mathematically, this
joint distribution is known as a probabilistic coupling.

In the simplest setting, one can assume that the selection of the source
and target samples is mutually independent, which constitutes an
independent coupling:
``` math
q(x_1, x_0) = q_1(x_1)q_0(x_0)
```
However, this strategy completely ignores the relative geometric
positions of the data points. When there are significant differences in
the geometric shapes of the source and target distributions, random
pairing can produce a large number of ’detouring’ or ’crossing’ paths,
which in turn introduce instability into the model’s training.

To construct more efficient paths, it is necessary to introduce a
**correlated coupling** that can reflect the geometric relationship
between the distributions, i.e., $`q(x_1, x_0) \neq q_1(x_1)q_0(x_0)`$.
Optimal Transport (OT) theory provides a solid mathematical foundation
for this. The goal of OT is to find a joint distribution
$`\pi(x_1, x_0)`$ that, among all possible pairing plans, minimizes the
total transportation cost between sample pairs $`(x_0, x_1)`$. For the
standard case using the squared Euclidean distance as the cost function,
this problem can be formulated as the following Monge-Kantorovich
problem:
``` math
q(x_1, x_0) = \pi(x_1, x_0) \in \underset{\pi \in \Pi(q_0, q_1)}{\arg\inf} \iint \|x_1 - x_0\|_2^2 \, d\pi(x_1, x_0)
```

where $`\Pi(q_0, q_1)`$ represents the set of all valid couplings whose
marginals are $`q_0`$ and $`q_1`$.

The solution to the above optimization problem is the optimal transport
coupling, and its minimized cost value is directly related to the
Squared 2-Wasserstein distance, $`W_2^2(q_0, q_1)`$. The coupling
$`\pi`$ obtained in this way is no longer random; instead, it precisely
matches each starting point $`x_0`$ with a geometrically ’optimal’
endpoint $`x_1`$. This geometry-based intelligent pairing can
significantly simplify the marginal paths that the model ultimately
needs to learn. It avoids unnecessary path crossings, thereby acting as
a form of regularization, reducing variance during training, and making
the learning process more stable and efficient.

#### Visualization of Coupling Strategies and Path Geometry

The preceding text introduced two core coupling strategies: independent
coupling, based on statistical independence, and optimal transport
coupling, based on geometric optimization. The choice of coupling
strategy fundamentally determines the geometry of the conditional paths,
which in turn directly affects the model’s training stability and
efficiency. To provide an intuitive illustration of this point,
Figure <a href="#fig:our_simulations" data-reference-type="ref"
data-reference="fig:our_simulations">5</a> uses three sets of numerical
simulations to compare the dynamic properties of paths under different
transformation tasks.

<figure id="fig:our_simulations">
<img src="figures/8.png" />
<figcaption> <strong>Schematic comparison of paths under three Flow
Matching frameworks:</strong> <strong>(Left)</strong> The probabilistic
paths of Flow Matching are highly overlapping; <strong>(Middle)</strong>
The randomly paired paths of Conditional Flow Matching intersect with
each other; <strong>(Right)</strong> The optimally paired paths of OT
Conditional Flow Matching are completely ordered and non-intersecting,
representing a more stable learning process. </figcaption>
</figure>

As shown in
Figure <a href="#fig:our_simulations" data-reference-type="ref"
data-reference="fig:our_simulations">5</a>, in the FM framework on the
left, the paths are probabilistic. A particle starting from one mode of
$`P_0`$ can end up in a superposition of any of the modes in $`P_1`$,
which leads to significant path crossing and overlap, increasing the
learning difficulty. The Conditional Flow Matching (CFM) with
independent coupling in the middle defines paths by randomly pairing
start and end points $`(x_0, x_1)`$. Although more direct than FM, the
problem of path crossing between modes still exists, potentially leading
to high training variance. In contrast, the OT Conditional Flow Matching
(OT-CFM) on the right employs the principle of Optimal Transport for
intelligent pairing, matching each starting point $`x_0`$ with an
optimal endpoint $`x_1`$. This results in highly ordered paths and
completely eliminates inter-modal crossing. Each mode flows
independently to its corresponding target, thereby effectively reducing
training variance and simplifying the learning process.

# Conclusion

This blog’s discourse begins with a deep insight into the randomness of
the physical world, starting from the physical intuition of **Brownian
motion** and introducing the core mathematical language to describe this
process—**Stochastic Differential Equations (SDEs) and Itô’s Lemma**.
This solid theoretical foundation directly gave rise to the first major
generative model paradigm: stochastic diffusion models. Under this
paradigm, models like **NCSN** and **DDPM** both utilize SDEs to
precisely simulate the stochastic process of data being gradually
corrupted by noise and then restored by a learned reverse SDE. However,
deep exploration of SDEs revealed their intrinsic connection to the
deterministic world, namely the profound **SDE-ODE duality**. This
theory points out that any SDE has an equivalent deterministic ODE
evolution path, a breakthrough insight that made it possible to design
deterministic samplers for stochastically trained models and directly
gave rise to efficient acceleration algorithms like **DDIM**.

Building on this, the blog’s perspective turns to another, completely
different technical route: **deterministic generative models**. Unlike
the former, which may only use an ODE during reverse sampling, this
class of models, such as **Continuous Normalizing Flows (CNF)**, is
governed by deterministic ODEs in both the forward and reverse
processes. To efficiently train these models, the article introduces the
advanced framework of **Flow Matching**. The article further explores
the core challenge in Flow Matching training: directly learning the
"marginal vector field," denoted as $`u_t(x)`$, is very difficult
because its statistical properties at different times (such as high
variance in the early stages) can lead to unstable training. Therefore,
the model instead learns the more tractable "conditional vector field,"
denoted as $`u_t(x|z)`$. To fundamentally solve the conditional field
variance problem caused by random pairing, the blog finally introduces
**Optimal Transport (OT)** theory. By using OT to intelligently match
start and end points, it is possible to construct geometrically ordered
and non-crossing paths, which greatly stabilizes the training process
and enhances the performance of models like **OT-CFM**.

<div class="appendices">

# The Wiener Process / Brownian Motion

A real-valued stochastic process $`\{W_t\}_{t \geq 0}`$ is called a
**Wiener process** (or **standard Brownian motion**) if it satisfies the
following conditions:

1.  **Initial Value:** $`W_0 = 0`$ (almost surely).

2.  **Independent Increments:** For any $`0 \leq s < t`$, the increment
    $`W_t - W_s`$ is independent of the history of the process before
    time $`s`$, $`\sigma(\{W_r\}_{0 \leq r \leq s})`$.

3.  **Gaussian Increments:** For any $`0 \leq s < t`$, the increment
    $`W_t - W_s`$ follows a normal distribution with a mean of zero and
    a variance of $`t - s`$, i.e.:
    ``` math
    W_t - W_s \sim \mathcal{N}(0, t - s)
    ```

4.  **Continuous Paths:** The sample paths $`t \mapsto W_t`$ are
    continuous almost everywhere.

From these properties, a key characteristic of the Wiener process can be
derived: its sample paths, while continuous, are nowhere differentiable
and have a non-zero **quadratic variation**. Specifically,
$`[W, W]_t = t`$. On an infinitesimal level, this corresponds to the
following heuristic relationship:
``` math
(\mathrm{d}W_t)^2 = \mathrm{d}t
\label{eq:quadratic_variation_en}
```
This relationship is the fundamental reason why stochastic calculus
differs from classical calculus. It shows that although
$`\mathrm{d}W_t`$ is itself an infinitesimal quantity (on the order of
$`O(\sqrt{\mathrm{d}t})`$), its square is of the same order as
$`\mathrm{d}t`$ and thus cannot be ignored in Taylor expansions.

# Itô’s Lemma and the Itô Integral

Because the paths of a Wiener process do not have bounded variation, the
classical Riemann-Stieltjes integral is not applicable. Kiyosi Itô
developed a new calculus theory to handle such processes. Its core
result is **Itô’s Lemma**, which provides the correct chain rule for
stochastic processes.

Consider a twice continuously differentiable function $`f(t, x)`$, and
let the stochastic process $`X_t`$ follow the SDE
$`\mathrm{d}X_t = \mu_t \mathrm{d}t + \sigma_t \mathrm{d}W_t`$. By
performing a Taylor expansion and applying the rule from
Eq. <a href="#eq:quadratic_variation_en" data-reference-type="ref"
data-reference="eq:quadratic_variation_en">[eq:quadratic_variation_en]</a>,
the general form of Itô’s Lemma can be obtained. For simplicity, we
consider a function that depends only on the Wiener process, $`f(W_t)`$.
Its infinitesimal change, $`\mathrm{d}f(W_t)`$, is:
``` math
\mathrm{d}f(W_t) = f(W_{t+\mathrm{d}t}) - f(W_t) \approx f'(W_t)\mathrm{d}W_t + \frac{1}{2}f''(W_t)(\mathrm{d}W_t)^2
```
Substituting $`(\mathrm{d}W_t)^2 = \mathrm{d}t`$, we obtain the basic
form of Itô’s Lemma:
``` math
\mathrm{d}f(W_t) = f'(W_t)\mathrm{d}W_t + \frac{1}{2}f''(W_t)\mathrm{d}t
\label{eq:ito_lemma}
```
The additional second-derivative term in this formula,
$`\frac{1}{2}f''(W_t)\mathrm{d}t`$, is the core difference between the
Itô integral and the classical integral.

We can use Itô’s Lemma to compute a classic **Itô Integral**
<a href="#eq:ito_integral_derivation" data-reference-type="ref"
data-reference="eq:ito_integral_derivation">[eq:ito_integral_derivation]</a>.
Let $`f(W_t) = W_t^2`$, then $`f'(W_t) = 2W_t`$ and $`f''(W_t) = 2`$.
According to the lemma in
Eq. <a href="#eq:ito_lemma" data-reference-type="eqref"
data-reference="eq:ito_lemma">[eq:ito_lemma]</a>, we have:
``` math
\begin{aligned}
    \int_0^T \mathrm{d}(W_s^2) &= \int_0^T 2W_s \mathrm{d}W_s + \int_0^T \mathrm{d}s \nonumber \\
    %
    W_T^2 - W_0^2 &= 2\int_0^T W_s \mathrm{d}W_s + T 
    W_T^2 &= 2\int_0^T W_s \mathrm{d}W_s + T \quad 
    \int_0^T W_s \mathrm{d}W_s &= \frac{1}{2}W_T^2 - \frac{1}{2}T
    \label{eq:ito_integral_derivation}
\end{aligned}
```

# Derivation of the Reverse SDE Formula

Given a Stochastic Differential Equation (SDE), its discretized form is
as follows:
``` math
x_{t+\Delta t} - x_t = f(x_t, t) \Delta t + g(t) \sqrt{\Delta t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
```

We can convert this into a conditional probability model:
``` math
x_{t+\Delta t} \mid x_t \sim \mathcal{N}(x_t + f(x_t, t) \Delta t, g^2(t) \Delta t)
```

Next, we consider the derivation of the conditional probability
$`p(x_t \mid x_{t+\Delta t})`$:
``` math
\begin{aligned}
p(x_t \mid x_{t+\Delta t}) &= \frac{p(x_{t+\Delta t} \mid x_t) p(x_t)}{p(x_{t+\Delta t})} \\
&= p(x_{t+\Delta t} \mid x_t) \exp(\log p(x_t) - \log p(x_{t+\Delta t})) \notag \\
&\approx p(x_{t+\Delta t} \mid x_t) \exp \Big\{ 
 - \left( x_{t+\Delta t} - x_t - f(x_t, t) \Delta t \right) \nabla_x \log p(x_t) \Delta t \notag \\
&\quad\quad\quad - \Delta t \frac{\partial}{\partial t} \log p(x_t) \Big\} \notag \\
&= \exp \Big\{ -\frac{1}{2 g^2(t) \Delta t} \Big\| (x_{t+\Delta t} - x_t) 
 - \left( f(x_t, t) - g^2(t) \nabla_x \log p(x_t) \right) \Delta t \Big\|^2 \notag \\
&\quad\quad - \Delta t \frac{\partial}{\partial t} \log p(x_t) 
 - \frac{f^2(x_t, t) \Delta t}{2 g^2(t)} \Big\} 
 + \frac{\left( f(x_t, t) - g^2(t) \nabla_x \log p(x_t) \right)^2 \Delta t}{2 g^2(t)} \notag \\
&\overset{\Delta t \to 0}{=} \exp \Bigg\{ \frac{1}{2 g^2(t + \Delta t) \Delta t} 
 \Big\| (x_{t+\Delta t} - x_t) 
 - \left( f(x_{t+\Delta t}, t + \Delta t) - g^2(t + \Delta t) \nabla_x \log p(x_{t+\Delta t}) \right) \Delta t \Big\|^2 \Bigg\} \notag
\end{aligned}
```

The resulting Gaussian distribution for the condition
$`x_t \mid x_{t+\Delta t}`$ has:

``` math
\mu = x_{t+\Delta t} - \left( f(x_{t+\Delta t}, t + \Delta t) - g^2(t + \Delta t) \nabla_{x_{t+\Delta t}} \log p(x_{t+\Delta t}) \right) \Delta t
```

``` math
\sigma^2 = g^2(t + \Delta t) \Delta t
```

Next, we obtain the corresponding SDE, which is expressed in its
discrete form as:
``` math
x_{t+\Delta t} - x_t = \left( f(x_{t+\Delta t}, t + \Delta t) - g^2(t + \Delta t) \nabla_{x_{t+\Delta t}} \log p(x_{t+\Delta t}) \right) \Delta t + g(t + \Delta t) \sqrt{\Delta t} \epsilon
```
In the limit, this corresponds to the continuous SDE:
``` math
dx = \left[f(x, t) - g(t)^2 \nabla_x \log p_t(x)\right] dt + g(t) d\bar{w}
```
where $`\epsilon \sim \mathcal{N}(0, I)`$ is a standard normal random
variable, and $`d\bar{w}`$ is a standard Wiener process.

# The Principle of Optimal Estimation and Its Application in Generative Models

In the training of probabilistic generative models, we often face a
challenge: the theoretically ideal learning targets (such as the true
marginal score or marginal vector field) are computationally
intractable. However, a fundamental principle from statistical
estimation theory provides a solid theoretical foundation for
constructing equivalent and tractable learning objectives.

#### The Principle of Optimal Estimation

Let $`X`$ and $`Y`$ be two random variables (or vectors). We wish to
find a function $`g(X)`$ that takes $`X`$ as input to serve as the best
possible estimate of $`Y`$. If we use the minimum Mean Squared Error
(MSE) as the criterion, then the optimal function $`g^*(X)`$ is the
conditional expectation of $`Y`$ given $`X`$:
``` math
g^*(X) = \underset{g}{\arg\min} \, \mathbb{E} \left[ \| Y - g(X) \|^2 \right] = \mathbb{E}[Y \mid X]
```

This theorem reveals a profound conclusion: for a loss function designed
to fit a random variable $`Y`$, its optimal solution is the conditional
expectation of $`Y`$.

#### Application to Denoising Score Matching

In diffusion models, our ultimate goal is to learn a network
$`s_\theta(x_t, t)`$ that approximates the intractable marginal score
function, $`\nabla_{x_t}\log p_t(x_t)`$.

According to
Theorem <a href="#thm:optimal_estimator" data-reference-type="ref"
data-reference="thm:optimal_estimator">7.0.0.1</a>, we can construct an
equivalent and tractable learning objective. We define:

- The observed variable $`X`$ is the noisy sample $`x_t`$.

- The target random variable $`Y`$ is the tractable conditional score
  function, $`\nabla_{x_t}\log p_t(x_t|x_0)`$. It is considered random
  because for a given $`x_t`$, the original sample $`x_0`$ from which it
  could have come is not unique.

In this case, the Denoising Score Matching loss function is
$`\mathcal{L}_{\text{DSM}}(\theta) = \mathbb{E}[\|s_\theta(x_t, t) - \nabla_{x_t}\log p_t(x_t|x_0)\|^2]`$.
According to the theorem, the optimal solution $`s_\theta^*`$ that
minimizes this loss is:
``` math
s_\theta^*(x_t, t) = \mathbb{E}[Y | X=x_t] = \mathbb{E}_{x_0 \sim p_t(x_0|x_t)} \left[ \nabla_{x_t} \log p_t(x_t|x_0) \right]
```

#### Application to Conditional Flow Matching

In Flow Matching, we encounter a completely analogous structure. Our
ultimate goal is to learn a network $`u_\theta(x_t, t)`$ to approximate
the intractable marginal vector field $`u_t(x_t)`$.

We can similarly apply
Theorem <a href="#thm:optimal_estimator" data-reference-type="ref"
data-reference="thm:optimal_estimator">7.0.0.1</a> to construct a
tractable learning objective. We define:

- The observed variable $`X`$ is a sample on the path, $`x_t`$.

- The target random variable $`Y`$ is the tractable conditional vector
  field, $`u_t(x_t|z)`$. It is considered random because for a given
  $`x_t`$, the possible path endpoints $`z=(x_0, x_1)`$ are not unique.

In this case, the Conditional Flow Matching loss function is
$`\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}[\|u_\theta(x_t, t) - u_t(x_t|z)\|^2]`$.
According to the theorem, the optimal solution $`u_\theta^*`$ that
minimizes this loss is:
``` math
u_\theta^*(x_t, t) = \mathbb{E}[Y | X=x_t] = \mathbb{E}_{z \sim p_t(z|x_t)} \left[ u_t(x_t|z) \right]
```

# Proof of $`\nabla_\theta \mathcal{L}_{\text{FM}}(\theta) = \nabla_\theta \mathcal{L}_{\text{CFM}}(\theta)`$

To prove that optimizing the computable conditional loss
$`\mathcal{L}_{\text{CFM}}(\theta)`$ is equivalent to optimizing the
theoretical marginal loss $`\mathcal{L}_{\text{FM}}(\theta)`$, we only
need to show that their gradients with respect to the model parameters
$`\theta`$ are identical, i.e.,
$`\nabla_\theta \mathcal{L}_{\text{FM}}(\theta) = \nabla_\theta \mathcal{L}_{\text{CFM}}(\theta)`$.

First, let’s write out the definitions of the two loss functions:
``` math
\begin{aligned}
\mathcal{L}_{\text{FM}}(\theta) &= \mathbb{E}_{t, x \sim p_t} \left[ \| u_\theta(x, t) - u_t(x) \|^2 \right] \\
\mathcal{L}_{\text{CFM}}(\theta) &= \mathbb{E}_{t, z \sim p_{\text{data}}, x \sim p_t(\cdot|z)} \left[ \| u_\theta(x, t) - u_t(x|z) \|^2 \right]
\end{aligned}
```
Taking the gradient of $`\mathcal{L}_{\text{CFM}}(\theta)`$ and
$`\mathcal{L}_{\text{FM}}(\theta)`$ with respect to $`\theta`$, we get:
``` math
\begin{aligned}
\nabla_\theta \mathcal{L}_{\text{CFM}}(\theta) &= \mathbb{E}_{t, z, x} \left[ \nabla_\theta \| u_\theta(x, t) - u_t(x|z) \|^2 \right] \nonumber \\
&= \mathbb{E}_{t, z, x} \left[ 2 \left( u_\theta(x, t) - u_t(x|z) \right) \cdot \nabla_\theta u_\theta(x, t) \right] \nonumber \\
&= \underbrace{2\mathbb{E}_{t, z, x} \left[ \langle u_\theta, \nabla_\theta u_\theta \rangle \right]}_{A} - \underbrace{2\mathbb{E}_{t, z, x} \left[ \langle u_t(x|z), \nabla_\theta u_\theta \rangle \right]}_{B}
\label{eq:grad_cfm}
\end{aligned}
```
``` math
\nabla_\theta \mathcal{L}_{\text{FM}}(\theta) = \underbrace{2\mathbb{E}_{t, x} \left[ \langle u_\theta, \nabla_\theta u_\theta \rangle \right]}_{C} - \underbrace{2\mathbb{E}_{t, x} \left[ \langle u_t(x), \nabla_\theta u_\theta \rangle \right]}_{D}
\label{eq:grad_fm}
```

To prove that the gradients are equal, we only need to show that in
Eq. <a href="#eq:grad_cfm" data-reference-type="eqref"
data-reference="eq:grad_cfm">[eq:grad_cfm]</a> and
<a href="#eq:grad_fm" data-reference-type="eqref"
data-reference="eq:grad_fm">[eq:grad_fm]</a>, the terms satisfy $`A=C`$
and $`B=D`$.

#### Proof that A=C

First, for any function $`H(x,t)`$ that depends only on $`(x,t)`$ (for
example, $`H = \langle u_\theta, \nabla_\theta u_\theta \rangle`$), its
expectation under the two different probability measures is equal:
``` math
\begin{aligned}
\mathbb{E}_{z \sim p_{\text{data}}, x \sim p_t(\cdot|z)} [H(x,t)] &= \iint H(x,t) \, p_t(x|z) p_{\text{data}}(z) \, \mathrm{d}z \mathrm{d}x \\
&= \int H(x,t) \left( \int p_t(x|z) p_{\text{data}}(z) \, \mathrm{d}z \right) \mathrm{d}x \\
&= \int H(x,t) \, p_t(x) \, \mathrm{d}x = \mathbb{E}_{x \sim p_t} [H(x,t)]
\end{aligned}
```

#### Proof that B=D

Using the identity $`u_t(x) = \mathbb{E}_{z \sim p_t(z|x)}[u_t(x|z)]`$：
``` math
\begin{aligned}
\mathbb{E}_{x \sim p_t} [\langle u_t(x), \nabla_\theta u_\theta \rangle] &= \int \langle u_t(x), \nabla_\theta u_\theta \rangle p_t(x) \, \mathrm{d}x \\
&= \int \left\langle \mathbb{E}_{z \sim p_t(z|x)}[u_t(x|z)], \nabla_\theta u_\theta \right\rangle p_t(x) \, \mathrm{d}x \\
&= \int \left\langle \int u_t(x|z) p_t(z|x) \, \mathrm{d}z, \nabla_\theta u_\theta \right\rangle p_t(x) \, \mathrm{d}x \\
&= \iint \langle u_t(x|z), \nabla_\theta u_\theta \rangle p_t(z|x) p_t(x) \, \mathrm{d}x \mathrm{d}z \\
&= \iint \langle u_t(x|z), \nabla_\theta u_\theta \rangle p_t(x|z) p_{\text{data}}(z) \, \mathrm{d}x \mathrm{d}z \quad (\text{因 } p_t(z|x)p_t(x) = p_t(x|z)p_{\text{data}}(z)) \\
&= \mathbb{E}_{z \sim p_{\text{data}}, x \sim p_t(\cdot|z)} [\langle u_t(x|z), \nabla_\theta u_\theta \rangle]
\end{aligned}
```

Since all corresponding terms in the gradient expressions for the two
loss functions are equal, we conclude that:
``` math
\nabla_\theta \mathcal{L}_{\text{FM}}(\theta) = \nabla_\theta \mathcal{L}_{\text{CFM}}(\theta)
```
Therefore, optimizing the computable $`\mathcal{L}_{\text{CFM}}`$ via
gradient descent is mathematically equivalent to optimizing the
theoretical $`\mathcal{L}_{\text{FM}}`$.

# The Conditional Vector Field

For the Gaussian probability path
$`p_t(x_t | x_1) = \mathcal{N}(x_t; \mu_t(x_1), \sigma_t(x_1)^2 \mathbf{I})`$,
the corresponding flow map $`\psi_t(x)`$ has a unique corresponding
vector field $`u_t(x|x_1)`$:
``` math
u_t(x|x_1) = \frac{\dot{\sigma}_t(x_1)}{\sigma_t(x_1)} \left(x - \mu_t(x_1)\right) + \dot{\mu}_t(x_1)
```

#### Proof

: Since $`\psi_t`$ is invertible, let $`x = \psi_t^{-1}(y)`$. Then we
can write:
``` math
\psi_t^{-1}(y) = \frac{y - \mu_t(x_1)}{\sigma_t(x_1)}
```

Simultaneously, differentiating $`\psi_t`$ yields:
``` math
\psi_t'(x) = \dot{\sigma}_t(x_1) x + \dot{\mu}_t(x_1)
```

According to the ODE, we derive:
``` math
\begin{aligned}
u_t(x|x_1) &= \dot{\psi}_t(x) = \dot{\psi}_t\left(\psi_t^{-1}(y)\right) = \psi_t'\left(\psi_t^{-1}(y)\right) \cdot \dot{y}(x_1) \\
&= \dot{\sigma}_t(x_1) \cdot \frac{y - \mu_t(x_1)}{\sigma_t(x_1)} + \dot{\mu}_t(x_1) \\
&= \frac{\dot{\sigma}_t(x_1)}{\sigma_t(x_1)} \left(x - \mu_t(x_1)\right) + \dot{\mu}_t(x_1)
\end{aligned}
```

</div>

<div id="refs" class="references csl-bib-body hanging-indent"
entry-spacing="0">

<div id="ref-anderson1982reverse" class="csl-entry">

Anderson, Brian DO. 1982. “Reverse-Time Diffusion Equation Models.”
*Stochastic Processes and Their Applications* 12 (3): 313–26.

</div>

<div id="ref-ho2020denoising" class="csl-entry">

Ho, Jonathan, Ajay Jain, and Pieter Abbeel. 2020. “Denoising Diffusion
Probabilistic Models.” *Advances in Neural Information Processing
Systems* 33: 6840–51.

</div>

<div id="ref-lipman2022flow" class="csl-entry">

Lipman, Yaron, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt
Le. 2022. “Flow Matching for Generative Modeling.” *arXiv Preprint
arXiv:2210.02747*.

</div>

<div id="ref-rezende2015variational" class="csl-entry">

Rezende, Danilo, and Shakir Mohamed. 2015. “Variational Inference with
Normalizing Flows.” In *International Conference on Machine Learning*,
1530–38. PMLR.

</div>

<div id="ref-song2020denoising" class="csl-entry">

Song, Jiaming, Chenlin Meng, and Stefano Ermon. 2020. “Denoising
Diffusion Implicit Models.” *arXiv Preprint arXiv:2010.02502*.

</div>

</div>
