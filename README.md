# PointProcessDecoder

This repo contains a C# implementation of the Bayesian state space point process neural decoder. The code uses the [TorchSharp library](https://github.com/dotnet/TorchSharp), a pytorch implementation in .NET/C#, and is inspired by the [replay_trajectory_classification repository](https://github.com/Eden-Kramer-Lab/replay_trajectory_classification) from the Eden-Kramer Lab. It provides a flexible framework for performing neural decoding of observations from spike-train or clusterless mark data based on point processes and Bayesian state space models.

## Overview

The goal of this software is to perform neural decoding. Bayesian state-space models, in particular, provide a framework to model the transitions between states based on neural activity and point processes capture the probabilistic relationship between neural activity and observations.

## Theory

The core of the model can be broken down into 3 parts: encoding, measurement of the likelihood, and decoding. The model encodes neural activity and state observations using a point process framework, and then decodes the state from neural activity alone using a bayesian state space approach. Currently, the software supports `StateSpaceDecoder`, which implements a Bayesian state space decoding model. The Bayesian state space decoder postulates that, given the initial conditions $p(x_0)$, the posterior probability $p(x_t | O_{1:t})$ at each time step $t$ can be calculated iteratively using:

$$
p(x_t | O_{1:t}) \propto \int p(O_t | x_t) p(x_t | x_{t-1}) p(x_{t-1} | O_{1:t-1}) dx_{t-1}
$$

We need to specify the initial conditions, $p(x_0)$, the transitions or state space dynamics, $p(x_t | x_{t-1})$, and the likelihood function, $p(O_t | x_t)$.

For the initial conditions, $p(x_0)$, we can specify prior knowledge on the distribution of our data. The software currently supports defining a `DiscreteUniformStateSpace`, in which all possible states are equally likely to occur. In this configuration, the state space is bounded by minimum and maximum values for each dimension of our data and gets partitioned into discrete bins based on the number of steps along each dimension.

For the transitions, $p(x_t | x_{t-1})$, we can specify how the latent variable evolves over time. The software currently supports `UniformTransitions`, such that the latent variable has equal probability of transitioning to any other point in the state space, or `RandomWalkTransitions`, such that dynamics are constrained by a multivariate normal distribution with a parameter `sigma` that determines the variance of the movement in all dimensions.

For the likelihood measure, $p(O_t | x_t)$, the method selected will depend on the data being encoded as a point process. There are currently 2 types of likelihoods: `PoissonLikelihood` and `ClusterlessLikelihood`. The `PoissonLikelihood` is used in conjunction with the sorted spike encoder and performs the following calculation:

$$
p(O_t | x_t) = p( \Delta N ^{1:U} _{t} | x_t) \propto \prod ^U _{i=1} [ \lambda _i (t | x_t) \Delta _t] ^{\Delta N ^i _{t}} exp[ - \lambda _i (t | x_t) \Delta _t]
$$

Here, $N^i_{t}$ represents whether unit $i$ has produced a spike/event at time $t$, $\Delta_t$ is the time difference, and $\lambda_i(t | x_t)$ is the conditional intensity, or instantaneous rate of events of unit $i$ given the observations of the state $x$ at time $t$. The conditional intensity of unit $i$ can be estimated using kernel density approaches described below.

The `ClusterlessLikelihood` method is used in conjunction with the clusterless mark encoder and performs the following computation:

$$
p(O_t | x_t) = p( \Delta N ^{1:C} _t, \vec{m} ^c _{t,j}) \propto \prod ^C _{c=1} \prod ^{\Delta N ^C _t} _{j=1} [ \lambda _c (t, \vec{m} ^c _{t,j} | x_t) \Delta _t ] ^{N ^c _{t}} exp [ - \Lambda _c (t | x_t) \Delta _t]
$$

Here, $N_{t}^{1:C}$ represents whether a mark was detected on channel $c$, and $\vec{m}^c_{t,j}$ represents the marks $\vec{m}$ detected on channel $c$ at time $t$ with marks detected through times $j$. Again, $\Delta_t$ is the time difference. The clusterless likelihood is comprised of two seperate conditional intensity functions. The conditional intensity $\lambda_c(t, \vec{m}^c_{t,j} | x_t)$ represents the firing rate of unique sets of marks $\vec{m}$ on channel $c$, whereas $\Lambda_c(t | x_t)$ represents the rate at which all events occur on channel $c$.

During the encoding process, activity of sorted units sorted or clusterless marks gives rise to unique conditional intensity functions. The conditional intensity function describes the rate of events occurring with respect to the observation of the state. In the case of the `SortedSpikesEncoder`, the conditional intensity function for each sorted unit takes the form:

$$
\lambda_i(t_k | x_k) = \mu_i \frac{p_i(x_k)}{\pi(x)}
$$

Where $\mu_i$ is the mean firing rate of unit $i$, $p_i(x_k)$ is the distribution of state observations only when spikes are observed for unit $i$, and $\pi(x)$ is the full distribution of state observations. 

For the `ClusterlessMarksEncoder`, each spike/event has an associated feature vector, called marks. In general, marks can be anything associated with an event (i.e. spike width, maximum amplitude, etc). However, since a single recording channel will detect events from multiple sources, the goal is to have marks that are unique to the underlying units, with the assumption being that each underlying unit provides unique information about your state. Thus, the conditional intensity function describing the instaneous event rate of marks associated with the state observations is:

$$
\lambda_c(t, \vec{m}^c_{t,j} | x_t) = \mu_c \frac{p_c(x_t,\vec{m}^c_{t,j})}{\pi(x)}
$$

Where $p_c(x_t,\vec{m}^c_{t,j})$ is the joint probability distribution of the state observations $x_t$ observed for unique sets of observed marks $\vec{m}$ on recording channel $c$. The conditional intensity function describing the event rate of a particular recording channel is:

$$
\Lambda_c(t | x_t) = \mu_c \frac{p_c(x_t)}{\pi(x)}
$$

Where $p_c(x_t)$ is just the distribution of state observations when events occur on recording channel $c$.

Since we do not have access to the true probability distributions $p_c(x_t,\vec{m}^c_{t,j})$, $p_c(x_t)$, $p_i(x_k)$ or $\pi(x)$, we can approximate them using methods for kernel density estimation. The software provides 2 methods for kernel density estimation: `KernelDensity` and `KernelCompression`. The `KernelDensity` estimation method can be formalized as follows:

$$
p(x) = \frac{1}{N} \sum ^{N} _{i=1} \frac{1}{ \prod ^d _{j=1} h_j \sqrt{(2 \pi ) ^d}} exp \left( -\frac{1}{2} \left( \frac{X_i - x}{h_d} \right) ^T \left( \frac{X_i - x}{h_d} \right) \right)
$$

The probability distribution $p(x)$ can be calculated by taking a given set of points $X_i = {X_1, \dots, X_N}$ with dimensionality $D$ and evaluating them over all gaussian kernels associated with the data $x$ saved during the encoding procedure. In this method, every point, $x$, observed during encoding is used to compute a gaussian kernel along the dimension $d$. This method of density estimation relies heavily on the kernel bandwidth parameter $h$, which describes the variance of the gaussian at dimension $d$. While this method works well for estimating probability distributions, the inference speed is quite slow and scales linearly with the number of observations encoded. At the cost of a small amount of accuracy, the `KernelCompression` estimation method is much faster and uses a gaussian mixture model to represent $p(x)$, which takes the following form:

$$
p(x) = \sum^C_{i=1}w_i\phi_i(x)
$$

In this case, significantly fewer gaussian components $C$ are needed to achieve similar accuracy. Each component $i$ is assigned a unique weight $w_i$ and set of parameters $\phi_i$. The parameters define a gaussian kernel of the form:

$$
\phi_i(x) = \frac{1}{\sqrt{(2\pi)^d\prod^d_{j=1}h^2_{i,j}}}exp\left(-\frac{1}{2}\sum^d_{j=1}\frac{(x_j-\mu_{i,j})^2}{h^2_{i,j}}\right)
$$

Where $\mu_{i,j}$ represents the mean of kernel $i$ along dimension $j$, and since we only use the diagonal of the covariance matrix, $\Sigma$ for kernel $i$, the term $h^2$ defines the variance of our kernel $i$ at dimension $j$. To determine whether new data points should lead to the creation of a new kernel or whether an existing kernel should be updated, the mahalanobis distance is calculated between existing kernels and the data point. The distance is then compared against the `DistanceThreshold` parameter, with further values leading to new kernels and nearer values leading to kernels being updated. While some information is lost when using kernel compression, there is a dramatic increase in performance, particularly when the number of observations is very large. 

## Steps to Build

1. Install .NET 8:
Download [the .NET SDK](https://dotnet.microsoft.com/download) if you haven't already.

2. Clone the repository:

```cmd
git clone https://github.com/ncguilbeault/PointProcessDecoder.cs
cd PointProcessDecoder
```

3. Restore dependencies:

```cmd
dotnet restore
```

4. Build the solution:

```
dotnet build
```

## Quickstart

Here is a minimal example of how to use the decoder in a console app:

```csharp
using PointProcessDecoder.Core;
using PointProcessDecoder.Plot;
using PointProcessDecoder.Simulation;

namespace DecoderDemo
{
    class Program
    {
        static void Main(string[] args)
        {
            // 1. Load data.
            // Example: Generate simulated data
            (position, spikeCounts) = Simulation.Utilities.InitializeSimulation1D(
                numNeurons: 40,
                placeFieldRadius: 0.8,
                firingThreshold: 0.2
            );

            // 2. Create the model and select parameters.
            var model = new PointProcessModel(
                estimationMethod: Core.Estimation.EstimationMethod.KernelDensity,
                transitionsType: Core.Transitions.TransitionsType.Uniform,
                encoderType: Core.Encoder.EncoderType.SortedSpikeEncoder,
                decoderType: Core.Decoder.DecoderType.StateSpaceDecoder,
                stateSpaceType: Core.StateSpace.StateSpaceType.DiscreteUniformStateSpace,
                likelihoodType: Core.Likelihood.LikelihoodType.Poisson,
                minStateSpace: [0],
                maxStateSpace: [120],
                stepsStateSpace: [50],
                observationBandwidth: [5],
                stateSpaceDimensions: 1,
                nUnits: 40
            );

            // 4. Encode neural data and observations
            model.Encode(spikeCounts, position);

            // 5. Predict or decode observations from spikes
            var prediction = model.Decode(spikeCounts);

            // 6. Display results
            Heatmap plotPrediction = new(
                xMin: 0,
                xMax: steps * cycles,
                yMin: 0,
                yMax: 120,
                title: "Prediction"
            );

            plotPrediction.Show<float>(
                prediction
            );

            plotPrediction.Save(png: true);
        }
    }
}
```

## References

This work is based on several previously published works. If you use this software, consider citing the following:

1. Denovellis, E. L., Gillespie, A. K., Coulter, M. E., Sosa, M., Chung, J. E., Eden, U. T., & Frank, L. M. (2021). Hippocampal replay of experience at real-world speeds. Elife, 10, e64505.

2. Sodkomkham, D., Ciliberti, D., Wilson, M. A., Fukui, K. I., Moriyama, K., Numao, M., & Kloosterman, F. (2016). Kernel density compression for real-time Bayesian encoding/decoding of unsorted hippocampal spikes. Knowledge-Based Systems, 94, 1-12.


**Contributions and feedback are welcome!**
