# PointProcessDecoder

This repo contains a C# implementation of the Bayesian state space point process neural decoder. The code uses the [TorchSharp library](https://github.com/dotnet/TorchSharp), a pytorch implementation in .NET/C#, and is inspired by the [replay_trajectory_classification repository](https://github.com/Eden-Kramer-Lab/replay_trajectory_classification) from the Eden-Kramer Lab. It provides a flexible framework for performing neural decoding of observations from spike-train or clusterless mark data based on point processes and Bayesian state space models.

## Overview

The goal of this software is to perform neural decoding. Bayesian state-space models, in particular, provide a framework to model latent states based on neural activity while point processes capture the probabilistic relationship between neural activity and latent observations.

## Description

There are 3 main components of the model: encoder, likelihood, and decoder.

### Decoder

Currently, the software supports the `StateSpaceDecoder`, which implements a Bayesian state space decoding model. The Bayesian state space decoder postulates that, given the initial conditions $p(x_0)$, the posterior probability $p(x_t | O_{1:t})$ at each time step $t$ can be calculated iteratively using:

$$
p(x_t | O_{1:t}) \propto \int p(O_t | x_t) p(x_t | x_{t-1}) p(x_{t-1} | O_{1:t-1}) dx_{t-1}
$$

We need to specify the initial conditions, $p(x_0)$, the transitions or state space dynamics, $p(x_t | x_{t-1})$, and the likelihood function, $p(O_t | x_t)$.

For the initial conditions, $p(x_0)$, the package currently supports defining a `DiscreteUniformStateSpace`, in which all possible states are equally likely to occur. In this configuration, the state space is bounded by minimum and maximum values for each dimension of our data and gets partitioned into discrete bins based on the number of steps along each dimension. Users must supply values for the `minStateSpace`, `maxStateSpace`, and `stepsStateSpace`, which can be uniquely specified for each dimension. The length of each array corresponds to the number of state space dimension, and all of them must match the user specified `stateSpaceDimensions` parameter.

For the transitions, $p(x_t | x_{t-1})$, we can specify how the latent variable evolves over time. The software currently supports `UniformTransitions`, where the latent variable has equal probability of transitioning to any other point in the state space, or `RandomWalkTransitions`, where the transitions are constrained by a multivariate normal distribution such that adjacent positions in the state space are more likely to occur. The variance of `RandomWalkTransitions` can be specified with the `sigmaRandomWalk` parameter that determines the variance of the movement in all dimensions.

### Likelihood

For the likelihood measure, $p(O_t | x_t)$, the method selected will depend on the type of encoder. There are currently 2 types of likelihoods: `PoissonLikelihood` and `ClusterlessLikelihood`. 

#### Poisson Likelihood

The `PoissonLikelihood` is used in conjunction with the `SortedSpikeEncoder` and performs the following calculation:

$$
p(O_t | x_t) = p( \Delta N ^{1:U} _{t} | x_t) \propto \prod ^U _{i=1} [ \lambda _i (t | x_t) \Delta _t] ^{\Delta N ^i _{t}} exp[ - \lambda _i (t | x_t) \Delta _t]
$$

Here, $N^i_{t}$ represents whether unit $i$ has produced a spike/event at time $t$ within the $\Delta_t$ time window. For purposes of this software, $\Delta_t = 1$. The conditional intensity, $\lambda_i(t | x_t)$, represents the instantaneous rate of events of unit $i$ given the observations of the latent variable $x$ at time $t$.

#### Clusterless Likelihood

The `ClusterlessLikelihood` method is used in conjunction with the clusterless mark encoder and performs the following computation:

$$
p(O_t | x_t) = p( \Delta N ^{1:C} _t, \vec{m} ^c _{t,j}) \propto \prod ^C _{c=1} \prod ^{\Delta N ^C _t} _{j=1} [ \lambda _c (t, \vec{m} ^c _{t,j} | x_t) \Delta _t ] exp [ - \Lambda _c (t | x_t) \Delta _t]
$$

Here, $N_{t}^{1:C}$ represents whether a mark was detected on channel $c$, and $\vec{m}^c_{t,j}$ represents the marks $\vec{m}$ detected on channel $c$ at time $t$ with marks detected through times $j$. The `ClusterlessLikelihood` is comprised of two seperate conditional intensity functions. The conditional intensity $\lambda_c(t, \vec{m}^c_{t,j} | x_t)$ represents the firing rate of unique sets of marks $\vec{m}$ on channel $c$, whereas $\Lambda_c(t | x_t)$ represents the rate at which all events occur on channel $c$.

### Encoder

The encoder is used to calculate the conditional intensity functions, the rate of events occurring with respect to the latent variable. There are two types of encoders currently supported: `SortedSpikeEncoder` and `ClusterlessMarkEncoder`

#### Sorted Spike Encoder

In the case of the `SortedSpikeEncoder`, the conditional intensity function for each sorted unit takes the form:

$$
\lambda_i(t | x_t) = \mu_i \frac{p_i(x_t)}{\pi(x)}
$$

Where $\mu$ is the mean firing rate, $p(x_t)$ is the distribution of latent observations only when spikes are observed for unit $i$, and $\pi(x)$ is the full distribution of the latent observation. When using the `SortedSpikeEncoder`, the user must specify the `nUnits` parameter to allocates the appropriate number of unit estimators at runtime. 

#### Clusterless Mark Encoder

For the `ClusterlessMarkEncoder`, we use a marked point process procedure where each spike/event has an associated feature vector or set of marks. In general, marks can be anything associated with a spike event (i.e. spike width, maximum amplitude, etc). The mark conditional intensity function is:

$$
\lambda_c(t, \vec{m}^c_{t,j} | x_t) = \mu_c \frac{p_c(x_t,\vec{m}^c_{t,j})}{\pi(x)}
$$

Where $p_c(x_t,\vec{m}^c_{t,j})$ is the joint probability distribution of the latent state observations $x_t$ observed for unique sets of marks $\vec{m}$ on recording channel $c$. 

Next, we define the channel conditional intensity function as:

$$
\Lambda_c(t | x_t) = \mu_c \frac{p_c(x_t)}{\pi(x)}
$$

Where $p_c(x_t)$ represents the marginal distribution over the latent state across all events observed on recording channel $c$. When using the `ClusterlessMarkEncoder`, users must specify the `markDimensions` and `markChannels` parameters which define the number of mark features associated with each spike event and the number of recording channels, respectively.

### Density Estimation

We approximate the probability distributions $p_c(x_t,\vec{m}^c_{t,j})$, $p_c(x_t)$, $p_i(x_t)$, and $\pi(x)$, using methods for kernel density estimation. The package provides 2 methods for estimation: `KernelDensity` and `KernelCompression`. 

#### Kernel Density

The `KernelDensity` estimation method can be formalized as follows:

$$
p(x) = \frac{1}{N} \sum ^{N} _{i=1} \frac{1}{ \sqrt{ (2 \pi) ^d \prod ^d _{j=1} h _j }} exp \left( -\frac{1}{2} \left( \frac{X _i - x}{h _d} \right) ^T \left( \frac{X _i - x}{h _d} \right) \right)
$$

The probability distribution $p(x)$ can be calculated by taking a given set of points $X_i = {X_1, \dots, X_N}$ with dimensionality $D$ and evaluating them over a normal distribution associated with the datapoints $x$ observed during the encoding procedure. The kernel bandwidth parameter, $h$, describes the variance of the gaussian at dimension $d$. The `KernelDensity` method is more accurate compared to the `KernelCompression` method, but requires more memory and computation time as the number of observations increases.

#### Kernel Compression

At the cost of a small amount of accuracy, the `KernelCompression` estimation method is faster than the `KernelDensity` method with greater observations and requires less memory. It works by computing a gaussian mixture model to represent $p(x)$ with fewer kernels. Thus, the distribution, $p(x)$, takes the following form:

$$
p(x) = \sum^C_{i=1}w_i\phi_i(x)
$$

Where each kernel component $i$ contributes a probability density, $\phi$, with some weight $w$. Again, the density, $\phi$, is taken as a gaussian kernel of the same form above.

#### Kernel Merging Procedure

The `KernelCompression` algorithm uses a kernel merging procedure to determine whether the observed data point should lead to the creation of a new kernel component or whether the data point should be used to update the paremeters of the closest existing kernel. First, the mahalanobis distance is calculated between existing kernels and the new data point. The distance is evaluated against the user specified `distanceThreshold` parameter, such that if the distance to the closest kernel is greater than the `distanceThreshold`, then the data point is used to create a new kernel. If the distance is less than this, the closest kernel is updated using a moment matching method. The new weight of the component is updated as follows:

$$
w = w_1 + w_2
$$

And the new $\mu$ of the component becomes:

$$
\mu = w ^{-1} (w _1 \mu _1 + w _2 \mu _2)
$$

Since only the diagonal of the covariance matrix is used in the kernel density estimate, we only update the diagonal elements of the matrix using:

$$
h = w ^{-1} \sum ^2 _{i=1} (h _i + \mu ^2 _i) - \mu ^2
$$

#### Bandwidth Selection

Users of the package must specify the bandwidth parameters used for density estimation. For the distribution, $\pi(x)$, users set the `observationBandwidth` parameter, where a unique bandwidth can be set for each dimension. Again, the length of this array must be equal to the number of `stateSpaceDimensions` defined above. For both the `SortedSpikeEncoder` and `ClusterlessMarkEncoder`, the `observationBandwidth` parameter is used to compute the distributions $p_i(x_t)$ and $p_c(x_t)$, respectively. The `ClusterlessMarkEncoder` also takes the `markBandwidth` parameter which is used for calculating the distribution, $p_c(x_t,\vec{m}^c_{t,j})$. A unique bandwidth can be specified for each mark feature so long as the length of the array is equal to the `markDimensions` parameter.

## Steps to Build

1. Install the .NET SDK:
Download [the .NET SDK](https://dotnet.microsoft.com/download) if you haven't already.

2. Clone the repository:

```bash
git clone https://github.com/ncguilbeault/PointProcessDecoder.cs
cd PointProcessDecoder
```

3. Restore dependencies:

```bash
dotnet restore
```

4. Build the solution:

```bash
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
