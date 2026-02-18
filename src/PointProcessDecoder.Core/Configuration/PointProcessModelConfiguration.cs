using static TorchSharp.torch;

namespace PointProcessDecoder.Core.Configuration;

/// <summary>
/// Represents the initialization configuration of the point process model.
/// </summary>
public class PointProcessModelConfiguration
{
    /// <summary>
    /// The estimation method of the model.
    /// </summary>
    public Estimation.EstimationMethod EstimationMethod { get; set; }

    /// <summary>
    /// The transitions type of the model.
    /// </summary>
    public Transitions.TransitionsType TransitionsType { get; set; }

    /// <summary>
    /// The encoder type of the model.
    /// </summary>
    public Encoder.EncoderType EncoderType { get; set; }

    /// <summary>
    /// The decoder type of the model.
    /// </summary>
    public Decoder.DecoderType DecoderType { get; set; }

    /// <summary>
    /// The state space type of the model.
    /// </summary>
    public StateSpace.StateSpaceType StateSpaceType { get; set; }

    /// <summary>
    /// The likelihood type of the model.
    /// </summary>
    public Likelihood.LikelihoodType LikelihoodType { get; set; }

    /// <summary>
    /// The minimum state space of the model.
    /// </summary>
    public double[] MinStateSpace { get; set; } = [];

    /// <summary>
    /// The maximum state space of the model.
    /// </summary>
    public double[] MaxStateSpace { get; set; } = [];

    /// <summary>
    /// The steps state space of the model.
    /// </summary>
    public long[] StepsStateSpace { get; set; } = [];

    /// <summary>
    /// The bandwidth used for estimating the covariate distribution of the model.
    /// </summary>
    public double[] CovariateBandwidth { get; set; } = [];

    /// <summary>
    /// The state space dimensions of the model.
    /// </summary>
    public int StateSpaceDimensions { get; set; }

    /// <summary>
    /// The dimensionality of the marks.
    /// </summary>
    public int? MarkDimensions { get; set; }

    /// <summary>
    /// The number of different channels for recording marked events.
    /// </summary>
    public int? NumChannels { get; set; }

    /// <summary>
    /// The bandwidth used for estimating the mark distribution.
    /// </summary>
    public double[]? MarkBandwidth { get; set; }

    /// <summary>
    /// The number of sorted units.
    /// </summary>
    public int? NumUnits { get; set; }

    /// <summary>
    /// The distance threshold for determining whether kernel values are close enough to be merged into a single kernel.
    /// </summary>
    public double? DistanceThreshold { get; set; }

    /// <summary>
    /// Determines whether to ignore the contribution of no spikes in the likelihood computation.
    /// </summary>
    public bool IgnoreNoSpikes { get; set; }

    /// <summary>
    /// The sigma of the random walk used for estimating the transitions.
    /// </summary>
    public double? SigmaRandomWalk { get; set; }

    /// <summary>
    /// The kernel limit of the model.
    /// </summary>
    public int? KernelLimit { get; set; }

    /// <summary>
    /// The stay probability of the model.
    /// </summary>
    public double? StayProbability { get; set; }

    /// <summary>
    /// The scalar type of the model.
    /// </summary>
    public ScalarType? ScalarType { get; set; }
}
