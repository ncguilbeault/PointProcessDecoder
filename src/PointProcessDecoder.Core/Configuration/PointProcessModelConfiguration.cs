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
    /// The observation bandwidth of the model.
    /// </summary>
    public double[] ObservationBandwidth { get; set; } = [];

    /// <summary>
    /// The state space dimensions of the model.
    /// </summary>
    public int StateSpaceDimensions { get; set; }

    /// <summary>
    /// The mark dimensions of the model.
    /// </summary>
    public int? MarkDimensions { get; set; }

    /// <summary>
    /// The mark channels of the model.
    /// </summary>
    public int? MarkChannels { get; set; }

    /// <summary>
    /// The mark bandwidth of the model.
    /// </summary>
    public double[]? MarkBandwidth { get; set; }

    /// <summary>
    /// The number of units of the model.
    /// </summary>
    public int? NUnits { get; set; }

    /// <summary>
    /// The distance threshold of the model.
    /// </summary>
    public double? DistanceThreshold { get; set; }

    /// <summary>
    /// The ignore no spikes flag of the model.
    /// </summary>
    public bool IgnoreNoSpikes { get; set; }

    /// <summary>
    /// The sigma random walk of the model.
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
