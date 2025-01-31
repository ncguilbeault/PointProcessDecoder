using static TorchSharp.torch;

namespace PointProcessDecoder.Core.Configuration;

public class PointProcessModelConfiguration
{
    public Estimation.EstimationMethod EstimationMethod { get; set; }
    public Transitions.TransitionsType TransitionsType { get; set; }
    public Encoder.EncoderType EncoderType { get; set; }
    public Decoder.DecoderType DecoderType { get; set; }
    public StateSpace.StateSpaceType StateSpaceType { get; set; }
    public Likelihood.LikelihoodType LikelihoodType { get; set; }
    public double[] MinStateSpace { get; set; } = [];
    public double[] MaxStateSpace { get; set; } = [];
    public long[] StepsStateSpace { get; set; } = [];
    public double[] ObservationBandwidth { get; set; } = [];
    public int StateSpaceDimensions { get; set; }
    public int? MarkDimensions { get; set; }
    public int? MarkChannels { get; set; }
    public double[]? MarkBandwidth { get; set; }
    public int? NUnits { get; set; }
    public double? DistanceThreshold { get; set; }
    public bool IgnoreNoSpikes { get; set; }
    public double? SigmaRandomWalk { get; set; }
    public ScalarType? ScalarType { get; set; }
}
