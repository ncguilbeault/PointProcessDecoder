namespace PointProcessDecoder.Core.Encoder;

/// <summary>
/// Represents the encoder type of the model.
/// </summary>
public enum EncoderType
{
    /// <summary>
    /// Represents a clusterless mark encoder.
    /// </summary>
    ClusterlessMarkEncoder,

    /// <summary>
    /// Represents a sorted spike encoder.
    /// </summary>
    SortedSpikeEncoder
}
