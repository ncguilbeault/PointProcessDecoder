namespace PointProcessDecoder.Core.Encoder;

/// <summary>
/// Represents the encoder type of the model.
/// </summary>
public enum EncoderType
{
    /// <summary>
    /// Represents an encoder for clusterless marks.
    /// </summary>
    ClusterlessMarks,

    /// <summary>
    /// Represents an encoder for sorted spikes.
    /// </summary>
    SortedSpikes
}
