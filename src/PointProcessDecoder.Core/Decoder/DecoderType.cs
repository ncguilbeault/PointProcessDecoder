namespace PointProcessDecoder.Core.Decoder;

/// <summary>
/// Represents the decoder type of the model.
/// </summary>
public enum DecoderType
{
    /// <summary>
    /// Represents a state space decoder.
    /// </summary>
    StateSpaceDecoder,

    /// <summary>
    /// Represents a hybrid state space classifier.
    /// </summary>
    HybridStateSpaceClassifier
}
