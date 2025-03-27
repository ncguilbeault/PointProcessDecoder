using static TorchSharp.torch;

namespace PointProcessDecoder.Core;

/// <summary>
/// Represents the decoder of the model.
/// </summary>
public interface IDecoder : IModelComponent
{
    /// <summary>
    /// The decoder type of the model.
    /// </summary>
    public Decoder.DecoderType DecoderType { get; }

    /// <summary>
    /// The initial state of the model.
    /// </summary>
    public Tensor InitialState { get; }

    /// <summary>
    /// The state transitions of the model.
    /// </summary>
    public Tensor[] Transitions { get; }

    /// <summary>
    /// Decodes the observations into the latent state based on the likelihood of the data.
    /// </summary>
    /// <param name="likelihood"></param>
    /// <returns></returns>
    public Tensor Decode(Tensor likelihood);
}
