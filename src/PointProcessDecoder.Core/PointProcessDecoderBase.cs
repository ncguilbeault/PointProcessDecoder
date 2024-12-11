using static TorchSharp.torch;

using PointProcessDecoder.Core.Estimation;
using PointProcessDecoder.Core.Transitions;
using PointProcessDecoder.Core.Detection;

namespace PointProcessDecoder.Core;

public abstract class PointProcessDecoderBase : IPointProcessDecoder
{
    /// <summary>
    /// The number of dimensions of the latent space.
    /// </summary>
    public abstract int LatentDimensions { get; }

    /// <summary>
    /// The device on which the model is running.
    /// </summary>
    public abstract Device Device { get; }

    /// <summary>
    /// The posterior distribution of the model.
    /// </summary>
    public abstract Tensor Posterior { get; }

    /// <summary>
    /// The transitions of the model.
    /// </summary>
    public abstract StateTransitions Transitions { get; }

    /// <summary>
    /// Encodes the observations into a latent representation based on the joint distribution of the observations and the data.
    /// </summary>
    /// <param name="observations"></param>
    /// <param name="data"></param>
    public abstract void Encode(Tensor observations, Tensor inputs);

    /// <summary>
    /// Decodes the latent representation from the data.
    /// </summary>
    /// <param name="data"></param>
    /// <returns></returns>
    public abstract Tensor Decode(Tensor data);
}