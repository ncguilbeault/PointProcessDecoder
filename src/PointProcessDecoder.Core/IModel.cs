using static TorchSharp.torch;

namespace PointProcessDecoder.Core;

/// <summary>
/// Represents the type of point process decoder model.
/// </summary>
public interface IModel : IModelComponent
{    
    /// <summary>
    /// The encoder of the model.
    /// </summary>
    public IEncoder Encoder { get; }

    /// <summary>
    /// The decoder of the model.
    /// </summary>
    public IDecoder Decoder { get; }

    /// <summary>
    /// The likelihood of the model.
    /// </summary>
    public ILikelihood Likelihood { get; }

    /// <summary>
    /// The state space of the model.
    /// </summary>
    public IStateSpace StateSpace { get; }

    /// <summary>
    /// Encodes the observations into a latent representation based on the joint distribution of the observations and the inputs.
    /// </summary>
    /// <param name="observations"></param>
    /// <param name="data"></param>
    public void Encode(Tensor observations, Tensor inputs);

    /// <summary>
    /// Decodes the the latent representation from the inputs.
    /// </summary>
    /// <param name="inputs"></param>
    /// <returns></returns>
    public Tensor Decode(Tensor inputs);
}
