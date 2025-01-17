using static TorchSharp.torch;

namespace PointProcessDecoder.Core;

public interface IModel : IDisposable
{    
    /// <summary>
    /// The device on which the model is running.
    /// </summary>
    public Device Device { get; }

    /// <summary>
    /// The scalar type of the model.
    /// </summary>
    public ScalarType ScalarType { get; }

    /// <summary>
    /// The encoder of the model.
    /// </summary>
    public IEncoder Encoder { get; }

    /// <summary>
    /// The decoder of the model.
    /// </summary>
    public IDecoder Decoder { get; }

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
