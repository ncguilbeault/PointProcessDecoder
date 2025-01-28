using static TorchSharp.torch;

namespace PointProcessDecoder.Core;

/// <summary>
/// Represents the encoder of the model.
/// </summary>
public interface IEncoder : IModelComponent
{
    /// <summary>
    /// The encoder type of the model.
    /// </summary>
    public Encoder.EncoderType EncoderType { get; }

    /// <summary>
    /// The conditional intensities of the model.
    /// </summary>
    public Tensor[] ConditionalIntensities { get; }

    /// <summary>
    /// The estimations of the model.
    /// </summary>
    public IEstimation[] Estimations { get; }

    /// <summary>
    /// Evaluates the conditional intensities of the model given the inputs.
    /// </summary>
    /// <param name="inputs"></param>
    /// <returns></returns>
    public IEnumerable<Tensor> Evaluate(params Tensor[] inputs);

    /// <summary>
    /// Encodes the observations into a latent representation based on the joint distribution of the observations and the inputs.
    /// </summary>
    /// <param name="observations"></param>
    /// <param name="input"></param>
    public void Encode(Tensor observations, Tensor input);
}
