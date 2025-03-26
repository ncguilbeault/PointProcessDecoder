using static TorchSharp.torch;

namespace PointProcessDecoder.Core;

/// <summary>
/// Represents the decoder of the model.
/// </summary>
public interface IClassifier : IModelComponent
{
    /// <summary>
    /// The decoder type of the model.
    /// </summary>
    public Classifier.ClassifierType ClassifierType { get; }

    /// <summary>
    /// Decodes the observations into the latent state based on the likelihood of the data.
    /// </summary>
    /// <param name="likelihood"></param>
    /// <returns></returns>
    public Tensor Decode(Tensor likelihood);
}
