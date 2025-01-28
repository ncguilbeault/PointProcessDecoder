using static TorchSharp.torch;

namespace PointProcessDecoder.Core;

/// <summary>
/// Represents the likelihood of the model.
/// </summary>
public interface ILikelihood : IModelComponent
{
    /// <summary>
    /// The likelihood type of the model.
    /// </summary>
    public Likelihood.LikelihoodType LikelihoodType { get; }

    /// <summary>
    /// Measures the likelihood of the model given the inputs and the conditional intensities.
    /// </summary>
    /// <param name="inputs"></param>
    /// <param name="conditionalIntensities"></param>
    /// <returns></returns>
    public Tensor LogLikelihood(Tensor inputs, IEnumerable<Tensor> conditionalIntensities);
}
