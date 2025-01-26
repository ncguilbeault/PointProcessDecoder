using static TorchSharp.torch;

namespace PointProcessDecoder.Core;

public interface ILikelihood
{
    public Likelihood.LikelihoodType LikelihoodType { get; }
    public Tensor LogLikelihood(Tensor inputs, IEnumerable<Tensor> conditionalIntensities);
}
