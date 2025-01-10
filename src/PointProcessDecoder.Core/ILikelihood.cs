using static TorchSharp.torch;

namespace PointProcessDecoder.Core;

public interface ILikelihood
{
    public Tensor LogLikelihood(Tensor inputs, IEnumerable<Tensor> conditionalIntensities);
}
