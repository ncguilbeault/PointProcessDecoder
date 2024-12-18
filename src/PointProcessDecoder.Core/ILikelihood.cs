using static TorchSharp.torch;

namespace PointProcessDecoder.Core;

public interface ILikelihood
{
    public Device Device { get; }
    public ScalarType ScalarType { get; }
    public Tensor LogLikelihood(Tensor inputs, Tensor conditionalIntensities);
}
