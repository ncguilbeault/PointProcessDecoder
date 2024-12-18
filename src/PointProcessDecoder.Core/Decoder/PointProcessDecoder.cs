using static TorchSharp.torch;

namespace PointProcessDecoder.Core.Decoder;

public abstract class PointProcessDecoder : IDecoder
{
    public abstract Device Device { get; }
    public abstract ScalarType ScalarType { get; }
    public abstract Tensor Decode(Tensor inputs, IEnumerable<Tensor> conditionalIntensities);
}
