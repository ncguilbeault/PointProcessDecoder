using static TorchSharp.torch;

namespace PointProcessDecoder.Core;

public interface IDecoder
{
    public Device Device { get; }
    public ScalarType ScalarType { get; }
    public Tensor Decode(Tensor input, IEnumerable<Tensor> conditionalIntensities);
}
