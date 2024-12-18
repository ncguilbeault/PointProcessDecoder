using static TorchSharp.torch;

namespace PointProcessDecoder.Core;

public interface IEncoder
{
    public Device Device { get; }
    public ScalarType ScalarType { get; }
    public IEnumerable<Tensor>? ConditionalIntensities { get; }
    public IEnumerable<Tensor> Evaluate();
    public void Encode(Tensor observations, Tensor input);
}
