using static TorchSharp.torch;

namespace PointProcessDecoder.Core;

public interface IEncoder
{
    public Device Device { get; }
    public ScalarType ScalarType { get; }
    public IEnumerable<Tensor> Evaluate(params Tensor[] inputs);
    public void Encode(Tensor observations, Tensor input);
}
