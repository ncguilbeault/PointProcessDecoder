using static TorchSharp.torch;

namespace PointProcessDecoder.Core;

public interface IPointProcessModel
{
    public Tensor Encode(Tensor observations, Tensor data);
    public Tensor Decode(Tensor data);
    public Tensor Likelihood(Tensor data);
}
