using static TorchSharp.torch;

namespace PointProcessDecoder.Core;

public interface IPointProcessDecoder
{
    public void Encode(Tensor observations, Tensor inputs);
    public Tensor Decode(Tensor data);
}
