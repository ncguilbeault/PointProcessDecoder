using static TorchSharp.torch;

namespace PointProcessDecoder.Core;

public interface IStateSpace
{
    public Device Device { get; }
    public ScalarType ScalarType { get; }
    public Tensor Points { get; }
    public long[] Shape { get; }
    public int Dimensions { get; }
}
