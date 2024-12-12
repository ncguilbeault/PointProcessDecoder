using static TorchSharp.torch;

namespace PointProcessDecoder.Core;

public interface IStateTransitions
{
    public Device Device { get; }
    public ScalarType ScalarType { get; }
    public Tensor Points { get; }
    public Tensor Transitions { get; }
}
