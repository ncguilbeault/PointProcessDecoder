using static TorchSharp.torch;

namespace PointProcessDecoder.Core;

public interface IStateTransitions : IDisposable
{
    public Device Device { get; }
    public ScalarType ScalarType { get; }
    public Transitions.TransitionsType TransitionsType { get; }
    public Tensor Transitions { get; }
}
