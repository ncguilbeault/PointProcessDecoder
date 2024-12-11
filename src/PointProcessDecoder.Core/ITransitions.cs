using static TorchSharp.torch;

namespace PointProcessDecoder.Core;

public interface IStateTransitions
{
    public Tensor Points { get; }
}
