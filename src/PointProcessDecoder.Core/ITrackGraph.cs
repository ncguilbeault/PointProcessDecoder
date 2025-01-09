using static TorchSharp.torch;

namespace PointProcessDecoder.Core;

public interface ITrackGraph
{
    public Tensor Nodes { get; }
    public Tensor Edges { get; }
    public Tensor EdgeSpacing { get; }
}
