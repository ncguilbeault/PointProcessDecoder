using static TorchSharp.torch;

namespace PointProcessDecoder.Core;

/// <summary>
/// Represents the state space of the model.
/// </summary>
public interface IStateSpace : IModelComponent
{
    /// <summary>
    /// The state space type of the model.
    /// </summary>
    public StateSpace.StateSpaceType StateSpaceType { get; }

    /// <summary>
    /// The points of the state space.
    /// </summary>
    public Tensor Points { get; }

    /// <summary>
    /// The axes points of the state space.
    /// </summary>
    public Tensor AxesPoints { get; }

    /// <summary>
    /// The shape of the state space.
    /// </summary>
    public long[] Shape { get; }

    /// <summary>
    /// The dimensions of the state space.
    /// </summary>
    public int Dimensions { get; }
}
