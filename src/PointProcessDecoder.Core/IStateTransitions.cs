using static TorchSharp.torch;

namespace PointProcessDecoder.Core;

/// <summary>
/// Represents the state transitions of the model.
/// </summary>
public interface IStateTransitions : IModelComponent
{
    /// <summary>
    /// The transitions type of the model.
    /// </summary>
    public Transitions.TransitionsType TransitionsType { get; }

    /// <summary>
    /// The state transitions of the model.
    /// </summary>
    public Tensor Transitions { get; }
}
