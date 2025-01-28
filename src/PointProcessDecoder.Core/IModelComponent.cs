using static TorchSharp.torch;

namespace PointProcessDecoder.Core;

/// <summary>
/// Represents a single component of the model.
/// </summary>
public interface IModelComponent : IDisposable
{
    /// <summary>
    /// The device on which the model component in located.
    /// </summary>
    public Device Device { get; }

    /// <summary>
    /// The scalar type of the model component.
    /// </summary>
    public ScalarType ScalarType { get; }
}
