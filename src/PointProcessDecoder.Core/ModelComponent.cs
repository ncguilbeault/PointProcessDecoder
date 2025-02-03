using static TorchSharp.torch;

namespace PointProcessDecoder.Core;

/// <summary>
/// Base abstract class for all model components.
/// </summary>
public abstract class ModelComponent : IModelComponent
{
    /// <summary>
    /// The device on which the model component is stored.
    /// </summary>
    public abstract Device Device { get; }

    /// <summary>
    /// The scalar type of the model component.
    /// </summary>
    public abstract ScalarType ScalarType { get; }

    /// <summary>
    /// Saves the state of the model component using the specified path.
    /// </summary>
    /// <param name="basePath"></param>
    public virtual void Save(string basePath) { }

    /// <summary>
    /// Loads the state of the model component using the specified path.
    /// </summary>
    /// <param name="basePath"></param>
    /// <returns></returns>
    public virtual IModelComponent Load(string basePath) { return this; }

    /// <summary>
    /// Disposes of the model component.
    /// </summary>
    public virtual void Dispose() { }
}
