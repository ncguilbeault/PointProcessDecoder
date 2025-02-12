using static TorchSharp.torch;

namespace PointProcessDecoder.Core;

/// <summary>
/// Represents a single component of the model.
/// </summary>
public interface IModelComponent
{
    /// <summary>
    /// The device on which the model component in located.
    /// </summary>
    public Device Device { get; }

    /// <summary>
    /// The scalar type of the model component.
    /// </summary>
    public ScalarType ScalarType { get; }

    /// <summary>
    /// Saves the state of the model component using the specified path.
    /// </summary>
    /// <param name="basePath"></param>
    public void Save(string basePath);

    /// <summary>
    /// Loads the state of the model component using the specified path.
    /// </summary>
    /// <param name="basePath"></param>
    /// <returns></returns>
    public IModelComponent Load(string basePath);
}
