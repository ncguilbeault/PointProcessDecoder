using static TorchSharp.torch;

namespace PointProcessDecoder.Core;

/// <summary>
/// Base abstract class for models.
/// </summary>
public abstract class ModelBase: IModel
{
    /// <inheritdoc/>
    public abstract Device Device { get; }

    /// <inheritdoc/>
    public abstract ScalarType ScalarType { get; }

    /// <inheritdoc/>
    public abstract IEncoder Encoder { get; }

    /// <inheritdoc/>
    public abstract IDecoder Decoder { get; }

    /// <inheritdoc/>
    public abstract ILikelihood Likelihood { get; }

    /// <inheritdoc/>
    public abstract IStateSpace StateSpace { get; }

    /// <inheritdoc/>
    public abstract void Encode(Tensor observations, Tensor inputs);

    /// <inheritdoc/>
    public abstract Tensor Decode(Tensor inputs);

    /// <inheritdoc/>
    public virtual void Save(string basePath) { }

    /// <inheritdoc/>
    public virtual IModelComponent Load(string basePath) => throw new NotImplementedException();

    /// <summary>
    /// Loads the model using the specified base path and device.
    /// </summary>
    /// <param name="basePath"></param>
    /// <param name="device"></param>
    /// <returns></returns>
    public static IModelComponent Load(string basePath, Device? device) => throw new NotImplementedException();

    /// <inheritdoc/>
    public virtual void Dispose() { }
}
