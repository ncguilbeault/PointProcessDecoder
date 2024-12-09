using static TorchSharp.torch;

namespace PointProcessDecoder.Core;

/// <summary>
/// Abstract class for online density estimation.
/// </summary>
public abstract class OnlineDensityEstimation : IDensityEstimation
{
    /// <summary>
    /// The device on which the density estimation is performed.
    /// </summary>
    public abstract Device Device { get; }

    /// <summary>
    /// The kernel bandwidth used for the density estimation.
    /// </summary>
    public abstract Tensor KernelBandwidth { get; }

    /// <summary>
    /// Evaluate the density estimation at the given points.
    /// </summary>
    /// <param name="points"></param>
    /// <returns></returns>
    public abstract Tensor Evaluate(Tensor points);

    /// <summary>
    /// Evaluate the density estimation at the given points.
    /// </summary>
    /// <param name="min"></param>
    /// <param name="max"></param>
    /// <param name="steps"></param>
    /// <returns></returns>
    public abstract Tensor Evaluate(Tensor min, Tensor max, Tensor steps);

    /// <summary>
    /// Add a new data point to the density estimation.
    /// </summary>
    /// <param name="data"></param>
    public abstract void Add(Tensor data);

    /// <summary>
    /// Clear the density estimation kernels.
    /// </summary>
    public abstract void Clear();
}
