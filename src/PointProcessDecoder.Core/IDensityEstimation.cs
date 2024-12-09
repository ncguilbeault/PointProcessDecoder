using static TorchSharp.torch;

namespace PointProcessDecoder.Core;

/// <summary>
/// Interface for density estimation.
/// </summary>
public interface IDensityEstimation
{
    /// <summary>
    /// The device on which the density estimation is performed.
    /// </summary>
    public Device Device { get; }

    /// <summary>
    /// The kernel bandwidth used for the density estimation.
    /// </summary>
    public Tensor KernelBandwidth { get; }

    /// <summary>
    /// Evaluate the density estimation at the given points.
    /// </summary>
    /// <param name="points"></param>
    /// <returns></returns>
    public Tensor Evaluate(Tensor points);

    /// <summary>
    /// Evaluate the density estimation over a range of points.
    /// </summary>
    /// <param name="min"></param>
    /// <param name="max"></param>
    /// <param name="steps"></param>
    /// <returns></returns>
    public Tensor Evaluate(Tensor min, Tensor max, Tensor steps);
}
