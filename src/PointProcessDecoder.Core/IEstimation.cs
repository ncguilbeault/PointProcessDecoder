using static TorchSharp.torch;

namespace PointProcessDecoder.Core;

/// <summary>
/// Interface for density estimation.
/// </summary>
public interface IEstimation : IDisposable
{
    /// <summary>
    /// The device on which the density estimation is performed.
    /// </summary>
    public Device Device { get; }

    /// <summary>
    /// The scalar type of the density estimation.
    /// </summary>
    public ScalarType ScalarType { get; }

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
    /// Evaluate the density estimation at the given points.
    /// </summary>
    /// <param name="min"></param>
    /// <param name="max"></param>
    /// <param name="steps"></param>
    /// <returns></returns>
    public Tensor Evaluate(Tensor min, Tensor max, Tensor steps);

    public Tensor Estimate(Tensor points, int? dimensionStart = null, int? dimensionEnd = null);

    public Tensor Normalize(Tensor points);

    /// <summary>
    /// Fits new data points to the density estimation.
    /// </summary>
    /// <param name="data"></param>
    public void Fit(Tensor data);
}
