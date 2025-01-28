using static TorchSharp.torch;

namespace PointProcessDecoder.Core;

/// <summary>
/// Represents the density estimation of the model.
/// </summary>
public interface IEstimation : IModelComponent
{
    /// <summary>
    /// The estimation method used for the density estimation.
    /// </summary>
    public Estimation.EstimationMethod EstimationMethod { get; }

    /// <summary>
    /// The kernel bandwidth used for the density estimation.
    /// </summary>
    public Tensor KernelBandwidth { get; }

    /// <summary>
    /// The kernels used for the density estimation.
    /// </summary>
    public Tensor Kernels { get; }

    /// <summary>
    /// Evaluates the density estimation at the given points.
    /// </summary>
    /// <param name="points"></param>
    /// <returns></returns>
    public Tensor Evaluate(Tensor points);

    /// <summary>
    /// Evaluates the density estimation at the given points.
    /// </summary>
    /// <param name="min"></param>
    /// <param name="max"></param>
    /// <param name="steps"></param>
    /// <returns></returns>
    public Tensor Evaluate(Tensor min, Tensor max, Tensor steps);

    /// <summary>
    /// Estimates the density at the given points.
    /// </summary>
    /// <param name="points"></param>
    /// <param name="dimensionStart"></param>
    /// <param name="dimensionEnd"></param>
    /// <returns></returns>
    public Tensor Estimate(Tensor points, int? dimensionStart = null, int? dimensionEnd = null);

    /// <summary>
    /// Normalizes the density estimation.
    /// </summary>
    /// <param name="points"></param>
    /// <returns></returns>
    public Tensor Normalize(Tensor points);

    /// <summary>
    /// Fits new data points to the density estimation.
    /// </summary>
    /// <param name="data"></param>
    public void Fit(Tensor data);
}
