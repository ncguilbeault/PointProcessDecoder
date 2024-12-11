using System;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;

namespace PointProcessDecoder.Core.Estimation;

/// <summary>
/// Kernel density estimation with gaussian kernel compression.
/// </summary>
public class KernelCompression : DensityEstimation
{
    private readonly Device _device;
    /// <inheritdoc/>
    public override Device Device => _device;

    private List<WeightedGaussian> _kernels = new();
    /// <summary>
    /// The weighted gaussian components.
    /// </summary>
    public List<WeightedGaussian> Kernels => _kernels;

    private readonly Tensor _kernelBandwidth;
    /// <inheritdoc/>
    public override Tensor KernelBandwidth => _kernelBandwidth;

    private readonly double _distanceThreshold;
    private readonly Tensor _weight = ones(1);


    private readonly int _dimensions;
    /// <summary>
    /// The number of dimensions of the observations.
    /// </summary>
    public int Dimensions => _dimensions;

    /// <summary>
    /// Initializes a new instance of the <see cref="KernelCompression"/> class.
    /// </summary>
    /// <param name="bandwidth"></param>
    /// <param name="dimensions"></param>
    /// <param name="distanceThreshold"></param>
    /// <param name="device"></param>
    public KernelCompression(double? bandwidth = null, int? dimensions = null, double? distanceThreshold = null, Device? device = null)
    {
        _device = device ?? CPU;
        _distanceThreshold = distanceThreshold ?? double.NegativeInfinity;
        _dimensions = dimensions ?? 1;
        _kernelBandwidth = tensor(bandwidth ?? 1.0).repeat(_dimensions).to(_device);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="KernelCompression"/> class.
    /// </summary>
    /// <param name="bandwidth"></param>
    /// <param name="dimensions"></param>
    /// <param name="distanceThreshold"></param>
    /// <param name="device"></param>
    /// <exception cref="ArgumentException"></exception>
    public KernelCompression(double[] bandwidth, int dimensions, double? distanceThreshold = null, Device? device = null)
    {
        if (bandwidth.Length != dimensions)
        {
            throw new ArgumentException("Bandwidth must be of the form (n_features) and match the number of dimensions");
        }

        _device = device ?? CPU;
        _distanceThreshold = distanceThreshold ?? double.NegativeInfinity;
        _dimensions = dimensions;
        _kernelBandwidth = tensor(bandwidth).to(_device);
    }

    /// <inheritdoc/>
    public override void Fit(Tensor data)
    {
        if (data.shape[1] != _dimensions)
        {
            throw new ArgumentException("Data shape must match expected dimensions");
        }

        for (int i = 0; i < data.shape[0]; i++)
        {
            var kernel = new WeightedGaussian(_weight, data[i], _kernelBandwidth);
            if (_kernels.Count == 0)
            {
                _kernels = [kernel];
                continue;
            }

            var dist = CalculateMahalanobisDistance(data[i]);
            var minDist = dist.min().ReadCpuSingle(0);
            if (minDist > _distanceThreshold)
            {
                _kernels.Add(kernel);
                continue;
            }
            var argminDist = (int)dist.argmin().item<long>();
            var kernelToMerge = _kernels[argminDist];
            _kernels.RemoveAt(argminDist);
            var mergedKernel = MergeKernels(kernel, kernelToMerge);
            _kernels.Add(mergedKernel);
        }
    }

    /// <inheritdoc/>
    public override void Clear()
    {
        _kernels = new();
    }

    private Tensor CalculateMahalanobisDistance(Tensor data)
    {
        using (var _ = NewDisposeScope())
        {
            var dist = empty(_kernels.Count);
            for (int i = 0; i < _kernels.Count; i++)
            {
                var kernel = _kernels[i];
                var mean = kernel.Mean;
                var diagonalCovariance = diag(kernel.DiagonalCovariance);
                var delta = data - mean;
                var sigmaInv = diagonalCovariance.inverse();
                var matMul = matmul(sigmaInv, delta);
                var temp = matmul(delta, matMul);
                dist[i] = sqrt(temp);
            }
            var flattened = dist.flatten();
            return flattened.MoveToOuterDisposeScope();
        }
    }

    private WeightedGaussian MergeKernels(WeightedGaussian kernel1, WeightedGaussian kernel2)
    {
        var weightSum = kernel1.Weight + kernel2.Weight;
        var mean = (kernel1.Mean * kernel1.Weight + kernel2.Mean * kernel2.Weight) / weightSum;
        var variance = (kernel1.DiagonalCovariance + pow(kernel1.Mean, 2)) * kernel1.Weight + (kernel2.DiagonalCovariance + pow(kernel2.Mean, 2)) * kernel2.Weight;
        var diagonalCovariance = variance / weightSum - pow(mean, 2);
        return new WeightedGaussian(
            weightSum, 
            mean, 
            diagonalCovariance
        );
    }

    /// <inheritdoc/>
    public override Tensor Evaluate(Tensor min, Tensor max, Tensor steps)
    {
        using (var _ = NewDisposeScope())
        {
            var arrays = new Tensor[min.shape[0]];
            for (int i = 0; i < min.shape[0]; i++)
            {
                arrays[i] = linspace(min[i].item<double>(), max[i].item<double>(), steps[i].item<long>()).to(_device);
            }
            var grid = meshgrid(arrays);
            var points = vstack(grid.Select(tensor => tensor.flatten()).ToList()).T;
            var evaluatedPoints = Evaluate(points);
            var dimensions = steps.data<long>().ToArray();
            var reshaped = evaluatedPoints.reshape(dimensions);
            return reshaped.MoveToOuterDisposeScope();
        }
    }

    /// <inheritdoc/>
    public override Tensor Evaluate(Tensor points)
    {
        using (var _ = NewDisposeScope())
        {
            var densities = new Tensor[_kernels.Count];
            for (int i = 0; i < _kernels.Count; i++)
            {
                var kernel = _kernels[i];
                var differences = kernel.Mean.unsqueeze(0) - points.unsqueeze(1);
                var squareDistances = pow(differences, 2);
                var normedSquareDistances = squareDistances / kernel.DiagonalCovariance;
                var sumDistances = sum(normedSquareDistances, dim: 2);
                var rawDensity = exp(-0.5 * sumDistances);
                var kernelDensity = rawDensity / sqrt(2 * Math.PI * kernel.DiagonalCovariance.prod());
                densities[i] = kernel.Weight * kernelDensity;
            }
            var density = sum(stack(densities), dim: 0);
            var normed = density / density.sum();
            return normed.MoveToOuterDisposeScope();
        }
    }
}
