using System;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;

namespace PointProcessDecoder.Core.Estimation;

/// <summary>
/// Online kernel density estimation with Gaussian kernel compression.
/// </summary>
public class OnlineKernelCompression : OnlineDensityEstimation
{
    private readonly Device _device;
    /// <inheritdoc/>
    public override Device Device => _device;

    private List<Gaussian> _kernels = new();
    /// <summary>
    /// The Gaussian components.
    /// </summary>
    public List<Gaussian> Kernels => _kernels;

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
    /// Initializes a new instance of the <see cref="OnlineKernelCompression"/> class.
    /// </summary>
    public OnlineKernelCompression()
    {
        _device = CPU;
        _kernelBandwidth = tensor(1.0).to(_device);
        _distanceThreshold = double.NaN;
        _dimensions = 1;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="OnlineKernelCompression"/> class.
    /// </summary>
    /// <param name="bandwidth"></param>
    /// <param name="dimensions"></param>
    /// <param name="distanceThreshold"></param>
    /// <param name="device"></param>
    public OnlineKernelCompression(double bandwidth, int dimensions, double distanceThreshold, Device? device = null)
    {
        _device = device ?? CPU;
        _kernelBandwidth = tensor(bandwidth).to(_device);
        _distanceThreshold = distanceThreshold;
        _dimensions = dimensions;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="OnlineKernelCompression"/> class.
    /// </summary>
    /// <param name="bandwidth"></param>
    /// <param name="dimensions"></param>
    /// <param name="distanceThreshold"></param>
    /// <param name="device"></param>
    public OnlineKernelCompression(Tensor bandwidth, int dimensions, double distanceThreshold, Device? device = null)
    {
        _device = CPU;
        _kernelBandwidth = bandwidth;
        _distanceThreshold = distanceThreshold;
        _dimensions = dimensions;
    }

    /// <inheritdoc/>
    public override void Add(Tensor data)
    {
        if (data.shape[0] != _dimensions)
        {
            throw new ArgumentException("Data shape must match bandwidth shape");
        }

        var kernel = new Gaussian(_weight, data, _kernelBandwidth);
        if (_kernels is null)
        {
            _kernels = [kernel];
            return;
        }

        var dist = CalculateMahalanobisDistance(data);
        var minDist = dist.min().ReadCpuSingle(0);
        if (minDist > _distanceThreshold)
        {
            _kernels.Add(kernel);
            return;
        }
        var argminDist = (int)dist.argmin().item<long>();
        var kernelToMerge = _kernels[argminDist];
        _kernels.RemoveAt(argminDist);
        var mergedKernel = MergeKernels(kernel, kernelToMerge);
        _kernels.Add(mergedKernel);
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
                var mean = _kernels[i].Mean;
                var diagonalCovariance = diag(_kernels[i].DiagonalCovariance);
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

    private Gaussian MergeKernels(Gaussian kernel1, Gaussian kernel2)
    {
        var weightSum = kernel1.Weight + kernel2.Weight;
        var mean = (kernel1.Mean * kernel1.Weight + kernel2.Mean * kernel2.Weight) / weightSum;
        var variance = (kernel1.DiagonalCovariance + pow(kernel1.Mean, 2)) * kernel1.Weight + (kernel2.DiagonalCovariance + pow(kernel2.Mean, 2)) * kernel2.Weight;
        var diagonalCovariance = variance / weightSum - pow(mean, 2);
        return new Gaussian(
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
            return density.MoveToOuterDisposeScope();
        }
    }
}
