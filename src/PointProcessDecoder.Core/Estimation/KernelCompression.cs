using System;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;

namespace PointProcessDecoder.Core.Estimation;

/// <summary>
/// Kernel density estimation with gaussian kernel compression.
/// </summary>
public class KernelCompression : IEstimation
{
    private readonly Device _device;
    /// <inheritdoc/>
    public Device Device => _device;

    private readonly ScalarType _scalarType;
    public ScalarType ScalarType => _scalarType;

    public EstimationMethod EstimationMethod => EstimationMethod.KernelCompression;

    private Tensor _kernels = empty(0);
    /// <summary>
    /// The weighted gaussian components.
    /// </summary>
    // public List<WeightedGaussian> GaussianKernels => _kernels;
    public Tensor Kernels => _kernels;

    private readonly Tensor _kernelBandwidth;
    /// <inheritdoc/>
    public Tensor KernelBandwidth => _kernelBandwidth;

    private readonly double _distanceThreshold;
    public double DistanceThreshold => _distanceThreshold;

    private readonly Tensor _weight;
    public Tensor InitialWeight => _weight;


    private readonly int _dimensions;
    /// <summary>
    /// The number of dimensions of the observations.
    /// </summary>
    public int Dimensions => _dimensions;

    private readonly double _eps;

    /// <summary>
    /// Initializes a new instance of the <see cref="KernelCompression"/> class.
    /// </summary>
    /// <param name="bandwidth"></param>
    /// <param name="dimensions"></param>
    /// <param name="distanceThreshold"></param>
    /// <param name="device"></param>
    public KernelCompression(
        double? bandwidth = null, 
        int? dimensions = null,
        double? distanceThreshold = null,
        Device? device = null, 
        ScalarType? scalarType = null
    )
    {
        _device = device ?? CPU;
        _scalarType = scalarType ?? ScalarType.Float32;
        _eps = finfo(_scalarType).eps;
        _distanceThreshold = distanceThreshold ?? double.NegativeInfinity;
        _dimensions = dimensions ?? 1;
        _kernelBandwidth = tensor(bandwidth ?? 1.0, device: _device, dtype: _scalarType)
            .repeat(_dimensions);
        _weight = ones(_dimensions, dtype: _scalarType, device: _device);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="KernelCompression"/> class.
    /// </summary>
    /// <param name="bandwidth"></param>
    /// <param name="dimensions"></param>
    /// <param name="distanceThreshold"></param>
    /// <param name="device"></param>
    /// <exception cref="ArgumentException"></exception>
    public KernelCompression(
        double[] bandwidth, 
        int dimensions, 
        double? distanceThreshold = null,
        Device? device = null, 
        ScalarType? scalarType = null
    )
    {
        if (bandwidth.Length != dimensions)
        {
            throw new ArgumentException("Bandwidth must be of the form (n_features) and match the number of dimensions");
        }

        _device = device ?? CPU;
        _scalarType = scalarType ?? ScalarType.Float32;
        _eps = finfo(_scalarType).eps;
        _distanceThreshold = distanceThreshold ?? double.NegativeInfinity;
        _dimensions = dimensions;
        _kernelBandwidth = tensor(bandwidth, device: _device, dtype: _scalarType);
        _weight = ones(_dimensions, dtype: _scalarType, device: _device);
    }

    /// <inheritdoc/>
    public void Fit(Tensor data)
    {
        if (data.shape[1] != _dimensions)
        {
            throw new ArgumentException("Data shape must match expected dimensions");
        }
        if (data.shape[0] == 0) return;

        using var _ = NewDisposeScope();

        if (_kernels.numel() == 0)
        {
            _kernels = stack([_weight, data[0], _kernelBandwidth], dim: 1)
                .unsqueeze(0);
            if (data.shape[0] == 1) 
            {
                _kernels.MoveToOuterDisposeScope();
                return;
            }
            data = data[TensorIndex.Slice(1)];
        }

        for (int i = 0; i < data.shape[0]; i++)
        {
            var kernel = stack([_weight, data[i], _kernelBandwidth], dim: 1);
            var dist = CalculateMahalanobisDistance(data[i]);
            var (minDist, argminDist) = dist.min(0);
            if ((minDist > _distanceThreshold).item<bool>())
            {
                _kernels = concat([_kernels, kernel.unsqueeze(0)], dim: 0);
                continue;
            }
            _kernels[argminDist]  = MergeKernels(kernel, _kernels[argminDist]);
        }
        _kernels.MoveToOuterDisposeScope();
    }

    private Tensor CalculateMahalanobisDistance(Tensor data)
    {
        using var _ = NewDisposeScope();
        var diff = pow(data.unsqueeze(0) - _kernels[TensorIndex.Ellipsis, 1], 2);
        var dist = sqrt(sum(diff / _kernels[TensorIndex.Ellipsis, 2], dim: 1));
        return dist.MoveToOuterDisposeScope();
    }

    private static Tensor MergeKernels(Tensor kernel1, Tensor kernel2)
    {
        using var _ = NewDisposeScope();
        var weightSum = kernel1[TensorIndex.Ellipsis, 0] + kernel2[TensorIndex.Ellipsis, 0];
        var mean = (kernel1[TensorIndex.Ellipsis, 1] * kernel1[TensorIndex.Ellipsis, 0] + kernel2[TensorIndex.Ellipsis, 1] * kernel2[TensorIndex.Ellipsis, 0]) / weightSum;
        var variance = (kernel1[TensorIndex.Ellipsis, 2] + pow(kernel1[TensorIndex.Ellipsis, 1], 2)) * kernel1[TensorIndex.Ellipsis, 0] + (kernel2[TensorIndex.Ellipsis, 2] + pow(kernel2[TensorIndex.Ellipsis, 1], 2)) * kernel2[TensorIndex.Ellipsis, 0];
        var diagonalCovariance = variance / weightSum - pow(mean, 2);
        return stack([weightSum, mean, diagonalCovariance], dim: 1)
            .MoveToOuterDisposeScope();
    }

    public Tensor Estimate(Tensor points, int? dimensionStart = null, int? dimensionEnd = null)
    {
        using var _ = NewDisposeScope();
        if (_kernels.numel() == 0)
        {
            return (ones([1, 1], dtype: _scalarType, device: _device) * float.NaN)
                .MoveToOuterDisposeScope();
        }
        var kernels = _kernels[TensorIndex.Colon, TensorIndex.Slice(dimensionStart, dimensionEnd)];
        var diff = pow(kernels[TensorIndex.Ellipsis, 1].unsqueeze(0) - points.unsqueeze(1), 2);
        var gaussian = exp(-0.5 * sum(diff / kernels[TensorIndex.Ellipsis, 2], dim: -1));
        var sumWeights = kernels[TensorIndex.Ellipsis, 0, 0];
        var sqrtDiagonalCovariance = sqrt(pow(2 * Math.PI, _dimensions) * kernels[TensorIndex.Ellipsis, 2].prod(dim: -1));
        return (sumWeights * gaussian / sqrtDiagonalCovariance)
            .to_type(_scalarType)
            .to(_device)
            .MoveToOuterDisposeScope();
    }

    public Tensor Normalize(Tensor points)
    {
        using var _ = NewDisposeScope();
        var density = points.sum(dim: -1)
            .clamp_min(_eps);
        density /= density.sum();
        return density
            .to_type(_scalarType)
            .to(_device)
            .MoveToOuterDisposeScope();
    }

    /// <inheritdoc/>
    public Tensor Evaluate(Tensor min, Tensor max, Tensor steps)
    {
        if (min.shape[0] != _dimensions || max.shape[0] != _dimensions || steps.shape[0] != _dimensions)
        {
            throw new ArgumentException("The lengths of min, max, and steps must be equal to the number of dimensions.");
        }

        if (min.dtype != ScalarType.Float64 || max.dtype != ScalarType.Float64)
        {
            throw new ArgumentException("The scalar type of min and max must be float64.");
        }
        
        if (steps.dtype != ScalarType.Int64)
        {
            throw new ArgumentException("The scalar type of steps must be int64.");
        }

        using var _ = NewDisposeScope();
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

    /// <inheritdoc/>
    public Tensor Evaluate(Tensor points)
    {
        if (points.shape[1] != _dimensions)
        {
            throw new ArgumentException("The number of dimensions must match the shape of the data.");
        }

        if (_kernels.numel() == 0)
        {
            return zeros(points.shape[0], dtype: _scalarType, device: _device);
        }

        using var _ = NewDisposeScope();
        var estimate = Estimate(points);
        return Normalize(estimate)
            .MoveToOuterDisposeScope();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _kernels.Dispose();
    }
}

