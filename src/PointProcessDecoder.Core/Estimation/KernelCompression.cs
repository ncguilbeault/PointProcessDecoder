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
    /// <inheritdoc/>
    public ScalarType ScalarType => _scalarType;

    private readonly Tensor _kernelBandwidth;
    /// <inheritdoc/>
    public Tensor KernelBandwidth => _kernelBandwidth;

    private Tensor _kernels = empty(0);
    /// <summary>
    /// The kernels.
    /// </summary>
    public Tensor Kernels => _kernels;

    private readonly double _distanceThreshold;
    private readonly Tensor _weight;

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
        _distanceThreshold = distanceThreshold ?? double.NegativeInfinity;
        _dimensions = dimensions ?? 1;

        _weight = ones(_dimensions)
            .to_type(_scalarType)
            .to(_device);

        _kernelBandwidth = tensor(bandwidth ?? 1.0)
            .repeat(_dimensions)
            .to_type(_scalarType)
            .to(_device);
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
        _distanceThreshold = distanceThreshold ?? double.NegativeInfinity;
        _dimensions = dimensions;

        _weight = ones(_dimensions)
            .to_type(_scalarType)
            .to(_device);

        _kernelBandwidth = tensor(bandwidth)
            .to_type(_scalarType)
            .to(_device);
    }

    /// <inheritdoc/>
    public void Fit(Tensor data)
    {
        if (data.shape[1] != _dimensions)
        {
            throw new ArgumentException("Data shape must match expected dimensions");
        }

        using var _ = NewDisposeScope();
        var count = data.shape[0];
        data = data.to_type(_scalarType).to(_device);
        var kernel = concat([_weight.unsqueeze(1), data[0].unsqueeze(1), _kernelBandwidth.unsqueeze(1)], dim: 1);

        if (_kernels.numel() == 0)
        {
            _kernels = kernel.unsqueeze(0);
            if (count == 1) return;
            data = data[TensorIndex.Slice(1)];
        }

        for (int i = 0; i < data.shape[0]; i++)
        {
            var mahalanobisDistance = CalculateMahalanobisDistance(data[i]);
            var (minDist, argminDist) = mahalanobisDistance.min(0);

            if ((minDist > _distanceThreshold).item<bool>())
            {
                _kernels = cat([_kernels, kernel.unsqueeze(0)], dim: 0);
                continue;
            }

            var mergedKernel = MergeKernels(kernel, _kernels[argminDist]);
            _kernels[argminDist] = mergedKernel;
        }
        _kernels.MoveToOuterDisposeScope();
    }

    /// <inheritdoc/>
    public void Clear()
    {
        _kernels.Dispose();
        _kernels = empty(0);
    }

    private Tensor CalculateMahalanobisDistance(Tensor data)
    {
        using var _ = NewDisposeScope();
        var diff = pow(data.unsqueeze(0) - _kernels[TensorIndex.Ellipsis, 1], 2);
        var mahalanobisDistance = sqrt(sum(diff / _kernels[TensorIndex.Ellipsis, 2], dim: -1));
        return mahalanobisDistance.MoveToOuterDisposeScope();
    }

    private Tensor MergeKernels(Tensor kernel, Tensor previousKernel)
    {
        using var _ = NewDisposeScope();
        var newWeight = previousKernel[TensorIndex.Ellipsis, 0] + kernel[TensorIndex.Ellipsis, 0];

        var previousMean = previousKernel[TensorIndex.Ellipsis, 1] * previousKernel[TensorIndex.Ellipsis, 0];
        var newMean = (previousMean + kernel[TensorIndex.Ellipsis, 1]) / newWeight;

        var previousDiagonalCovariance = (previousKernel[TensorIndex.Ellipsis, 2] + pow(previousKernel[TensorIndex.Ellipsis, 1], 2)) * previousKernel[TensorIndex.Ellipsis, 0];
        var variance = previousDiagonalCovariance + (kernel[TensorIndex.Ellipsis, 2] + pow(kernel[TensorIndex.Ellipsis, 1], 2));
        var newDiagonalCovariance = variance / newWeight - pow(newMean, 2);

        return concat([newWeight.unsqueeze(1), newMean.unsqueeze(1), newDiagonalCovariance.unsqueeze(1)], dim: 1)
            .MoveToOuterDisposeScope();
    }

    /// <inheritdoc/>
    public Tensor Evaluate(Tensor min, Tensor max, Tensor steps)
    {
        if (min.shape[0] != _dimensions || max.shape[0] != _dimensions || steps.shape[0] != _dimensions)
        {
            throw new ArgumentException("The lengths of min, max, and steps must be equal to the number of dimensions.");
        }

        if (min.dtype != ScalarType.Float64)
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
            arrays[i] = linspace(min[i].item<double>(), max[i].item<double>(), steps[i].item<long>(), dtype: _scalarType, device: _device);
        }
        var grid = meshgrid(arrays);
        var points = vstack(grid.Select(tensor => tensor.flatten()).ToList()).T;
        var evaluatedPoints = Evaluate(points);
        var dimensions = steps.data<long>().ToArray();
        return evaluatedPoints
            .reshape(dimensions)
            .MoveToOuterDisposeScope();
    }

    /// <inheritdoc/>
    public Tensor Evaluate(Tensor points)
    {
        if (points.shape[1] != _dimensions)
        {
            throw new ArgumentException("The number of dimensions must match the shape of the data.");
        }

        using var _ = NewDisposeScope();
        points = points.to_type(_scalarType).to(_device);
        var diff = pow(_kernels[TensorIndex.Ellipsis, 1] - points.unsqueeze(1), 2);
        var gaussian = exp(-0.5 * sum(diff / _kernels[TensorIndex.Ellipsis, 2], dim: -1));
        var kernelWeights = _kernels[TensorIndex.Ellipsis, 0];
        var kernelSqrtDiag = sqrt(2 * Math.PI * _kernels[TensorIndex.Ellipsis, 2].prod(dim: 1));
        var kernelDensity = kernelWeights.T * gaussian / kernelSqrtDiag.unsqueeze(0);
        var density = sum(kernelDensity, dim: 1);
        var normed = density / density.sum();
        return normed
            .to_type(_scalarType)
            .to(_device)
            .MoveToOuterDisposeScope();
    }
}
