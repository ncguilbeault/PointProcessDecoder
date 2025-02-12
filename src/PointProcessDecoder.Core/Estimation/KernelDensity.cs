using System;
using static TorchSharp.torch;
using TorchSharp;

namespace PointProcessDecoder.Core.Estimation;

/// <summary>
/// Kernel density estimation.
/// </summary>
public class KernelDensity : ModelComponent, IEstimation
{
    private readonly Device _device;
    /// <inheritdoc/>
    public override Device Device => _device;

    private readonly ScalarType _scalarType;
    /// <inheritdoc/>
    public override ScalarType ScalarType => _scalarType;

    /// <inheritdoc/>
    public EstimationMethod EstimationMethod => EstimationMethod.KernelDensity;

    private readonly Tensor _kernelBandwidth;
    /// <inheritdoc/>
    public Tensor KernelBandwidth => _kernelBandwidth;

    private readonly int _dimensions;
    /// <inheritdoc/>
    public int Dimensions => _dimensions;

    private Tensor _kernels = empty(0);
    /// <inheritdoc/>
    public Tensor Kernels => _kernels;
    
    private readonly double _eps;
    private readonly int _kernelLimit;

    /// <summary>
    /// Initializes a new instance of the <see cref="KernelDensity"/> class.
    /// </summary>
    /// <param name="bandwidth"></param>
    /// <param name="dimensions"></param>
    /// <param name="device"></param>
    public KernelDensity(
        double? bandwidth = null, 
        int? dimensions = null,
        int? kernelLimit = null,
        Device? device = null,
        ScalarType? scalarType = null
    )
    {
        _dimensions = dimensions ?? 1;
        _device = device ?? CPU;
        _scalarType = scalarType ?? ScalarType.Float32;
        _eps = finfo(_scalarType).eps;
        _kernelBandwidth = tensor(bandwidth ?? 1.0, device: _device, dtype: _scalarType)
            .repeat(_dimensions);
        _kernelLimit = kernelLimit ?? int.MaxValue;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="KernelDensity"/> class.
    /// </summary>
    /// <param name="bandwidth"></param>
    /// <param name="dimensions"></param>
    /// <param name="device"></param>
    /// <exception cref="ArgumentException"></exception>
    public KernelDensity(
        double[] bandwidth, 
        int dimensions,
        int? kernelLimit = null,
        Device? device = null,
        ScalarType? scalarType = null
    )
    {
        if (bandwidth.Length != dimensions)
        {
            throw new ArgumentException("Bandwidth must be of the form (n_features) and match the number of dimensions");
        }

        _dimensions = dimensions;
        _device = device ?? CPU;
        _scalarType = scalarType ?? ScalarType.Float32;
        _eps = finfo(_scalarType).eps;
        _kernelBandwidth = tensor(bandwidth, device: _device, dtype: _scalarType);
        _kernelLimit = kernelLimit ?? int.MaxValue;
    }

    /// <inheritdoc/>
    public void Fit(Tensor data) 
    {
        if (data.shape[1] != _dimensions)
        {
            throw new ArgumentException("The number of dimensions must match the shape of the data.");
        }

        using var _ = NewDisposeScope();

        if (_kernels.numel() == 0)
        {
            _kernels = data.MoveToOuterDisposeScope();
            return;      
        }

        _kernels = cat([ _kernels, data ], dim: 0);

        if (_kernels.shape[0] > _kernelLimit)
        {
            var start = _kernels.shape[0] - _kernelLimit;
            _kernels = _kernels[TensorIndex.Slice(start)];
        }

        
        _kernels.MoveToOuterDisposeScope();
    }

    /// <inheritdoc/>
    public Tensor Estimate(Tensor points, int? dimensionStart = null, int? dimensionEnd = null)
    {
        using var _ = NewDisposeScope();
        if (_kernels.numel() == 0)
        {
            return (ones([1, 1], dtype: _scalarType, device: _device) * float.NaN)
                .MoveToOuterDisposeScope();
        }
        var kernels = _kernels[TensorIndex.Colon, TensorIndex.Slice(dimensionStart, dimensionEnd)];
        var dist = (kernels.unsqueeze(0) - points.unsqueeze(1)) / _kernelBandwidth;
        var sumSquaredDiff = dist
            .pow(exponent: 2)
            .sum(dim: -1);
        var estimate = exp(-0.5 * sumSquaredDiff);
        var sqrtDiagonalCovariance = sqrt(pow(2 * Math.PI, _dimensions) * _kernelBandwidth.prod(dim: -1));
        return (estimate / sqrtDiagonalCovariance)
            .to_type(_scalarType)
            .to(_device)
            .MoveToOuterDisposeScope();
    }

    /// <inheritdoc/>
    public Tensor Normalize(Tensor points)
    {
        using var _ = NewDisposeScope();

        var density = (points.sum(dim: -1)
            / points.shape[1])
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
        var reshaped = evaluatedPoints.reshape(dimensions);
        return reshaped
            .to_type(_scalarType)
            .to(_device)
            .MoveToOuterDisposeScope();
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
    public override void Save(string basePath)
    {
        _kernels.Save(Path.Combine(basePath, "kernels.bin"));
    }

    /// <inheritdoc/>
    public override IModelComponent Load(string basePath)
    {
        _kernels = Tensor.Load(Path.Combine(basePath, "kernels.bin")).to(_device);
        return this;
    }
}
