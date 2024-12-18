using System;
using static TorchSharp.torch;
using TorchSharp;

namespace PointProcessDecoder.Core.Estimation;

/// <summary>
/// Kernel density estimation.
/// </summary>
public class KernelDensity : IEstimation
{
    private readonly Device _device;
    /// <summary>
    /// The device on which the data is stored.
    /// </summary>
    public Device Device => _device;

    private readonly ScalarType _scalarType;
    public ScalarType ScalarType => _scalarType;

    private readonly Tensor _kernelBandwidth;
    /// <summary>
    /// The kernel bandwidth.
    /// </summary>
    public Tensor KernelBandwidth => _kernelBandwidth;

    private double _tolerance = 1e-12;
    /// <summary>
    /// The tolerance.
    /// </summary>
    public double Tolerance 
    { 
        get => _tolerance; 
        set => _tolerance = value; 
    }

    private readonly int _dimensions;
    /// <summary>
    /// The number of dimensions of the observations.
    /// </summary>
    public int Dimensions => _dimensions;

    private Tensor _kernels = empty(0);
    /// <summary>
    /// The kernels.
    /// </summary>
    public Tensor Kernels => _kernels;
    

    /// <summary>
    /// Initializes a new instance of the <see cref="KernelDensity"/> class.
    /// </summary>
    /// <param name="bandwidth"></param>
    /// <param name="dimensions"></param>
    /// <param name="device"></param>
    public KernelDensity(
        double? bandwidth = null, 
        int? dimensions = null, 
        Device? device = null,
        ScalarType? scalarType = null
    )
    {
        _dimensions = dimensions ?? 1;
        _device = device ?? CPU;
        _scalarType = scalarType ?? ScalarType.Float32;
        _kernelBandwidth = tensor(bandwidth ?? 1.0, device: _device, dtype: _scalarType)
            .repeat(_dimensions);
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
        _kernelBandwidth = tensor(bandwidth, device: _device, dtype: _scalarType);
    }

    /// <summary>
    /// Add a new data point to the density estimation.
    /// </summary>
    /// <param name="data"></param>
    /// <exception cref="ArgumentException"></exception>
    public void Fit(Tensor data) 
    {
        if (data.shape[1] != _dimensions)
        {
            throw new ArgumentException("The number of dimensions must match the shape of the data.");
        }

        // data = data.to_type(_scalarType).to(_device);

        // Check if the kernels are empty
        if (_kernels.numel() == 0)
        {
            _kernels = data;
            return;      
        }

        // Concatenate the data points with the kernels
        _kernels = cat([ _kernels, data ], dim: 0);
    }

    /// <summary>
    /// Clear the density estimation kernels.
    /// </summary>
    public void Clear() 
    {
        _kernels.Dispose();
        _kernels = empty(0);
    }

    /// <summary>
    /// Evaluate the density estimation at the given points.
    /// </summary>
    /// <param name="min"></param>
    /// <param name="max"></param>
    /// <param name="steps"></param>
    /// <returns></returns>
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

    /// <summary>
    /// Evaluate the density estimation at the given points.
    /// </summary>
    /// <param name="points"></param>
    /// <returns></returns>
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
        var diff = (_kernels.unsqueeze(0) - points.unsqueeze(1)) / _kernelBandwidth;
        var density = mean(exp(-0.5 * diff.pow(2).sum(2)), [1]) / _kernelBandwidth.prod();
        density /= sum(density);
        return density
            .nan_to_num()
            .to_type(_scalarType)
            .to(_device)
            .MoveToOuterDisposeScope();
    }
}
