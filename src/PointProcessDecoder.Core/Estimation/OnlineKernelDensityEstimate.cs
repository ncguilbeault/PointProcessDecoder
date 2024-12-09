using System;
using static TorchSharp.torch;
using TorchSharp;

namespace PointProcessDecoder.Core.Estimation;

/// <summary>
/// Online kernel density estimation.
/// </summary>
public class OnlineKernelDensityEstimate : OnlineDensityEstimation
{
    private readonly Device _device;
    /// <summary>
    /// The device on which the data is stored.
    /// </summary>
    public override Device Device => _device;

    private readonly Tensor _kernelBandwidth;
    /// <summary>
    /// The kernel bandwidth.
    /// </summary>
    public override Tensor KernelBandwidth => _kernelBandwidth;

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
    /// Initializes a new instance of the <see cref="OnlineKernelDensityEstimate"/> class.
    /// </summary>
    public OnlineKernelDensityEstimate()
    {
        _dimensions = 1;
        _device = CPU;
        _kernelBandwidth = tensor(1.0).to(_device);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="OnlineKernelDensityEstimate"/> class.
    /// </summary>
    /// <param name="bandwidth"></param>
    /// <param name="dimensions"></param>
    /// <param name="device"></param>
    public OnlineKernelDensityEstimate(double bandwidth, int dimensions, Device? device = null)
    {
        _dimensions = dimensions;
        _device = device ?? CPU;
        _kernelBandwidth = tensor(bandwidth).to(_device);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="OnlineKernelDensityEstimate"/> class.
    /// </summary>
    /// <param name="bandwidth"></param>
    /// <param name="dimensions"></param>
    /// <param name="device"></param>
    public OnlineKernelDensityEstimate(Tensor bandwidth, int dimensions, Device? device = null)
    {
        _dimensions = dimensions;
        _device = device ?? CPU;
        _kernelBandwidth = bandwidth.to(_device);
    }

    /// <summary>
    /// Add a new data point to the density estimation.
    /// </summary>
    /// <param name="data"></param>
    /// <exception cref="ArgumentException"></exception>
    public override void Add(Tensor data) 
    {
        // Check if the data shape matches the bandwidth shape
        if (data.shape[0] != _dimensions)
        {
            throw new ArgumentException("Data shape must match bandwidth shape");
        }

        // Check if the kernels are empty
        if (_kernels.numel() == 0)
        {
            _kernels = data.unsqueeze(0);
            return;      
        }

        // Concatenate the data points with the kernels
        _kernels = cat([ _kernels, data ], dim: 0);
    }

    /// <summary>
    /// Clear the density estimation kernels.
    /// </summary>
    public override void Clear() 
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

    /// <summary>
    /// Evaluate the density estimation at the given points.
    /// </summary>
    /// <param name="points"></param>
    /// <returns></returns>
    public override Tensor Evaluate(Tensor points)
    {
        using (var _ = NewDisposeScope())
        {
            var differences = _kernels.unsqueeze(0) - points.unsqueeze(1);
            var distances = differences / _kernelBandwidth;
            var squareDistances = pow(distances, 2);
            var sumDistances = sum(squareDistances, dim: 2);
            var values = exp(-0.5 * sumDistances);
            var meanValues = mean(values, dimensions: [ 1 ]);
            var kernelBandwidthProduct = prod(_kernelBandwidth);
            var kdeValues = meanValues / kernelBandwidthProduct;
            var normalizedKdeValues = kdeValues / sum(kdeValues) + _tolerance;
            return normalizedKdeValues.MoveToOuterDisposeScope();
        }
    }
}
