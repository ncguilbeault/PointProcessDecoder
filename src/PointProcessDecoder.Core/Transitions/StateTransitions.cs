using static TorchSharp.torch;

namespace PointProcessDecoder.Core.Transitions;

public abstract class StateTransitions : IStateTransitions
{
    /// <summary>
    /// The device on which the tensor is stored.
    /// </summary>
    public abstract Device Device { get; }

    /// <summary>
    /// The scalar type of the tensor.
    /// </summary>
    public abstract ScalarType ScalarType { get; }

    /// <summary>
    /// The number of dimensions in the latent space.
    /// </summary>
    public abstract int Dimensions { get; }

    /// <summary>
    /// The points in the latent space. Points should be in the form of a tensor with shape (n, d) where n is the number of points and d is the dimensionality of the latent space.
    /// </summary>
    public abstract Tensor Points { get; }

    /// <summary>
    /// The transitions in the latent space.
    /// </summary>
    public abstract Tensor Transitions { get; }

    /// <summary>
    /// Method to compute the latent space given the minimum, maximum, and steps.
    /// </summary>
    /// <param name="min"></param>
    /// <param name="max"></param>
    /// <param name="steps"></param>
    /// <param name="device"></param>
    /// <returns></returns>
    public static Tensor ComputeLatentSpace(
        double[] min, 
        double[] max, 
        long[] steps,
        Device device,
        ScalarType scalarType
    )
    {
        var dims = min.Length;
        using var _ = NewDisposeScope();
        var arrays = new Tensor[dims];
        for (int i = 0; i < dims; i++)
        {
            arrays[i] = linspace(min[i], max[i], steps[i]);
        }
        var grid = meshgrid(arrays);
        var gridSpace = vstack(grid.Select(tensor => tensor.flatten()).ToList()).T;
        return gridSpace
            .to_type(scalarType)
            .to(device)
            .MoveToOuterDisposeScope();
    }
}
