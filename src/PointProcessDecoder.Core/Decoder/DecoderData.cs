using static TorchSharp.torch;

namespace PointProcessDecoder.Core.Decoder;

public struct DecoderData
{
    /// <summary>
    /// Initializes a new instance of the <see cref="DecoderData"/> struct.
    /// </summary>
    public DecoderData(IStateSpace stateSpace, Tensor posterior)
    {
        Posterior = posterior;
        CenterOfMass = CalculateCenterOfMass(stateSpace, posterior);
        MaximumAPosterioriEstimate = CalculateMaximumAPosterioriEstimate(stateSpace, posterior);
        Spread = CalculateSpread(stateSpace, posterior, CenterOfMass);
    }

    /// <summary>
    /// Gets the posterior tensor.
    /// </summary>
    public Tensor Posterior { get; set; }

    /// <summary>
    /// Gets the center of mass of the posterior.
    /// </summary>
    public Tensor CenterOfMass { get; set; }

    /// <summary>
    /// Gets the maximum a posteriori estimate of the posterior.
    /// </summary>
    public Tensor MaximumAPosterioriEstimate { get; set; }

    /// <summary>
    /// Gets the spread of the posterior distribution.
    /// </summary>
    public Tensor Spread { get; set; }

    public static Tensor CalculateCenterOfMass(IStateSpace stateSpace, Tensor posterior)
    {
        var mass = posterior.flatten(1).unsqueeze(-1);
        var points = stateSpace.Points.unsqueeze(0);
        var com = sum(points * mass, 1) / sum(mass, 1);
        return sum(points * mass, 1) / sum(mass, 1);
    }

    public static Tensor CalculateMaximumAPosterioriEstimate(IStateSpace stateSpace, Tensor posterior)
    {
        var points = stateSpace.Points;
        var mass = posterior.flatten(1);
        var indices = argmax(mass, 1);
        return points[indices];
    }

    public static Tensor CalculateSpread(IStateSpace stateSpace, Tensor posterior, Tensor centerOfMass)
    {
        var center = centerOfMass.unsqueeze(1);
        var mass = posterior.flatten(1).unsqueeze(-1);
        var points = stateSpace.Points.unsqueeze(0);
        var squaredDifference = (points - center)
            .pow(2)
            .sum(dim: -1)
            .unsqueeze(-1);
        return (mass * squaredDifference)
            .sum(dim: 1)
            .sqrt();
    }
}
