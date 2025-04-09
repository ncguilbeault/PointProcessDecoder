namespace PointProcessDecoder.Core.Transitions;

/// <summary>
/// Represents the transitions type of the model.
/// </summary>
public enum TransitionsType
{
    /// <summary>
    /// Represents uniform state transitions.
    /// </summary>
    Uniform,

    /// <summary>
    /// Represents random walk state transitions.
    /// </summary>
    RandomWalk,

    /// <summary>
    /// Represents stationary state transitions.
    /// </summary>
    Stationary,

    ReciprocalGaussian
}
