using static TorchSharp.torch;

namespace PointProcessDecoder.Core.Decoder;

public struct ClassifierData
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ClassifierData"/> struct.
    /// </summary>
    public ClassifierData(IStateSpace stateSpace, Tensor posterior)
    {
        DecoderData = new DecoderData(stateSpace, posterior.sum(dim: 1));
        var sumOverDims = arange(2, posterior.Dimensions, dtype: ScalarType.Int64)
            .data<long>()
            .ToArray();
        StateProbabilities = posterior.sum(sumOverDims);
    }

    /// <summary>
    /// Gets the posterior tensor.
    /// </summary>
    public DecoderData DecoderData { get; set; }

    /// <summary>
    /// Gets the center of mass of the posterior.
    /// </summary>
    public Tensor StateProbabilities { get; set; }
}
