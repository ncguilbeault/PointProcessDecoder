using PointProcessDecoder.Core.Estimation;
using static TorchSharp.torch;

namespace PointProcessDecoder.Core.Encoder;

public abstract class EncoderModel
{
    public abstract Device Device { get; }
    // public abstract DensityEstimation DensityEstimation { get; }
    // public abstract Tensor GetEncoder(Tensor input);
    public abstract IEnumerable<Tensor> Evaluate();
    public abstract IEnumerable<Tensor> Evaluate(Tensor min, Tensor max, Tensor steps);
    public abstract void Encode(Tensor observations, Tensor input);
}
