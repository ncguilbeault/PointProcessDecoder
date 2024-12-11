using static TorchSharp.torch;

namespace PointProcessDecoder.Core.Estimation;

public class WeightedGaussian
{
    public Tensor Weight { get; set; }
    public Tensor Mean { get; set; }
    public Tensor DiagonalCovariance { get; set; }

    public WeightedGaussian(Tensor weight, Tensor mean, Tensor diagonalCovariance)
    {
        Weight = weight;
        Mean = mean;
        DiagonalCovariance = diagonalCovariance;
    }
}
