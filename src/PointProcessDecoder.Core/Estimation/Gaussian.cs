using static TorchSharp.torch;

namespace PointProcessDecoder.Core.Estimation;

public class Gaussian
{
    public Tensor Weight { get; set; }
    public Tensor Mean { get; set; }
    public Tensor DiagonalCovariance { get; set; }

    public Gaussian(Tensor weight, Tensor mean, Tensor diagonalCovariance)
    {
        Weight = weight;
        Mean = mean;
        DiagonalCovariance = diagonalCovariance;
    }
}
