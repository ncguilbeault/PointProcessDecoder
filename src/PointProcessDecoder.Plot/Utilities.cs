using OxyPlot;

namespace PointProcessDecoder.Plot;

/// <summary>
/// A class containing utility methods for the Plot namespace.
/// </summary>
public static class Utilities
{
    private static Random random = new();

    /// <summary>
    /// Generates a list of colors.
    /// </summary>
    /// <param name="n"></param>
    /// <returns></returns>
    public static List<OxyColor> GenerateRandomColors(int n, int? seed = null)
    {
        if (seed.HasValue) 
            random = new Random(seed.Value);

        var colors = new List<OxyColor>();
        for (int i = 0; i < n; i++)
        {
            colors.Add(OxyColor.FromHsv((double)i / n, 1, 1));
        }

        for (int i = 0; i < n; i++)
        {
            int j = random.Next(i, n);
            (colors[j], colors[i]) = (colors[i], colors[j]);
        }

        return colors;
    }
}