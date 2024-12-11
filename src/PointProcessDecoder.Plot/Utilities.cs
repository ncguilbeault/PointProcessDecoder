using OxyPlot;

namespace PointProcessDecoder.Plot;

/// <summary>
/// A class containing utility methods for the Plot namespace.
/// </summary>
public static class Utilities
{
    /// <summary>
    /// Generates a list of colors.
    /// </summary>
    /// <param name="n"></param>
    /// <returns></returns>
    public static List<OxyColor> GenerateRandomColors(int n)
    {
        var colors = new List<OxyColor>();
        for (int i = 0; i < n; i++)
        {
            colors.Add(OxyColor.FromHsv((double)i / n, 1, 1));
        }
        return colors;
    }
}