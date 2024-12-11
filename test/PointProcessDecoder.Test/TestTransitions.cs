using static TorchSharp.torch;
using PointProcessDecoder.Core;
using PointProcessDecoder.Plot;
using PointProcessDecoder.Simulation;
using PointProcessDecoder.Core.Transitions;

namespace PointProcessDecoder.Test;

[TestClass]
public class TestTransitions
{

    private string outputDirectory = "TestTransitions";

    [TestMethod]
    public void TestUniform1D()
    {
        var steps = 200;
        var min = 0.0;
        var max = 100.0;

        var uniformTransitions = new UniformTransitions(min, max, steps);
        Heatmap transitionsPlot = new(min, max, min, max, title: "UniformTransitions1D");
        transitionsPlot.OutputDirectory = Path.Combine(transitionsPlot.OutputDirectory, outputDirectory);
        transitionsPlot.Show<float>(uniformTransitions.Values);
        transitionsPlot.Save(png: true);
    }

    [TestMethod]
    public void TestUniform2D()
    {
        var dimensions = 2;
        var steps = new long[] { 100, 100 };
        var min = new double[] { 0, 0 };
        var max = new double[] { 100, 100 };

        var uniformTransitions = new UniformTransitions(dimensions, min, max, steps);
        Heatmap transitionsPlot = new(min[0], max[0], min[1], max[1], title: "UniformTransitions2D");
        transitionsPlot.OutputDirectory = Path.Combine(transitionsPlot.OutputDirectory, outputDirectory);
        transitionsPlot.Show<float>(uniformTransitions.Values);
        transitionsPlot.Save(png: true);
    }

    [TestMethod]
    public void TestRandomWalk1D()
    {
        var steps = 200;
        var min = 0.0;
        var max = 100.0;

        var randomWalkTransitions = new RandomWalkTransitions(min, max, steps);
        Heatmap transitionsPlot = new(min, max, min, max, title: "RandomWalkTransitions1D");
        transitionsPlot.OutputDirectory = Path.Combine(transitionsPlot.OutputDirectory, outputDirectory);
        transitionsPlot.Show<float>(randomWalkTransitions.Values);
        transitionsPlot.Save(png: true);
    }

    [TestMethod]
    public void TestRandomWalk1DWithSigma()
    {
        var steps = 200;
        var min = 0.0;
        var max = 100.0;
        var sigma = 5.0;

        var randomWalkTransitions = new RandomWalkTransitions(min, max, steps, sigma);
        Heatmap transitionsPlot = new(min, max, min, max, title: "RandomWalkTransitions1DWithSigma");
        transitionsPlot.OutputDirectory = Path.Combine(transitionsPlot.OutputDirectory, outputDirectory);
        transitionsPlot.Show<double>(randomWalkTransitions.Values);
        transitionsPlot.Save(png: true);
    }

    [TestMethod]
    public void TestRandomWalk2D()
    {
        var dimensions = 2;
        var steps = new long[] { 100, 100 };
        var min = new double[] { 0, 0 };
        var max = new double[] { 100, 100 };
        var sigma = new double[] { 5, 1 };

        var randomWalkTransitions = new RandomWalkTransitions(dimensions, min, max, steps, sigma);
        Heatmap transitionsPlot = new(
            min[0], 
            max[0] * max[1], 
            min[1], 
            max[0] * max[1],
            title: "RandomWalkTransitions2D"
        );
        transitionsPlot.OutputDirectory = Path.Combine(transitionsPlot.OutputDirectory, outputDirectory);
        transitionsPlot.Show<double>(randomWalkTransitions.Values);
        transitionsPlot.Save(png: true);
    }
}