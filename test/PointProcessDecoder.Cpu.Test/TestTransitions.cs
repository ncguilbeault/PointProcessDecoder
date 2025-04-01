using PointProcessDecoder.Test.Common;
using PointProcessDecoder.Core;
using static TorchSharp.torch;

using OxyPlot;
using OxyPlot.Annotations;
using OxyPlot.Series;
using OxyPlot.Axes;
using PointProcessDecoder.Core.Decoder;
using PointProcessDecoder.Plot;

namespace PointProcessDecoder.Cpu.Test;

[TestClass]
public class TestTransitions
{
    [TestMethod]
    public void TestUniform()
    {
        var evaluationSteps = new long[] { 50, 50 };
        var min = new double[] { 0, 0 };
        var max = new double[] { 100, 100 };

        var outputDirectory = Path.Combine("TestTransitions", "Uniform");

        var stateSpace = new Core.StateSpace.DiscreteUniform(
            dimensions: 2,
            min: min,
            max: max,
            steps: evaluationSteps
        );

        var uniformTransitions = new Core.Transitions.Uniform(
            stateSpace: stateSpace
        );

        int seed = 0;
        var generator = manual_seed(seed);
        var randomSelection = randint(0, stateSpace.Points.shape[0], 40, generator: generator);

        for (int i = 0; i < randomSelection.size(0); i++)
        {
            var index = randomSelection[i].item<long>();
            var transition = uniformTransitions.Transitions[index]
                .reshape(stateSpace.Shape);
            var transitionPlot = new Heatmap(
                title: $"Uniform Transition {index}",
                xMin: min[0],
                xMax: max[0],
                yMin: min[1],
                yMax: max[1]
            );
            transitionPlot.OutputDirectory = Path.Combine(transitionPlot.OutputDirectory, outputDirectory);
            transitionPlot.Show(transition);
            transitionPlot.Save(png: true);
        }
    }

    [TestMethod]
    public void TestRandomWalk()
    {
        var evaluationSteps = new long[] { 50, 50 };
        var min = new double[] { 0, 0 };
        var max = new double[] { 100, 100 };
        var sigma = 25;

        var outputDirectory = Path.Combine("TestTransitions", "RandomWalk");

        var stateSpace = new Core.StateSpace.DiscreteUniform(
            dimensions: 2,
            min: min,
            max: max,
            steps: evaluationSteps
        );

        var randomWalkTransitions = new Core.Transitions.RandomWalk(
            stateSpace: stateSpace,
            sigma: sigma
        );

        int seed = 0;
        var generator = manual_seed(seed);
        var randomSelection = randint(0, stateSpace.Points.shape[0], 40, generator: generator);

        for (int i = 0; i < randomSelection.size(0); i++)
        {
            var index = randomSelection[i].item<long>();
            var transition = randomWalkTransitions.Transitions[index]
                .reshape(stateSpace.Shape);
            var transitionPlot = new Heatmap(
                title: $"Random Walk Transition {index}",
                xMin: min[0],
                xMax: max[0],
                yMin: min[1],
                yMax: max[1]
            );
            transitionPlot.OutputDirectory = Path.Combine(transitionPlot.OutputDirectory, outputDirectory);
            transitionPlot.Show(transition);
            transitionPlot.Save(png: true);
        }
    }

    [TestMethod]
    public void TestStationary()
    {
        var evaluationSteps = new long[] { 50, 50 };
        var min = new double[] { 0, 0 };
        var max = new double[] { 100, 100 };

        var outputDirectory = Path.Combine("TestTransitions", "Stationary");

        var stateSpace = new Core.StateSpace.DiscreteUniform(
            dimensions: 2,
            min: min,
            max: max,
            steps: evaluationSteps
        );

        var stationaryTransitions = new Core.Transitions.Stationary(
            stateSpace: stateSpace
        );

        int seed = 0;
        var generator = manual_seed(seed);
        var randomSelection = randint(0, stateSpace.Points.shape[0], 40, generator: generator);

        for (int i = 0; i < randomSelection.size(0); i++)
        {
            var index = randomSelection[i].item<long>();
            var transition = stationaryTransitions.Transitions[index]
                .reshape(stateSpace.Shape);
            var transitionPlot = new Heatmap(
                title: $"Stationary Transition {index}",
                xMin: min[0],
                xMax: max[0],
                yMin: min[1],
                yMax: max[1]
            );
            transitionPlot.OutputDirectory = Path.Combine(transitionPlot.OutputDirectory, outputDirectory);
            transitionPlot.Show(transition);
            transitionPlot.Save(png: true);
        }
    }

    [TestMethod]
    public void TestReciprocalGaussian()
    {
        var evaluationSteps = new long[] { 50, 50 };
        var min = new double[] { 0, 0 };
        var max = new double[] { 100, 100 };
        var sigma = 25;

        var outputDirectory = Path.Combine("TestTransitions", "ReciprocalGaussian");

        var stateSpace = new Core.StateSpace.DiscreteUniform(
            dimensions: 2,
            min: min,
            max: max,
            steps: evaluationSteps
        );

        var reciprocalGaussian = new Core.Transitions.ReciprocalGaussian(
            stateSpace: stateSpace,
            sigma: sigma
        );

        int seed = 0;
        var generator = manual_seed(seed);
        var randomSelection = randint(0, stateSpace.Points.shape[0], 40, generator: generator);

        for (int i = 0; i < randomSelection.size(0); i++)
        {
            var index = randomSelection[i].item<long>();
            var transition = reciprocalGaussian.Transitions[index]
                .reshape(stateSpace.Shape);
            var transitionPlot = new Heatmap(
                title: $"Reciprocal Gaussian Transition {index}",
                xMin: min[0],
                xMax: max[0],
                yMin: min[1],
                yMax: max[1]
            );
            transitionPlot.OutputDirectory = Path.Combine(transitionPlot.OutputDirectory, outputDirectory);
            transitionPlot.Show(transition);
            transitionPlot.Save(png: true);
        }
    }
}
