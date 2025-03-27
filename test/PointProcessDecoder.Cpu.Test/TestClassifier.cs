using PointProcessDecoder.Test.Common;
using PointProcessDecoder.Core;
using static TorchSharp.torch;

using OxyPlot;
using OxyPlot.Annotations;
using OxyPlot.Series;
using OxyPlot.Axes;

namespace PointProcessDecoder.Cpu.Test;

[TestClass]
public class TestClassifier
{

    [TestMethod]
    public void TestClassifierModelSimulatedData()
    {
        var bandwidth = 5;
        var evaluationSteps = 50;
        var min = 0;
        var max = 100;
        var stayProbability = 0.99;
        var scale = 1;
        var dimensions = 1;
        var nUnits = 40;
        var sigma = 1;
        var device = CPU;
        var scalarType = ScalarType.Float32;
        var estimationMethod = Core.Estimation.EstimationMethod.KernelDensity;

        var outputDirectory = Path.Combine("TestReplayModel", "SimulatedData1D", estimationMethod.ToString());
        
        var replayClassifierModel = new PointProcessModel(
            estimationMethod,
            Core.Transitions.TransitionsType.RandomWalk,
            Core.Encoder.EncoderType.SortedSpikes,
            Core.Decoder.DecoderType.HybridStateSpaceReplayClassifier,
            Core.StateSpace.StateSpaceType.DiscreteUniform,
            Core.Likelihood.LikelihoodType.Poisson,
            [min],
            [max],
            [evaluationSteps],
            [bandwidth],
            dimensions,
            sigmaRandomWalk: sigma,
            nUnits: nUnits,
            stayProbability: stayProbability,
            device: device,
            scalarType: scalarType
        );

        var steps = 200;
        var cycles = 10;
        var placeFieldRadius = 8.0;
        var firingThreshold = 0.2;
        int seed = 0;

        var position = Simulation.Simulate.SinPosition(
            steps, 
            cycles, 
            min, 
            max, 
            scalarType,
            device
        );
        
        var position1DExpanded = concat([zeros_like(position), position], dim: 1);

        var placeFieldCenters = Simulation.Simulate.PlaceFieldCenters(
            min, 
            max, 
            nUnits,
            scalarType,
            device
        );

        var placeFieldCenters2D = concat([zeros_like(placeFieldCenters), placeFieldCenters], dim: 1);

        var spikingData = Simulation.Simulate.SpikesAtPosition(
            position1DExpanded, 
            placeFieldCenters2D, 
            placeFieldRadius, 
            firingThreshold, 
            seed,
            device: device
        );

        var nTraining = 1800;
        var trainingPosition = position[TensorIndex.Slice(0, nTraining)];
        var trainingSpikes = spikingData[TensorIndex.Slice(0, nTraining)];

        var fragmentedPosition = Simulation.Simulate.RandPosition(
            500, 
            min, 
            max, 
            seed,
            scalarType,
            device
        );

        var fragmentedPositionExpanded = concat([zeros_like(fragmentedPosition), fragmentedPosition], dim: 1);

        var fragmentedSpikes = Simulation.Simulate.SpikesAtPosition(
            fragmentedPositionExpanded, 
            placeFieldCenters2D, 
            placeFieldRadius, 
            firingThreshold, 
            seed,
            device: device
        );

        var insertionIndex = 100;

        var testingPosition = vstack([
            fragmentedPosition[TensorIndex.Slice(0, insertionIndex)],
            position[TensorIndex.Slice(nTraining)], 
            fragmentedPosition[TensorIndex.Slice(insertionIndex)]
        ]);

        var time = arange(testingPosition.size(0)).unsqueeze(1);
        var positionPoints = concat([time, testingPosition], dim: 1);

        var testingSpikes = vstack([
            fragmentedSpikes[TensorIndex.Slice(0, insertionIndex)],
            spikingData[TensorIndex.Slice(nTraining)], 
            fragmentedSpikes[TensorIndex.Slice(insertionIndex)]
        ]);

        replayClassifierModel.Encode(trainingPosition, trainingSpikes);

        var result = replayClassifierModel.Decode(testingSpikes);

        var posteriorPrediction = result.sum(dim: 1);
        var statePrediction = result.sum(dim: -1);
        
        Plot.Heatmap plotPosteriorPrediction = new(
            0,
            testingSpikes.size(0),
            min,
            max,
            title: "ReplayClassifierPosteriorPrediction"
        );

        plotPosteriorPrediction.OutputDirectory = Path.Combine(plotPosteriorPrediction.OutputDirectory, outputDirectory);
        plotPosteriorPrediction.Show(
            posteriorPrediction, 
            positionPoints
        );

        plotPosteriorPrediction.Save(png: true);

        Plot.ScatterPlot plotStatePrediction = new(
            0,
            testingSpikes.size(0),
            0,
            1,
            title: "ReplayClassifierStatePrediction"
        );


        plotStatePrediction.OutputDirectory = Path.Combine(plotStatePrediction.OutputDirectory, outputDirectory);

        var colors = Plot.Utilities.GenerateRandomColors(2, seed);

        var state0 = concat([time, statePrediction[TensorIndex.Colon, 0].unsqueeze(1)], dim: 1);

        plotStatePrediction.Show(
            state0, 
            color: colors[0],
            addLine: true,
            seriesLabel: "State 0"
        );

        var state1 = concat([time, statePrediction[TensorIndex.Colon, 1].unsqueeze(1)], dim: 1);

        plotStatePrediction.Show(
            state1, 
            color: colors[1],
            addLine: true,
            seriesLabel: "State 1"
        );

        plotStatePrediction.Save(png: true);

        var decoderModel = new PointProcessModel(
            estimationMethod,
            Core.Transitions.TransitionsType.RandomWalk,
            Core.Encoder.EncoderType.SortedSpikes,
            Core.Decoder.DecoderType.StateSpaceDecoder,
            Core.StateSpace.StateSpaceType.DiscreteUniform,
            Core.Likelihood.LikelihoodType.Poisson,
            [min],
            [max],
            [evaluationSteps],
            [bandwidth],
            dimensions,
            sigmaRandomWalk: sigma,
            nUnits: nUnits,
            device: device,
            scalarType: scalarType
        );

        decoderModel.Encode(trainingPosition, trainingSpikes);
        var prediction = decoderModel.Decode(testingSpikes);

        Plot.Heatmap plotDecoder = new(
            0,
            testingSpikes.size(0),
            min,
            max,
            title: "DecoderPrediction"
        );

        plotDecoder.OutputDirectory = Path.Combine(plotDecoder.OutputDirectory, outputDirectory);
        plotDecoder.Show(
            prediction, 
            positionPoints
        );

        plotDecoder.Save(png: true);
    }
}
