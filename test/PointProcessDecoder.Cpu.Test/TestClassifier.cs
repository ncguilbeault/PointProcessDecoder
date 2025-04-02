using PointProcessDecoder.Test.Common;
using PointProcessDecoder.Core;
using static TorchSharp.torch;

using OxyPlot;
using OxyPlot.Annotations;
using OxyPlot.Series;
using OxyPlot.Axes;
using PointProcessDecoder.Core.Decoder;

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
        var stayProbability = 0.33;
        var dimensions = 1;
        var nUnits = 40;
        var sigma = 25;
        var device = CPU;
        var scalarType = ScalarType.Float32;
        var estimationMethod = Core.Estimation.EstimationMethod.KernelDensity;

        var outputDirectory = Path.Combine("TestClassifier", "SimulatedData1D", estimationMethod.ToString());
        
        var replayClassifierModel = new PointProcessModel(
            estimationMethod,
            Core.Transitions.TransitionsType.RandomWalk,
            Core.Encoder.EncoderType.SortedSpikes,
            Core.Decoder.DecoderType.HybridStateSpaceClassifier,
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
        var gen = manual_seed(seed);

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
        
        var insertionIndex = randint(0, 500, [1], device: device, dtype: ScalarType.Int32, generator: gen).item<int>();

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

        var numStates = (int)statePrediction.size(1);
        var colors = Plot.Utilities.GenerateRandomColors(numStates, seed);

        for (int i = 0; i < numStates; i++)
        {
            var state = concat([time, statePrediction[TensorIndex.Colon, i].unsqueeze(1)], dim: 1);

            plotStatePrediction.Show(
                state, 
                color: colors[i],
                addLine: true,
                seriesLabel: $"State {i}"
            );
        }

        var insertion = tensor(new double[] {
            insertionIndex,
            0,
            insertionIndex,
            1
        }).reshape(2, 2);

        plotStatePrediction.Show(
            insertion, 
            color: OxyColors.Black,
            addLine: true,
            seriesLabel: "Insertion"
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

    [TestMethod]
    public void TestClassifierModelSimulatedData2D()
    {
        var bandwidth = new double[] { 5, 5 };
        var evaluationSteps = new long[] { 50, 50 };
        var min = new double[] { 0, 0 };
        var max = new double[] { 100, 100 };
        var stayProbability = 0.99;
        var scale = 0.1;
        var dimensions = 2;
        var nUnits = 40;
        var sigma = 25;
        var device = CPU;
        var scalarType = ScalarType.Float32;
        var estimationMethod = Core.Estimation.EstimationMethod.KernelDensity;

        var outputDirectory = Path.Combine("TestClassifier", "SimulatedData2D", estimationMethod.ToString());
        
        var replayClassifierModel = new PointProcessModel(
            estimationMethod,
            Core.Transitions.TransitionsType.RandomWalk,
            Core.Encoder.EncoderType.SortedSpikes,
            Core.Decoder.DecoderType.HybridStateSpaceClassifier,
            Core.StateSpace.StateSpaceType.DiscreteUniform,
            Core.Likelihood.LikelihoodType.Poisson,
            min,
            max,
            evaluationSteps,
            bandwidth,
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
            min[0], 
            max[0],
            min[1],
            max[1],
            scale,
            scalarType,
            device
        );

        var placeFieldCenters = Simulation.Simulate.PlaceFieldCenters(
            min[0], 
            max[0],
            min[1],
            max[1], 
            nUnits,
            seed,
            scalarType,
            device
        );

        var spikingData = Simulation.Simulate.SpikesAtPosition(
            position, 
            placeFieldCenters, 
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
            min[0], 
            max[0],
            min[1],
            max[1], 
            seed,
            scalarType,
            device
        );

        var fragmentedSpikes = Simulation.Simulate.SpikesAtPosition(
            fragmentedPosition, 
            placeFieldCenters, 
            placeFieldRadius, 
            firingThreshold, 
            seed,
            device: device
        );

        var gen = manual_seed(seed);
        var insertionIndex = randint(0, 500, [1], device: device, dtype: ScalarType.Int32, generator: gen).item<int>();

        var testingPosition = vstack([
            fragmentedPosition[TensorIndex.Slice(0, insertionIndex)],
            position[TensorIndex.Slice(nTraining)], 
            fragmentedPosition[TensorIndex.Slice(insertionIndex)]
        ]);

        var testingSpikes = vstack([
            fragmentedSpikes[TensorIndex.Slice(0, insertionIndex)],
            spikingData[TensorIndex.Slice(nTraining)], 
            fragmentedSpikes[TensorIndex.Slice(insertionIndex)]
        ]);

        replayClassifierModel.Encode(trainingPosition, trainingSpikes);
        var result = replayClassifierModel.Decode(testingSpikes);

        var classifierData = new ClassifierData(replayClassifierModel.StateSpace, result);

        var posteriorPrediction = classifierData.DecoderData.Posterior.sum(dim: 0);
        var statePrediction = classifierData.StateProbabilities;
        
        Plot.Heatmap plotPosteriorPrediction = new(
            min[0],
            max[0],
            min[1],
            max[1],
            title: "ReplayClassifierPosteriorPrediction"
        );

        plotPosteriorPrediction.OutputDirectory = Path.Combine(plotPosteriorPrediction.OutputDirectory, outputDirectory);
        plotPosteriorPrediction.Show(
            posteriorPrediction, 
            testingPosition
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

        var time = arange(testingPosition.size(0)).unsqueeze(1);
        var numStates = (int)statePrediction.size(1);
        var colors = Plot.Utilities.GenerateRandomColors(numStates, seed);

        for (int i = 0; i < numStates; i++)
        {
            var state = concat([time, statePrediction[TensorIndex.Colon, i].unsqueeze(1)], dim: 1);

            plotStatePrediction.Show(
                state, 
                color: colors[i],
                addLine: true,
                seriesLabel: $"State {i}"
            );
        }

        var insertion = tensor(new double[] {
            insertionIndex,
            0,
            insertionIndex,
            1
        }).reshape(2, 2);

        plotStatePrediction.Show(
            insertion, 
            color: OxyColors.Black,
            addLine: true,
            seriesLabel: "Insertion"
        );

        plotStatePrediction.Save(png: true);
    }

    [TestMethod]
    public void CompareEffectsOfSigmaOnClassifier()
    {
        var bandwidth = new double[] { 5, 5 };
        var evaluationSteps = new long[] { 50, 50 };
        var min = new double[] { 0, 0 };
        var max = new double[] { 100, 100 };
        var stayProbability = 0.99;
        var scale = 0.1;
        var dimensions = 2;
        var nUnits = 40;
        var device = CPU;
        var scalarType = ScalarType.Float32;
        var estimationMethod = Core.Estimation.EstimationMethod.KernelDensity;

        var outputDirectory = Path.Combine("TestClassifier", "EffectsOfSigma", estimationMethod.ToString());

        var steps = 200;
        var cycles = 10;
        var placeFieldRadius = 8.0;
        var firingThreshold = 0.2;
        int seed = 0;

        var position = Simulation.Simulate.SinPosition(
            steps, 
            cycles, 
            min[0], 
            max[0],
            min[1],
            max[1],
            scale,
            scalarType,
            device
        );

        var placeFieldCenters = Simulation.Simulate.PlaceFieldCenters(
            min[0], 
            max[0],
            min[1],
            max[1], 
            nUnits,
            seed,
            scalarType,
            device
        );

        var spikingData = Simulation.Simulate.SpikesAtPosition(
            position, 
            placeFieldCenters, 
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
            min[0], 
            max[0],
            min[1],
            max[1], 
            seed,
            scalarType,
            device
        );

        var fragmentedSpikes = Simulation.Simulate.SpikesAtPosition(
            fragmentedPosition, 
            placeFieldCenters, 
            placeFieldRadius, 
            firingThreshold, 
            seed,
            device: device
        );

        var gen = manual_seed(seed);
        var insertionIndex = randint(0, 500, [1], device: device, dtype: ScalarType.Int32, generator: gen).item<int>();

        var testingPosition = vstack([
            fragmentedPosition[TensorIndex.Slice(0, insertionIndex)],
            position[TensorIndex.Slice(nTraining)], 
            fragmentedPosition[TensorIndex.Slice(insertionIndex)]
        ]);

        var testingSpikes = vstack([
            fragmentedSpikes[TensorIndex.Slice(0, insertionIndex)],
            spikingData[TensorIndex.Slice(nTraining)], 
            fragmentedSpikes[TensorIndex.Slice(insertionIndex)]
        ]);

        PointProcessModel classifierModel;
        Tensor result;
        ClassifierData classifierData;

        classifierModel = new PointProcessModel(
            estimationMethod,
            Core.Transitions.TransitionsType.RandomWalk,
            Core.Encoder.EncoderType.SortedSpikes,
            Core.Decoder.DecoderType.HybridStateSpaceClassifier,
            Core.StateSpace.StateSpaceType.DiscreteUniform,
            Core.Likelihood.LikelihoodType.Poisson,
            min,
            max,
            evaluationSteps,
            bandwidth,
            dimensions,
            sigmaRandomWalk: 0.1,
            nUnits: nUnits,
            stayProbability: stayProbability,
            device: device,
            scalarType: scalarType
        );

        classifierModel.Encode(trainingPosition, trainingSpikes);
        result = classifierModel.Decode(testingSpikes);
        classifierData = new ClassifierData(classifierModel.StateSpace, result);

        PlotClassifierData(
            min,
            max,
            Path.Combine(outputDirectory, "Sigma0.1"),
            seed,
            classifierData,
            testingPosition,
            testingSpikes,
            insertionIndex
        );

        classifierModel = new PointProcessModel(
            estimationMethod,
            Core.Transitions.TransitionsType.RandomWalk,
            Core.Encoder.EncoderType.SortedSpikes,
            Core.Decoder.DecoderType.HybridStateSpaceClassifier,
            Core.StateSpace.StateSpaceType.DiscreteUniform,
            Core.Likelihood.LikelihoodType.Poisson,
            min,
            max,
            evaluationSteps,
            bandwidth,
            dimensions,
            sigmaRandomWalk: 1.0,
            nUnits: nUnits,
            stayProbability: stayProbability,
            device: device,
            scalarType: scalarType
        );

        classifierModel.Encode(trainingPosition, trainingSpikes);
        result = classifierModel.Decode(testingSpikes);
        classifierData = new ClassifierData(classifierModel.StateSpace, result);

        PlotClassifierData(
            min,
            max,
            Path.Combine(outputDirectory, "Sigma1.0"),
            seed,
            classifierData,
            testingPosition,
            testingSpikes,
            insertionIndex
        );

        classifierModel = new PointProcessModel(
            estimationMethod,
            Core.Transitions.TransitionsType.RandomWalk,
            Core.Encoder.EncoderType.SortedSpikes,
            Core.Decoder.DecoderType.HybridStateSpaceClassifier,
            Core.StateSpace.StateSpaceType.DiscreteUniform,
            Core.Likelihood.LikelihoodType.Poisson,
            min,
            max,
            evaluationSteps,
            bandwidth,
            dimensions,
            sigmaRandomWalk: 10.0,
            nUnits: nUnits,
            stayProbability: stayProbability,
            device: device,
            scalarType: scalarType
        );

        classifierModel.Encode(trainingPosition, trainingSpikes);
        result = classifierModel.Decode(testingSpikes);
        classifierData = new ClassifierData(classifierModel.StateSpace, result);

        PlotClassifierData(
            min,
            max,
            Path.Combine(outputDirectory, "Sigma10.0"),
            seed,
            classifierData,
            testingPosition,
            testingSpikes,
            insertionIndex
        );
    }

    [TestMethod]
    public void CompareEffectsOfStayProbability()
    {
        var bandwidth = new double[] { 5, 5 };
        var evaluationSteps = new long[] { 50, 50 };
        var min = new double[] { 0, 0 };
        var max = new double[] { 100, 100 };
        var sigma = 25;
        var scale = 0.1;
        var dimensions = 2;
        var nUnits = 40;
        var device = CPU;
        var scalarType = ScalarType.Float32;
        var estimationMethod = Core.Estimation.EstimationMethod.KernelDensity;

        var outputDirectory = Path.Combine("TestClassifier", "EffectsOfStayProbability", estimationMethod.ToString());

        var steps = 200;
        var cycles = 10;
        var placeFieldRadius = 8.0;
        var firingThreshold = 0.2;
        int seed = 0;

        var position = Simulation.Simulate.SinPosition(
            steps, 
            cycles, 
            min[0], 
            max[0],
            min[1],
            max[1],
            scale,
            scalarType,
            device
        );

        var placeFieldCenters = Simulation.Simulate.PlaceFieldCenters(
            min[0], 
            max[0],
            min[1],
            max[1], 
            nUnits,
            seed,
            scalarType,
            device
        );

        var spikingData = Simulation.Simulate.SpikesAtPosition(
            position, 
            placeFieldCenters, 
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
            min[0], 
            max[0],
            min[1],
            max[1], 
            seed,
            scalarType,
            device
        );

        var fragmentedSpikes = Simulation.Simulate.SpikesAtPosition(
            fragmentedPosition, 
            placeFieldCenters, 
            placeFieldRadius, 
            firingThreshold, 
            seed,
            device: device
        );

        var gen = manual_seed(seed);
        var insertionIndex = randint(0, 500, [1], device: device, dtype: ScalarType.Int32, generator: gen).item<int>();

        var testingPosition = vstack([
            fragmentedPosition[TensorIndex.Slice(0, insertionIndex)],
            position[TensorIndex.Slice(nTraining)], 
            fragmentedPosition[TensorIndex.Slice(insertionIndex)]
        ]);

        var testingSpikes = vstack([
            fragmentedSpikes[TensorIndex.Slice(0, insertionIndex)],
            spikingData[TensorIndex.Slice(nTraining)], 
            fragmentedSpikes[TensorIndex.Slice(insertionIndex)]
        ]);

        PointProcessModel classifierModel;
        Tensor result;
        ClassifierData classifierData;

        classifierModel = new PointProcessModel(
            estimationMethod,
            Core.Transitions.TransitionsType.RandomWalk,
            Core.Encoder.EncoderType.SortedSpikes,
            Core.Decoder.DecoderType.HybridStateSpaceClassifier,
            Core.StateSpace.StateSpaceType.DiscreteUniform,
            Core.Likelihood.LikelihoodType.Poisson,
            min,
            max,
            evaluationSteps,
            bandwidth,
            dimensions,
            sigmaRandomWalk: sigma,
            nUnits: nUnits,
            stayProbability: 0.5,
            device: device,
            scalarType: scalarType
        );

        classifierModel.Encode(trainingPosition, trainingSpikes);
        result = classifierModel.Decode(testingSpikes);
        classifierData = new ClassifierData(classifierModel.StateSpace, result);

        PlotClassifierData(
            min,
            max,
            Path.Combine(outputDirectory, "StayProbability0.5"),
            seed,
            classifierData,
            testingPosition,
            testingSpikes,
            insertionIndex
        );

        classifierModel = new PointProcessModel(
            estimationMethod,
            Core.Transitions.TransitionsType.RandomWalk,
            Core.Encoder.EncoderType.SortedSpikes,
            Core.Decoder.DecoderType.HybridStateSpaceClassifier,
            Core.StateSpace.StateSpaceType.DiscreteUniform,
            Core.Likelihood.LikelihoodType.Poisson,
            min,
            max,
            evaluationSteps,
            bandwidth,
            dimensions,
            sigmaRandomWalk: sigma,
            nUnits: nUnits,
            stayProbability: 0.9,
            device: device,
            scalarType: scalarType
        );

        classifierModel.Encode(trainingPosition, trainingSpikes);
        result = classifierModel.Decode(testingSpikes);
        classifierData = new ClassifierData(classifierModel.StateSpace, result);

        PlotClassifierData(
            min,
            max,
            Path.Combine(outputDirectory, "StayProbability0.9"),
            seed,
            classifierData,
            testingPosition,
            testingSpikes,
            insertionIndex
        );

        classifierModel = new PointProcessModel(
            estimationMethod,
            Core.Transitions.TransitionsType.RandomWalk,
            Core.Encoder.EncoderType.SortedSpikes,
            Core.Decoder.DecoderType.HybridStateSpaceClassifier,
            Core.StateSpace.StateSpaceType.DiscreteUniform,
            Core.Likelihood.LikelihoodType.Poisson,
            min,
            max,
            evaluationSteps,
            bandwidth,
            dimensions,
            sigmaRandomWalk: sigma,
            nUnits: nUnits,
            stayProbability: 0.999999,
            device: device,
            scalarType: scalarType
        );

        classifierModel.Encode(trainingPosition, trainingSpikes);
        result = classifierModel.Decode(testingSpikes);
        classifierData = new ClassifierData(classifierModel.StateSpace, result);

        PlotClassifierData(
            min,
            max,
            Path.Combine(outputDirectory, "StayProbability0.999999"),
            seed,
            classifierData,
            testingPosition,
            testingSpikes,
            insertionIndex
        );

        classifierModel = new PointProcessModel(
            estimationMethod,
            Core.Transitions.TransitionsType.RandomWalk,
            Core.Encoder.EncoderType.SortedSpikes,
            Core.Decoder.DecoderType.HybridStateSpaceClassifier,
            Core.StateSpace.StateSpaceType.DiscreteUniform,
            Core.Likelihood.LikelihoodType.Poisson,
            min,
            max,
            evaluationSteps,
            bandwidth,
            dimensions,
            sigmaRandomWalk: sigma,
            nUnits: nUnits,
            stayProbability: 1.0,
            device: device,
            scalarType: scalarType
        );

        classifierModel.Encode(trainingPosition, trainingSpikes);
        result = classifierModel.Decode(testingSpikes);
        classifierData = new ClassifierData(classifierModel.StateSpace, result);

        PlotClassifierData(
            min,
            max,
            Path.Combine(outputDirectory, "StayProbability1.0"),
            seed,
            classifierData,
            testingPosition,
            testingSpikes,
            insertionIndex
        );

        classifierModel = new PointProcessModel(
            estimationMethod,
            Core.Transitions.TransitionsType.RandomWalk,
            Core.Encoder.EncoderType.SortedSpikes,
            Core.Decoder.DecoderType.HybridStateSpaceClassifier,
            Core.StateSpace.StateSpaceType.DiscreteUniform,
            Core.Likelihood.LikelihoodType.Poisson,
            min,
            max,
            evaluationSteps,
            bandwidth,
            dimensions,
            sigmaRandomWalk: sigma,
            nUnits: nUnits,
            stayProbability: 0.25,
            device: device,
            scalarType: scalarType
        );

        classifierModel.Encode(trainingPosition, trainingSpikes);
        result = classifierModel.Decode(testingSpikes);
        classifierData = new ClassifierData(classifierModel.StateSpace, result);

        PlotClassifierData(
            min,
            max,
            Path.Combine(outputDirectory, "StayProbability0.25"),
            seed,
            classifierData,
            testingPosition,
            testingSpikes,
            insertionIndex
        );
    }

    private static void PlotClassifierData(
        double[] min,
        double[] max,
        string outputDirectory,
        int seed,
        ClassifierData classifierData,
        Tensor testingPosition,
        Tensor testingSpikes,
        int insertionIndex
    )
    {

        var posteriorPrediction = classifierData.DecoderData.Posterior.sum(dim: 0);
        var statePrediction = classifierData.StateProbabilities;

        Plot.Heatmap plotPosteriorPrediction = new(
            min[0],
            max[0],
            min[1],
            max[1],
            title: "ReplayClassifierPosteriorPrediction"
        );

        plotPosteriorPrediction.OutputDirectory = Path.Combine(plotPosteriorPrediction.OutputDirectory, outputDirectory);
        plotPosteriorPrediction.Show(
            posteriorPrediction, 
            testingPosition
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

        var time = arange(testingPosition.size(0)).unsqueeze(1);
        var numStates = (int)statePrediction.size(1);
        var colors = Plot.Utilities.GenerateRandomColors(numStates, seed);

        for (int i = 0; i < numStates; i++)
        {
            var state = concat([time, statePrediction[TensorIndex.Colon, i].unsqueeze(1)], dim: 1);

            plotStatePrediction.Show(
                state, 
                color: colors[i],
                addLine: true,
                seriesLabel: $"State {i}"
            );
        }

        var insertion = tensor(new double[] {
            insertionIndex,
            0,
            insertionIndex,
            1
        }).reshape(2, 2);

        plotStatePrediction.Show(
            insertion, 
            color: OxyColors.Black,
            addLine: true,
            seriesLabel: "Insertion"
        );

        plotStatePrediction.Save(png: true);
    }

    [TestMethod]
    public void EvaluateStayProbability1()
    {
        var bandwidth = new double[] { 5, 5 };
        var evaluationSteps = new long[] { 50, 50 };
        var min = new double[] { 0, 0 };
        var max = new double[] { 100, 100 };
        var sigma = 25;
        var scale = 0.1;
        var dimensions = 2;
        var stayProbability = 0.99;
        var nUnits = 40;
        var device = CPU;
        var scalarType = ScalarType.Float32;
        var estimationMethod = Core.Estimation.EstimationMethod.KernelDensity;

        var outputDirectory = Path.Combine("TestClassifier", "EvaluateStayProbability1", estimationMethod.ToString());

        var steps = 200;
        var cycles = 10;
        var placeFieldRadius = 8.0;
        var firingThreshold = 0.2;
        int seed = 0;

        var position = Simulation.Simulate.SinPosition(
            steps, 
            cycles, 
            min[0], 
            max[0],
            min[1],
            max[1],
            scale,
            scalarType,
            device
        );

        var placeFieldCenters = Simulation.Simulate.PlaceFieldCenters(
            min[0], 
            max[0],
            min[1],
            max[1], 
            nUnits,
            seed,
            scalarType,
            device
        );

        var spikingData = Simulation.Simulate.SpikesAtPosition(
            position, 
            placeFieldCenters, 
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
            min[0], 
            max[0],
            min[1],
            max[1], 
            seed,
            scalarType,
            device
        );

        var fragmentedSpikes = Simulation.Simulate.SpikesAtPosition(
            fragmentedPosition, 
            placeFieldCenters, 
            placeFieldRadius, 
            firingThreshold, 
            seed,
            device: device
        );

        var gen = manual_seed(seed);
        var insertionIndex = randint(0, 500, [1], device: device, dtype: ScalarType.Int32, generator: gen).item<int>();

        var testingPosition = vstack([
            fragmentedPosition[TensorIndex.Slice(0, insertionIndex)],
            position[TensorIndex.Slice(nTraining)], 
            fragmentedPosition[TensorIndex.Slice(insertionIndex)]
        ]);

        var testingSpikes = vstack([
            fragmentedSpikes[TensorIndex.Slice(0, insertionIndex)],
            spikingData[TensorIndex.Slice(nTraining)], 
            fragmentedSpikes[TensorIndex.Slice(insertionIndex)]
        ]);

        PointProcessModel classifierModel;
        Tensor result;
        ClassifierData classifierData;

        classifierModel = new PointProcessModel(
            estimationMethod,
            Core.Transitions.TransitionsType.RandomWalk,
            Core.Encoder.EncoderType.SortedSpikes,
            Core.Decoder.DecoderType.HybridStateSpaceClassifier,
            Core.StateSpace.StateSpaceType.DiscreteUniform,
            Core.Likelihood.LikelihoodType.Poisson,
            min,
            max,
            evaluationSteps,
            bandwidth,
            dimensions,
            sigmaRandomWalk: sigma,
            nUnits: nUnits,
            stayProbability: stayProbability,
            device: device,
            scalarType: scalarType
        );

        classifierModel.Encode(trainingPosition, trainingSpikes);
        result = classifierModel.Decode(testingSpikes);
        classifierData = new ClassifierData(classifierModel.StateSpace, result);

        PlotClassifierData(
            min,
            max,
            outputDirectory,
            seed,
            classifierData,
            testingPosition,
            testingSpikes,
            insertionIndex
        );
    }
}
