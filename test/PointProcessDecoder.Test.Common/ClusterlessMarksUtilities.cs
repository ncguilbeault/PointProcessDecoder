using static TorchSharp.torch;
using PointProcessDecoder.Plot;
using PointProcessDecoder.Core;
using PointProcessDecoder.Core.Estimation;
using PointProcessDecoder.Core.Transitions;
using PointProcessDecoder.Core.Encoder;
using PointProcessDecoder.Core.Decoder;
using PointProcessDecoder.Core.Likelihood;
using PointProcessDecoder.Core.StateSpace;
using PointProcessDecoder.Simulation;

namespace PointProcessDecoder.Test.Common;

public static class ClusterlessMarksUtilities
{
    public static void BayesianStateSpaceClusterlessMarksSimulated(
        double[]? observationBandwidth = null,
        int dimensions = 1,
        long[]? evaluationSteps = null,
        int steps = 200,
        int cycles = 10,
        double[]? min = null,
        double[]? max = null,
        double scale = 0.1,
        int numNeurons = 40,
        double placeFieldRadius = 8.0,
        double firingThreshold = 0.2,
        int markDimensions = 4,
        int markChannels = 8,
        double[]? markBandwidth = null,
        int nTraining = 1800,
        int nTesting = 200,
        double? sigma = null,
        double? distanceThreshold = null,
        string outputDirectory = "TestClusterlessMarks",
        string modelDirectory = "BayesianStateSpaceClusterlessMarksSimulatedData",
        ScalarType scalarType = ScalarType.Float32,
        Device? device = null,
        int seed = 0,
        double spikeScale = 5.0,
        double noiseScale = 0.5,
        string figureName = "Prediction",
        EstimationMethod estimationMethod = EstimationMethod.KernelDensity,
        TransitionsType transitionsType = TransitionsType.Uniform
    )
    {
        observationBandwidth ??= [5];
        evaluationSteps ??= [50];
        min ??= [0];
        max ??= [100];
        markBandwidth ??= [1, 1, 1, 1];
        device ??= CPU;

        outputDirectory = string.IsNullOrEmpty(modelDirectory) ? outputDirectory : Path.Combine(outputDirectory, modelDirectory);
        outputDirectory = Path.Combine(outputDirectory, estimationMethod.ToString(), transitionsType.ToString());
        
        var pointProcessModel = new PointProcessModel(
            estimationMethod,
            transitionsType,
            EncoderType.ClusterlessMarkEncoder,
            DecoderType.StateSpaceDecoder,
            StateSpaceType.DiscreteUniformStateSpace,
            LikelihoodType.Clusterless,
            min,
            max,
            evaluationSteps,
            observationBandwidth,
            dimensions,
            markDimensions: markDimensions,
            markChannels: markChannels,
            markBandwidth: markBandwidth,
            distanceThreshold: distanceThreshold,
            sigmaRandomWalk: sigma,
            device: device,
            scalarType: scalarType
        );

        Tensor position = empty(0);
        Tensor spikingData = empty(0);
        Tensor positionPoints = empty(0);
        double[] heatmapRange = new double[4];

        if (dimensions == 1)
        {
            (position, spikingData) = Simulation.Utilities.InitializeSimulation1D(
                steps: steps,
                cycles: cycles,
                min: min[0],
                max: max[0],
                numNeurons: numNeurons,
                placeFieldRadius: placeFieldRadius,
                firingThreshold: firingThreshold,
                scalarType: scalarType,
                seed: seed
            );

            positionPoints = concat([arange(steps * cycles - nTraining).unsqueeze(-1), position[TensorIndex.Slice(nTraining, nTraining + nTesting)]], dim: 1);
            heatmapRange = [0, steps * cycles - nTraining, min[0], max[0]];
        }
        else if (dimensions == 2)
        {
            (position, spikingData) = Simulation.Utilities.InitializeSimulation2D(
                steps: steps,
                cycles: cycles,
                xMin: min[0],
                xMax: max[0],
                yMin: min[1],
                yMax: max[1],
                numNeurons: numNeurons,
                placeFieldRadius: placeFieldRadius,
                firingThreshold: firingThreshold,
                scale: scale,
                scalarType: scalarType,
                seed: seed
            );

            positionPoints = position[TensorIndex.Slice(nTraining, nTraining + nTesting)];
            heatmapRange = [min[0], max[0], min[1], max[1]];
        }
        
        var marks = Simulate.MarksAtPosition(
            position,
            spikingData, 
            markDimensions, 
            markChannels, 
            scalarType: scalarType, 
            device: device,
            spikeScale: spikeScale,
            noiseScale: noiseScale
        );

        pointProcessModel.Encode(position[TensorIndex.Slice(0, nTraining)], marks[TensorIndex.Slice(0, nTraining)]);
        var prediction = pointProcessModel.Decode(marks[TensorIndex.Slice(nTraining, nTraining + nTesting)]);
        if (dimensions == 2)
            prediction = prediction.max(dim: 0).values;

        Heatmap plotPrediction = new(
            heatmapRange[0],
            heatmapRange[1],
            heatmapRange[2],
            heatmapRange[3],
            title: figureName
        );

        plotPrediction.OutputDirectory = Path.Combine(plotPrediction.OutputDirectory, outputDirectory);
        plotPrediction.Show(
            prediction, 
            positionPoints
        );
        plotPrediction.Save(png: true);
    }

    public static void BayesianStateSpaceClusterlessMarksRealData(
        double[]? observationBandwidth = null,
        int dimensions = 2,
        long[]? evaluationSteps = null,
        double[]? minVals = null,
        double[]? maxVals = null,
        double trainingFraction = 0.8,
        double? testFraction = null,
        int markDimensions = 4,
        int markChannels = 28,
        double[]? markBandwidth = null,
        double? sigma = null,
        double? distanceThreshold = null,
        string outputDirectory = "TestClusterlessMarks",
        string modelDirectory = "BayesianStateSpaceClusterlessMarksRealData",
        ScalarType scalarType = ScalarType.Float32,
        Device? device = null,
        string figureName = "Prediction",
        EstimationMethod estimationMethod = EstimationMethod.KernelDensity,
        TransitionsType transitionsType = TransitionsType.Uniform,
        string positionFile = "../../../../data/position.bin",
        string marksFile = "../../../../data/marks.bin"
    )
    {
        observationBandwidth ??= [5, 5];
        evaluationSteps ??= [50, 50];
        minVals ??= [0, 0];
        maxVals ??= [120, 120];
        markBandwidth ??= [1, 1, 1, 1];
        device ??= CPU;

        outputDirectory = string.IsNullOrEmpty(modelDirectory) ? outputDirectory : Path.Combine(outputDirectory, modelDirectory);
        outputDirectory = Path.Combine(outputDirectory, estimationMethod.ToString(), transitionsType.ToString());
        
        var (position, marks) = Utilities.InitializeRealClusterlessMarksData(
            positionFile: positionFile,
            marksFile: marksFile,
            device: device,
            scalarType: scalarType
        );

        position = position.reshape(-1, 2);
        marks = marks.reshape(position.shape[0], markDimensions, markChannels);

        position = position[TensorIndex.Slice(0, 20000)];
        marks = marks[TensorIndex.Slice(0, 20000)];

        double[] heatmapRange = [minVals[0], maxVals[0], minVals[1], maxVals[1]];

        var nTraining = (int)(position.shape[0] * trainingFraction);
        var nTesting = testFraction == null ? (int)position.shape[0] - nTraining : Math.Min((int)(position.shape[0] * testFraction), (int)position.shape[0] - nTraining);

        var pointProcessModel = new PointProcessModel(
            estimationMethod,
            transitionsType,
            EncoderType.ClusterlessMarkEncoder,
            DecoderType.StateSpaceDecoder,
            StateSpaceType.DiscreteUniformStateSpace,
            LikelihoodType.Clusterless,
            minVals,
            maxVals,
            evaluationSteps,
            observationBandwidth,
            dimensions,
            markDimensions: markDimensions,
            markChannels: markChannels,
            markBandwidth: markBandwidth,
            distanceThreshold: distanceThreshold,
            sigmaRandomWalk: sigma,
            device: device,
            scalarType: scalarType
        );

        pointProcessModel.Encode(position[TensorIndex.Slice(0, nTraining)], marks[TensorIndex.Slice(0, nTraining)]);
        var prediction = pointProcessModel.Decode(marks[TensorIndex.Slice(nTraining, nTraining + nTesting)]);
        prediction = prediction.sum(dim: 0);

        Heatmap plotPrediction = new(
            heatmapRange[0],
            heatmapRange[1],
            heatmapRange[2],
            heatmapRange[3],
            title: figureName
        );

        plotPrediction.OutputDirectory = Path.Combine(plotPrediction.OutputDirectory, outputDirectory);
        plotPrediction.Show(
            prediction, 
            position[TensorIndex.Slice(nTraining, nTraining + nTesting)]
        );
        plotPrediction.Save(png: true);
    }
}