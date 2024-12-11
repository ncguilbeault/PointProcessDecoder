using static TorchSharp.torch;
using OxyPlot;
using OxyPlot.Series;
using OxyPlot.Axes;

namespace PointProcessDecoder.Plot;

public class Heatmap : OxyPlotBase
{
    public override PlotModel Plot => plot;
    private PlotModel plot;
    public double XMin { get; } = double.NaN;
    public double XMax { get; } = double.NaN;
    public double YMin { get; } = double.NaN;
    public double YMax { get; } = double.NaN;
    public string Title { get; } = "Heatmap";

    public Heatmap()
    {
        FigureName = Title;

        plot = new PlotModel 
        { 
            Title = Title,
            TitleFont = "DejaVu Sans",
            DefaultFont = "DejaVu Sans",
            Background = OxyColors.White
        };

        Initialize();
    }

    public Heatmap(double? xMin = null, double? xMax = null, double? yMin = null, double? yMax = null, string? title = null, string? figureName = null)
    {
        XMin = xMin ?? XMin;
        XMax = xMax ?? XMax;
        YMin = yMin ?? YMin;
        YMax = yMax ?? YMax;
        Title = title ?? Title;
        FigureName = figureName ?? Title;

        plot = new PlotModel 
        { 
            Title = Title,
            TitleFont = "DejaVu Sans",
            DefaultFont = "DejaVu Sans",
            Background = OxyColors.White
        };

        Initialize();
    }

    public override void Initialize()
    {
        var xAxis = new LinearAxis 
        { 
            Position = AxisPosition.Bottom, 
            Title = "X Axis",
            TitleFont = "DejaVu Sans",
            Minimum = XMin,
            Maximum = XMax
        };

        var yAxis = new LinearAxis 
        { 
            Position = AxisPosition.Left, 
            Title = "Y Axis",
            TitleFont = "DejaVu Sans",
            Minimum = YMin,
            Maximum = YMax
        };

        var colorAxis = new LinearColorAxis 
        { 
            Position = AxisPosition.Right, 
            Palette = OxyPalettes.Viridis(100), 
            TitleFont = "DejaVu Sans",
            Key = "color"
        };

        var scatterColorAxis = new RangeColorAxis 
        { 
            Position = AxisPosition.None,
            Key = "scatterColor"
        };

        plot.Axes.Add(colorAxis);
        plot.Axes.Add(scatterColorAxis);
        plot.Axes.Add(xAxis);
        plot.Axes.Add(yAxis);
    }

    public void Show(Tensor density, Tensor? points = null)
    {
        var densityArray = density.data<double>().ToNDArray();
        var densityData = new double[density.shape[0], density.shape[1]];
        Array.Copy(densityArray, densityData, densityArray.Length);

        var heatMapSeries = new HeatMapSeries
        {
            X0 = XMin,
            X1 = XMax,
            Y0 = YMin,
            Y1 = YMax,
            Interpolate = true,
            Data = densityData,
            ColorAxisKey = "color"
        };

        plot.Series.Add(heatMapSeries);

        if (points is not null)
        {
            AddPoints(points);
        }
    }

    public void Show<T>(Tensor density, Tensor? points = null) where T : unmanaged
    {
        var densityArray = density.data<T>().ToNDArray();
        var densityData = new double[density.shape[0], density.shape[1]];
        Array.Copy(densityArray, densityData, densityArray.Length);

        var heatMapSeries = new HeatMapSeries
        {
            X0 = XMin,
            X1 = XMax,
            Y0 = YMin,
            Y1 = YMax,
            Interpolate = true,
            Data = densityData,
            ColorAxisKey = "color"
        };

        plot.Series.Add(heatMapSeries);

        if (points is not null)
        {
            AddPoints(points);
        }
    }

    private void AddPoints(Tensor points)
    {
        var scatterSeries = new ScatterSeries
        {
            MarkerType = MarkerType.Circle,
            MarkerFill = OxyColors.Red,
            MarkerSize = 4,
            MarkerStrokeThickness = 1,
            ColorAxisKey = "scatterColor"
        };

        var pointsArray = points.data<double>().ToNDArray();
        var pointsData = new double[points.shape[0], points.shape[1]];
        Array.Copy(pointsArray, pointsData, pointsArray.Length);

        for (int i = 0; i < points.shape[0]; i++)
        {
            scatterSeries.Points.Add(new ScatterPoint(pointsData[i,0], pointsData[i,1], double.NaN, 0));
        }

        plot.Series.Add(scatterSeries);
    }
}