using static TorchSharp.torch;
using OxyPlot;
using OxyPlot.Series;
using OxyPlot.Axes;

namespace PointProcessDecoder.Plot;

public class ScatterPlot : OxyPlotBase
{
    public override PlotModel Plot => plot;
    private PlotModel plot;
    public double XMin { get; } = 0;
    public double XMax { get; } = 100;
    public double YMin { get; } = 0;
    public double YMax { get; } = 100;
    public string Title { get; } = "ScatterPlot";

    public ScatterPlot()
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

    public ScatterPlot(double? xMin = null, double? xMax = null, double? yMin = null, double? yMax = null, string? title = null, string? figureName = null)
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

        plot.Axes.Add(xAxis);
        plot.Axes.Add(yAxis);
    }

    public void Show(Tensor positionData, OxyColor? color = null)
    {
        
        var scatterSeries = new ScatterSeries
        {
            MarkerType = MarkerType.Circle,
            MarkerFill = color ?? OxyColors.Red,
            MarkerSize = 4,
            MarkerStrokeThickness = 1,
        };
        
        for (int i = 0; i < positionData.shape[0]; i++)
        {
            scatterSeries.Points.Add(new ScatterPoint(positionData[i,0].item<double>(), positionData[i,1].item<double>()));
        }

        plot.Series.Add(scatterSeries);
    }

    public void Show<T>(Tensor positionData, OxyColor? color = null) where T : unmanaged
    {
        
        var scatterSeries = new ScatterSeries
        {
            MarkerType = MarkerType.Circle,
            MarkerFill = color ?? OxyColors.Red,
            MarkerSize = 4,
            MarkerStrokeThickness = 1,
        };
        
        for (int i = 0; i < positionData.shape[0]; i++)
        {
            scatterSeries.Points.Add(new ScatterPoint(
                Convert.ToDouble(positionData[i,0].item<T>()), 
                Convert.ToDouble(positionData[i,1].item<T>())
            ));
        }

        plot.Series.Add(scatterSeries);
    }
}