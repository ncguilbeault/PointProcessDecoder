using static TorchSharp.torch;
using OxyPlot;
using OxyPlot.Series;
using OxyPlot.Axes;

namespace ClusterlessDecoder.Plot
{
    public class PlotSpikingNeurons : PlotBase
    {
        public override PlotModel Plot => plot;
        private PlotModel plot;
        public double XMin { get; } = 0;
        public double XMax { get; } = 100;
        public double YMin { get; } = 0;
        public double YMax { get; } = 100;
        public string Title { get; } = "SpikingNeurons";

        public PlotSpikingNeurons()
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

        public PlotSpikingNeurons(double? xMin = null, double? xMax = null, double? yMin = null, double? yMax = null, string? title = null, string? figureName = null)
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

        public void Show(Tensor positionData, Tensor spikingData)
        {
            var numNeurons = spikingData.shape[0];

            for (int i = 0; i < numNeurons; i++)
            {
                var rgb = randint(0, 255, 3).to_type(ScalarType.Byte).data<byte>().ToArray();
                var alpha = Convert.ToByte(255*0.5);

                var scatterSeries = new ScatterSeries
                {
                    MarkerType = MarkerType.Circle,
                    MarkerSize = 4,
                    MarkerStrokeThickness = 1,
                    MarkerFill = OxyColor.FromArgb(alpha, rgb[0], rgb[1], rgb[2])
                };

                for (int j = 0; j < positionData.shape[0]; j++)
                {
                    if (spikingData[i, j].item<bool>())
                    {
                        scatterSeries.Points.Add(new ScatterPoint(positionData[j, 0].item<double>(), positionData[j, 1].item<double>()));
                    }
                }
                
                plot.Series.Add(scatterSeries);
            }
        }
    }
}