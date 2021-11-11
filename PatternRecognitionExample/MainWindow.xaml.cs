using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Lenium.NeuralNetwork;
using Lenium.NeuralNetwork.Education.Backpropagation;
using Lenium.NeuralNetwork.Enums;
using Lenium.NeuralNetwork.InputDatas;
using Lenium.NeuralNetwork.InputDatas.Interfaces;
using Lenium.NeuralNetwork.InputDatas.Normalizators;
using Lenium.NeuralNetwork.Logging;
using Lenium.NeuralNetwork.Logging.Args;
using Lenium.NeuralNetwork.SavingAndLoading.ByteArraySavingAndLoading;
using Lenium.NeuralNetwork.TransferFunctions;
using Microsoft.Win32;
using Color = System.Drawing.Color;
using PixelFormat = System.Drawing.Imaging.PixelFormat;
using Point = System.Drawing.Point;

namespace PatternRecognitionExample
{
    public partial class MainWindow : Window
    {
        private string[] _symbolsArray = new string[] { "0", "1", "5", "7"}; //Массив значений для распознавания

        private int educationCount = 3; //Количество вариаций объектов каждого класса

        private IList<IInputData<double>> _inputData;
        private InputSettings _inputSettings;
        private bool _isPaint;

        private MultiLayerSegment _segment;

        private WriteableBitmap _writeableBitmap = new WriteableBitmap(250, 250, 96, 96,
            PixelFormats.Pbgra32, null); //Инициализация холста

        public MainWindow()
        {
            InitializeComponent();

            Loaded += MainWindow_Loaded;

            image1.IsEnabled = false;
            recognizeButton.IsEnabled = false;
            saveButton.IsEnabled = false;

            Logger.Current.MessageReceived += InstanceOnMessageReceived;

            image1.MouseLeftButtonDown += image1_MouseLeftButtonDown;
            image1.MouseMove += image1_MouseMove;
            image1.MouseLeftButtonUp += image1_MouseLeftButtonUp;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void image1_MouseLeftButtonDown(object sender, MouseButtonEventArgs e) //Обработка нажатия ЛКМ на холсте
        {
            image1.CaptureMouse();
            _isPaint = true;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void image1_MouseMove(object sender, MouseEventArgs e) //Отрисовка на холсте мышкой
        {
            if (_isPaint)
            {
                _writeableBitmap.DrawEllipse((int) e.GetPosition(image1).X - 10, (int) e.GetPosition(image1).Y - 10,
                    (int) e.GetPosition(image1).X + 10, (int) e.GetPosition(image1).Y + 10, Colors.Black);

                image1.Source = _writeableBitmap;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void image1_MouseLeftButtonUp(object sender, MouseButtonEventArgs e) //Обработка прекращения нажатия ЛКМ на холсте
        {
            _isPaint = false;
            image1.ReleaseMouseCapture();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        private Bitmap GetBitmap(BitmapSource source) //Создание холста
        {
            var bmp = new Bitmap(
                source.PixelWidth,
                source.PixelHeight,
                PixelFormat.Format32bppPArgb);
            BitmapData data = bmp.LockBits(
                new Rectangle(Point.Empty, bmp.Size),
                ImageLockMode.WriteOnly,
                PixelFormat.Format32bppPArgb);
            source.CopyPixels(
                Int32Rect.Empty,
                data.Scan0,
                data.Height*data.Stride,
                data.Stride);
            bmp.UnlockBits(data);
            return bmp;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void MainWindow_Loaded(object sender, RoutedEventArgs e)
        {
            _writeableBitmap.FillRectangle(0, 0, 250, 250, Colors.White);

            image1.Source = new WriteableBitmap(_writeableBitmap);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void LearnButton_OnClick(object sender, RoutedEventArgs e) //Обработка кнопки "Распознать"
        {
            learnButton.IsEnabled = false; //Отключение возможности обучения

            var backgroundWorker = new BackgroundWorker(); //Инициализация асинхронново объекта

            backgroundWorker.DoWork += (o, args) => //Запуск асинхронки
            {
                var hiddenLayersDescription = new List<LayerDescription>(); //Инициализация листа скрытых слоев

                //Создание скрытых слоев с указанием количества нейронов и функции активации
                hiddenLayersDescription.Add(new LayerDescription(24, new RationalSigmoidTransferFunction()));
                hiddenLayersDescription.Add(new LayerDescription(16, new RationalSigmoidTransferFunction()));

                _inputSettings = new InputSettings(LayersCreationType.Manual, hiddenLayersDescription,
                    new RationalSigmoidTransferFunction()); //Параметры для Сети (входной, скрытый и выходной слои)

                _inputData = new BitmapDataNormalizator(_inputSettings).Normalize(CreateSymbolImages(true)); //Нормализация изображения

                var segment = new MultiLayerSegment {Title = "Segment1"}; //Параметр многослойности Сети

                //Задание параметнов обучения
                var educationStrategy = 
                    new BackpropagationEducationStrategy(new BackpropagationSettings
                    {
                        EducationSpeed = 0.1, //Скорость обучения
                        LogMessageFrequency = 1,
                        MaxEpoches = 100000, //Количество эпох
                        Momentum = 0.1,
                        NormalError = 1.5 //Допустимый порог ошибки
                    });

                educationStrategy.Train(segment, _inputData); //Обучение сети

                args.Result = segment;
            };

            //Измение состояния формы после завершения обучения
            backgroundWorker.RunWorkerCompleted += (o, args) =>
            {
                image1.IsEnabled = true;
                recognizeButton.IsEnabled = true;
                saveButton.IsEnabled = true;
                _segment = (MultiLayerSegment) args.Result;
            };

            backgroundWorker.RunWorkerAsync();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="logMessageEventArgs"></param>
        private void InstanceOnMessageReceived(object sender, LogMessageEventArgs logMessageEventArgs) //Логирование процесса обучения
        {
            Dispatcher.BeginInvoke(
                new Action<string>(p => { statusTextBlock.Text = string.Format("Logger message: {0}", p); }),
                logMessageEventArgs.Message);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="recreate"></param>
        /// <returns></returns>
        private List<Bitmap> CreateSymbolImages(bool recreate = false) //Создание изображений для обучения
        {
            var directoryInfo = new DirectoryInfo(Path.Combine(Directory.GetCurrentDirectory(), "Symbols"));
            if (!directoryInfo.Exists)
                directoryInfo.Create();

            var bitmaps = new List<Bitmap>();

            foreach (string s in _symbolsArray)
            {
                for (int i = 0; i < educationCount; i++)
                {
                    string path = Path.Combine(Directory.GetCurrentDirectory(), "Symbols", s + $"{i}.png");

                    Bitmap bitmap = GetBitmap(s);

                    bitmaps.Add(bitmap);

                    if (!recreate)
                        if (File.Exists(path)) continue;

                    Dispatcher.BeginInvoke(
                        new Action<string>(p => { statusTextBlock.Text = string.Format("Создание изображения {0}", p); }),
                        path);

                    bitmap.Save(path);
                }
            }

            return bitmaps;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="text"></param>
        /// <returns></returns>
        readonly string[] fontArray = new string[] //Набор шрифтов для создания изображений
        {
            "Verdana",
            "Arial",
            "Times new Roman",
            "Courier New",
            "Calibri",
            "Candara",
            "brush script mt",
            "dayton",
            "footlight mt light",
            "mv boli"
        };
        static Random rand = new Random();
        private Bitmap GetBitmap(string text)
        {
            var bitmap = new Bitmap(250, 250);
            using (Graphics gr = Graphics.FromImage(bitmap))
            {
                //Случайное задание положения и размеры объекта
                float xText = 0 + rand.Next(-8, 8);
                float yText = 0 + rand.Next(-8, 8);
                float fontSize = rand.Next(120, 180);

                gr.FillRectangle(new SolidBrush(Color.FromArgb(160, 255, 255, 255)), 0, 0, 250, 250);
                gr.DrawString(text, new Font(fontArray[rand.Next(0, fontArray.Length)], fontSize, System.Drawing.FontStyle.Bold),
                    new SolidBrush(Color.Black), xText, yText); //Отрисовка изображения для обучения
                return bitmap;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Button_Click_1(object sender, RoutedEventArgs e) //Очистка холста
        {
            _writeableBitmap = new WriteableBitmap(250, 250, 96, 96,
                PixelFormats.Pbgra32, null);
            _writeableBitmap.FillRectangle(0, 0, 250, 250, Colors.White);

            image1.Source = _writeableBitmap;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void ButtonBase_OnClick(object sender, RoutedEventArgs e) //Распознание пользовательского изображения
        {
            IInputData<double> input =
                new BitmapDataNormalizator(_inputSettings).Normalize(new List<Bitmap> {GetBitmap(_writeableBitmap)})[0];
            //Нормализация изображения на холсте

            IInputData<double> output = _segment.CalculateOutput(input); //Распознание объекта

            int index = output.Output.ToList().IndexOf(output.Output.Max());

            statusTextBlock.Text = string.Format("Эскиз похож на букву: {0}", _symbolsArray[index/educationCount]); //Вывод результата
        }

        /// <summary>
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void SaveButton_OnClick(object sender, RoutedEventArgs e) //Сохранение результатов обучения в бинарном файле
        {
            if (_segment != null)
            {
                var savingStrategy = new ByteArraySavingStrategy();

                var dialog = new SaveFileDialog();

                dialog.RestoreDirectory = true;

                dialog.Filter = "Bin Files .bin | *.bin";

                if (dialog.ShowDialog() == true)
                {
                    savingStrategy.SaveToStream(_segment, dialog.OpenFile());
                }
            }
        }

        /// <summary>
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void LoadButton_OnClick(object sender, RoutedEventArgs e) //Загрузка результатов обучения
        {
            if (_segment == null)
                _segment = new MultiLayerSegment();

            var loadingStrategy = new ByteArrayLoadingStrategy(null);

            var dialog = new OpenFileDialog();

            dialog.RestoreDirectory = true;

            dialog.Filter = "Bin Files .bin | *.bin";

            if (dialog.ShowDialog() == true)
            {
                loadingStrategy.LoadFromStream(_segment, dialog.OpenFile());

                image1.IsEnabled = true;
                recognizeButton.IsEnabled = true;
            }
        }
    }
}