using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace NeuralNetwork1
{
    public partial class NeuralNetworksStand : Form
    {
        private CameraProcessor _camera;
        private SamplesSet _trainingSet = new SamplesSet();

        private string _debugFolderPath = Path.Combine(Application.StartupPath, "DebugDataset");
        private string _defaultStructure = "1024;400;100;10"; // Сделал сеть поглубже

        private readonly Dictionary<string, Func<int[], BaseNetwork>> networksFabric;
        private Dictionary<string, BaseNetwork> networksCache = new Dictionary<string, BaseNetwork>();

        public BaseNetwork Net
        {
            get
            {
                if (netTypeBox.SelectedItem == null) return null;
                var selectedItem = (string)netTypeBox.SelectedItem;
                if (!networksCache.ContainsKey(selectedItem))
                    networksCache.Add(selectedItem, CreateNetwork(selectedItem));
                return networksCache[selectedItem];
            }
        }

        public NeuralNetworksStand(Dictionary<string, Func<int[], BaseNetwork>> networksFabric)
        {
            InitializeComponent();
            this.networksFabric = networksFabric;

            netTypeBox.Items.AddRange(this.networksFabric.Keys.Select(s => (object)s).ToArray());
            netTypeBox.SelectedIndex = 0;

            StartCamera();
            CustomizeInterface();
            PrepareDebugFolder();
        }

        private void PrepareDebugFolder()
        {
            try
            {
               // if (Directory.Exists(_debugFolderPath)) Directory.Delete(_debugFolderPath, true);
                Directory.CreateDirectory(_debugFolderPath);
            }
            catch { }
        }

        private void CustomizeInterface()
        {
            netStructureBox.Text = _defaultStructure;
            classCounter.Minimum = 0;
            classCounter.Maximum = 9;
            classCounter.Value = 0;

            trainOneButton.Text = "Добавить пример";
            netTrainButton.Text = "Обучить сеть";
            testNetButton.Text = "Распознать";
            recreateNetButton.Text = "Сброс сети";

            StatusLabel.Text = $"Фото сохраняются в: {_debugFolderPath}";
            label1.Text = "Выберите цифру и нажмите 'Добавить пример' или загрузите датасет";
        }

        // Обработчик загрузки датасета из папок
        private void BtnLoadDataset_Click(object sender, EventArgs e)
        {
            using (var dialog = new FolderBrowserDialog())
            {
                dialog.Description = "Выберите корневую папку с цифрами (внутри должны быть папки 0, 1 ... 9)";
                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    string rootPath = dialog.SelectedPath;
                    int loadedCount = 0;
                    _trainingSet = new SamplesSet(); // Сброс старого датасета

                    for (int i = 0; i < 10; i++)
                    {
                        string dir = Path.Combine(rootPath, i.ToString());
                        if (!Directory.Exists(dir)) continue;

                        var files = Directory.GetFiles(dir).Where(f => f.EndsWith(".png") || f.EndsWith(".jpg") || f.EndsWith(".bmp"));

                        // Параллельная загрузка изображений для скорости
                        object sync = new object();
                        Parallel.ForEach(files, file =>
                        {
                            Sample s = LoadSampleFromFile(file, (FigureType)i);
                            if (s != null)
                            {
                                lock (sync)
                                {
                                    _trainingSet.AddSample(s);
                                    loadedCount++;
                                }
                            }
                        });
                    }

                    if (loadedCount > 0)
                    {
                        label1.Text = $"Загружено: {loadedCount} фото.";
                        label1.ForeColor = Color.Blue;
                        MessageBox.Show($"Успешно загружено {loadedCount} изображений!");
                    }
                    else
                    {
                        MessageBox.Show("Изображения не найдены. Проверьте структуру папок (0..9).");
                    }
                }
            }
        }

        private Sample LoadSampleFromFile(string path, FigureType type)
        {
            try
            {
                using (Bitmap original = new Bitmap(path))
                {
                    // Приводим к размеру 32x32
                    using (Bitmap scaled = new Bitmap(original, new Size(32, 32)))
                    {
                        double[] input = new double[32 * 32];
                        for (int y = 0; y < 32; y++)
                        {
                            for (int x = 0; x < 32; x++)
                            {
                                Color c = scaled.GetPixel(x, y);
                                // Если изображение цветное или Grayscale:
                                // Принимаем, что белый фон (255) -> 0.0, черные чернила (0) -> 1.0
                                double brightness = (c.R + c.G + c.B) / (3.0 * 255.0);
                                input[y * 32 + x] = 1.0 - brightness;
                            }
                        }
                        return new Sample(input, 10, type);
                    }
                }
            }
            catch { return null; }
        }

        private void StartCamera()
        {
            _camera = new CameraProcessor();
            try
            {
                _camera.Start();
                Timer timer = new Timer();
                timer.Interval = 40;
                timer.Tick += (s, ev) =>
                {
                    var bmp = _camera.GetFrame();
                    if (bmp != null) pictureBox1.Image = bmp;
                };
                timer.Start();
            }
            catch (Exception ex)
            {
                MessageBox.Show("Ошибка камеры: " + ex.Message);
            }
        }

        private void SaveDebugImage(Sample sample, string suffix)
        {
            try
            {
                // Создаем подпапку для конкретной цифры (например, "DebugDataset/5")
                string classFolder = Path.Combine(_debugFolderPath, ((int)sample.actualClass).ToString());
                if (!Directory.Exists(classFolder)) Directory.CreateDirectory(classFolder);

                int size = 32;
                using (Bitmap bmp = new Bitmap(size, size))
                {
                    for (int y = 0; y < size; y++)
                    {
                        for (int x = 0; x < size; x++)
                        {
                            double val = sample.input[y * size + x];
                            // Инверсия: 1.0 -> Черный (0), 0.0 -> Белый (255)
                            int colorVal = (int)((1.0 - val) * 255);
                            colorVal = Math.Max(0, Math.Min(255, colorVal));
                            bmp.SetPixel(x, y, Color.FromArgb(colorVal, colorVal, colorVal));
                        }
                    }
                    // Сохраняем внутрь папки цифры
                    string filename = $"{Guid.NewGuid().ToString().Substring(0, 8)}_{suffix}.png";
                    bmp.Save(Path.Combine(classFolder, filename));
                }
            }
            catch (Exception ex)
            {
                // Можно вывести ошибку в консоль, если нужно
            }
        }


        // Обработчик кнопки "Обучить"
        private async void button1_Click(object sender, EventArgs e)
        {
            // Если данных нет - ругаемся
            if (_trainingSet.Count == 0)
            {
                MessageBox.Show("Сначала соберите примеры (с камеры или загрузите датасет)!");
                return;
            }

            // Блокируем интерфейс
            label1.Text = "Идет обучение...";
            label1.ForeColor = Color.DarkOrange;
            SetButtonsEnabled(false);
            progressBar1.Value = 0;

            try
            {
                // Считываем настройки в переменные (чтобы не лезть к UI из другого потока)
                var currentNet = Net;
                int epochs = (int)EpochesCounter.Value;
                double errorLimit = (100 - AccuracyCounter.Value) / 100.0;
                bool useParallel = parallelCheckBox.Checked;

                // Запускаем обучение в фоновом потоке
                double error = await Task.Run(() =>
                    currentNet.TrainOnDataSet(
                        _trainingSet,
                        epochs,
                        errorLimit,
                        useParallel
                    )
                );

                // После обучения проверяем точность
                double accuracy = _trainingSet.TestNeuralNetwork(currentNet);

                label1.Text = $"Ошибка: {error:F5} | Точность: {accuracy * 100:F1}%";
                label1.ForeColor = accuracy > 0.8 ? Color.Green : Color.Red;
                StatusLabel.Text = $"Last Run: Err={error:F5}, Acc={accuracy * 100:F1}%";
            }
            catch (Exception ex)
            {
                MessageBox.Show("Ошибка обучения: " + ex.Message);
            }
            finally
            {
                // Разблокируем интерфейс
                SetButtonsEnabled(true);
            }
        }
        private void btnTrainOne_Click(object sender, EventArgs e)
        {
            if (_camera == null) return;
            FigureType currentDigit = (FigureType)(int)classCounter.Value;

            // 1. Захватываем оригинал
            Sample originalSample = _camera.ProcessCurrentFrame(currentDigit);
            if (originalSample == null)
            {
                MessageBox.Show("Цифра не найдена в синем квадрате!");
                return;
            }

            // Сохраняем оригинал
            _trainingSet.AddSample(originalSample);
            SaveDebugImage(originalSample, "ORIG");

            // 2. ГЕНЕРАЦИЯ ДАТАСЕТА
            // Генерируем 200 штук для этой цифры (можно поставить хоть 1000)
            int countToGenerate = 200;

            // Используем наш новый Helper
            Sample[] augmentedSeries = AugmentationHelper.GenerateSeries(originalSample, countToGenerate);

            // Добавляем всю пачку в базу и сохраняем файлы
            for (int i = 0; i < augmentedSeries.Length; i++)
            {
                _trainingSet.AddSample(augmentedSeries[i]);

                // Обязательно сохраняем, чтобы потом можно было загрузить через "Загрузить датасет"
                // (Сохраняем каждый 5-й, чтобы диск не забивать, или убери условие, чтобы сохранить ВСЕ)
                if (i % 1 == 0) // Сейчас сохраняет ВСЕ (200 файлов)
                {
                    SaveDebugImage(augmentedSeries[i], $"AUG_{i}");
                }
            }

            // Обновляем текст на форме
            label1.Text = $"В базе: {_trainingSet.Count} (+1 оригинал, +{countToGenerate} копий)";
            label1.ForeColor = Color.Blue;
            StatusLabel.Text = $"Сгенерировано {countToGenerate} вариаций для цифры {(int)currentDigit}";
        }

        private void button2_Click(object sender, EventArgs e)
        {
            Sample sample = _camera.ProcessCurrentFrame();
            if (sample == null) return;

            Net.Predict(sample);

            int result = (int)sample.recognizedClass;
            label1.Text = $"Распознано: {result}";
            label1.ForeColor = (result == (int)classCounter.Value) ? Color.Green : Color.Red;

            // Выводим вероятности
            string probs = "";
            for (int i = 0; i < sample.Output.Length; i++)
                probs += $"{i}: {sample.Output[i]:F2}\n";
            label8.Text = probs;
        }

        private void button3_Click(object sender, EventArgs e)
        {
            networksCache.Clear();
            label1.Text = "Сеть сброшена.";
            label1.ForeColor = Color.Black;
        }

        private void pictureBox1_MouseClick(object sender, MouseEventArgs e)
        {
            button2_Click(sender, e);
        }

        private BaseNetwork CreateNetwork(string networkName)
        {
            var parts = netStructureBox.Text.Split(';').Select(int.Parse).ToArray();
            var network = networksFabric[networkName](parts);
            network.TrainProgress += UpdateLearningInfo;
            return network;
        }

        public void UpdateLearningInfo(double progress, double error, TimeSpan elapsedTime)
        {
            if (progressBar1.InvokeRequired)
            {
                progressBar1.Invoke(new TrainProgressHandler(UpdateLearningInfo), progress, error, elapsedTime);
                return;
            }
            StatusLabel.Text = $"Err: {error:F5}";
            int p = (int)(progress * 100);
            progressBar1.Value = Math.Min(100, Math.Max(0, p));
            elapsedTimeLabel.Text = "Время: " + elapsedTime.ToString(@"mm\:ss");
        }

        private void SetButtonsEnabled(bool enabled)
        {
            netTrainButton.Enabled = enabled;
            testNetButton.Enabled = enabled;
            trainOneButton.Enabled = enabled;
            recreateNetButton.Enabled = enabled;
            btnLoadDataset.Enabled = enabled;
            groupBox1.Enabled = enabled;
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            _camera?.Dispose();
            base.OnFormClosing(e);
        }

        // Пустые обработчики, чтобы дизайнер не ругался
        private void classCounter_ValueChanged(object sender, EventArgs e) { }
        private void recreateNetButton_MouseEnter(object sender, EventArgs e) { }
        private void netTrainButton_MouseEnter(object sender, EventArgs e) { }
        private void testNetButton_MouseEnter(object sender, EventArgs e) { }
        private void AccuracyCounter_Scroll(object sender, EventArgs e) { }
        private void parallelCheckBox_CheckedChanged(object sender, EventArgs e) { }
    }
}