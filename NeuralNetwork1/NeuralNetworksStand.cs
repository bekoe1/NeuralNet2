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

        private string _defaultStructure = "1024;150;10";

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
                if (Directory.Exists(_debugFolderPath))
                {
                    Directory.Delete(_debugFolderPath, true);
                }
                Directory.CreateDirectory(_debugFolderPath);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Не удалось создать папку отладки: " + ex.Message);
            }
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
            label1.Text = "Выберите цифру и нажмите 'Добавить пример'";
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
                MessageBox.Show("Camera Error: " + ex.Message);
            }
        }

        private void SaveDebugImage(Sample sample, string suffix)
        {
            try
            {
                int size = 32;
                Bitmap bmp = new Bitmap(size, size);

                for (int y = 0; y < size; y++)
                {
                    for (int x = 0; x < size; x++)
                    {
                        double val = sample.input[y * size + x];

                        int colorVal = (int)(val * 255);

                        colorVal = Math.Max(0, Math.Min(255, colorVal));

                        bmp.SetPixel(x, y, Color.FromArgb(colorVal, colorVal, colorVal));
                    }
                }

                string filename = $"{sample.actualClass}_{Guid.NewGuid().ToString().Substring(0, 4)}_{suffix}.png";
                string fullPath = Path.Combine(_debugFolderPath, filename);

                bmp.Save(fullPath, System.Drawing.Imaging.ImageFormat.Png);
            }
            catch (Exception ex)
            {
            }
        }

        private void btnTrainOne_Click(object sender, EventArgs e)
        {
            FigureType currentDigit = (FigureType)(int)classCounter.Value;

            Sample originalSample = _camera.ProcessCurrentFrame(currentDigit);
            if (originalSample == null) return;

            List<Sample> samplesToAdd = new List<Sample>();
            samplesToAdd.Add(originalSample);

            SaveDebugImage(originalSample, "ORIGINAL");

            Random rnd = new Random();
            for (int i = 0; i < 10; i++)
            {
                double[] augmentedInput = (double[])originalSample.input.Clone();
                for (int j = 0; j < augmentedInput.Length; j++)
                {
                    if (rnd.NextDouble() < 0.15)
                    {
                        double noise = (rnd.NextDouble() - 0.5) * 0.4;
                        augmentedInput[j] = Math.Max(0, Math.Min(1, augmentedInput[j] + noise));
                    }
                }

                var fakeSample = new Sample(augmentedInput, 10, currentDigit);
                samplesToAdd.Add(fakeSample);

                SaveDebugImage(fakeSample, $"AUG_{i}");
            }

            foreach (var s in samplesToAdd)
            {
                _trainingSet.AddSample(s);
            }

            if (Net != null) Net.Train(originalSample, 0.01, parallelCheckBox.Checked);

            label1.Text = $"Всего: {_trainingSet.Count} (+11). Цифра: {currentDigit}";
            label1.ForeColor = Color.Blue;
        }

        private async void button1_Click(object sender, EventArgs e)
        {
            if (_trainingSet.Count == 0)
            {
                MessageBox.Show("Сначала соберите примеры!");
                return;
            }

            label1.Text = "Идет обучение...";
            label1.ForeColor = Color.DarkOrange;
            SetButtonsEnabled(false);

            try
            {
                double error = await Task.Run(() =>
                    Net.TrainOnDataSet(
                        _trainingSet,
                        (int)EpochesCounter.Value,
                        (100 - AccuracyCounter.Value) / 100.0,
                        parallelCheckBox.Checked
                    )
                );

                label1.Text = $"Готово. Ошибка: {error:F5}";
                label1.ForeColor = Color.Green;
                StatusLabel.Text = "Ошибка: " + error;
            }
            catch (Exception ex)
            {
                MessageBox.Show("Ошибка: " + ex.Message);
            }
            finally
            {
                SetButtonsEnabled(true);
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            Sample sample = _camera.ProcessCurrentFrame();
            if (sample == null) return;

            SaveDebugImage(sample, "PREDICT_TRY");

            Net.Predict(sample);

            int result = (int)sample.recognizedClass;
            label1.Text = $"Распознано: {result}";

            if (result == (int)classCounter.Value)
                label1.ForeColor = Color.Green;
            else
                label1.ForeColor = Color.Red;

            label8.Text = "Выходы:\n" + string.Join("\n", sample.Output.Select((val, i) => $"{i}: {val:F4}"));
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
            StatusLabel.Text = $"Err: {error:F4} | Path: {_debugFolderPath}";
            int p = (int)(progress * 100);
            progressBar1.Value = Math.Min(100, Math.Max(0, p));
        }

        private void SetButtonsEnabled(bool enabled)
        {
            netTrainButton.Enabled = enabled;
            testNetButton.Enabled = enabled;
            trainOneButton.Enabled = enabled;
            recreateNetButton.Enabled = enabled;
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            _camera?.Dispose();
            base.OnFormClosing(e);
        }

        private void classCounter_ValueChanged(object sender, EventArgs e) { }
        private void recreateNetButton_MouseEnter(object sender, EventArgs e) { }
        private void netTrainButton_MouseEnter(object sender, EventArgs e) { }
        private void testNetButton_MouseEnter(object sender, EventArgs e) { }
        private void AccuracyCounter_Scroll(object sender, EventArgs e) { }
        private void parallelCheckBox_CheckedChanged(object sender, EventArgs e) { }
    }
}