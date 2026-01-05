using System;
using System.Diagnostics;
using System.Threading.Tasks;
using System.Linq;

namespace NeuralNetwork1
{
   
    internal class ThreadLocalData
    {
        public double Loss;
        public double[][][] WGrad;
        public double[][] BGrad;

        // БУФЕРЫ (чтобы избежать new double[] внутри цикла)
        public double[][] NeuronOutputs; // Выходы слоев
        public double[][] Delats;        // Ошибки слоев

        public ThreadLocalData(int numLayers, int[] structure)
        {
            Loss = 0;
            // Инициализация градиентов
            WGrad = new double[numLayers][][];
            BGrad = new double[numLayers][];

            // Инициализация буферов
            NeuronOutputs = new double[structure.Length][];
            Delats = new double[structure.Length][];

            for (int i = 0; i < structure.Length; i++)
            {
                NeuronOutputs[i] = new double[structure[i]];
                Delats[i] = new double[structure[i]];
            }

            for (int l = 0; l < numLayers; l++)
            {
                int prev = structure[l];
                int next = structure[l + 1];
                WGrad[l] = new double[next][];
                BGrad[l] = new double[next];
                for (int j = 0; j < next; j++)
                {
                    WGrad[l][j] = new double[prev];
                }
            }
        }
    }

    public class StudentNetwork : BaseNetwork
    {
        private int[] layers;
        private double[][][] weights;
        private double[][] biases;

        // Для одиночных предсказаний (UI thread)
        private double[][] globalOutputs;

        // --- RProp ---
        private double[][][] weightsGradient;
        private double[][] biasesGradient;
        private double[][][] prevWeightsGradient;
        private double[][] prevBiasesGradient;
        private double[][][] deltaWeights;
        private double[][] deltaBiases;

        private const double DeltaMin = 1e-6;
        private const double DeltaMax = 50.0;
        private const double InitialDelta = 0.1;
        private const double EtaPlus = 1.2;
        private const double EtaMinus = 0.5;

        private object syncLock = new object();

        public StudentNetwork(int[] structure)
        {
            ReInit(structure, 0.25);
        }

        public void ReInit(int[] structure, double initialLearningRate = 0.25)
        {
            layers = structure;
            int numConnectionLayers = structure.Length - 1;

            weights = new double[numConnectionLayers][][];
            biases = new double[numConnectionLayers][];
            globalOutputs = new double[structure.Length][];

            // RProp init
            weightsGradient = new double[numConnectionLayers][][];
            biasesGradient = new double[numConnectionLayers][];
            prevWeightsGradient = new double[numConnectionLayers][][];
            prevBiasesGradient = new double[numConnectionLayers][];
            deltaWeights = new double[numConnectionLayers][][];
            deltaBiases = new double[numConnectionLayers][];

            Random rand = new Random();
            globalOutputs[0] = new double[structure[0]]; // Входной слой

            for (int l = 0; l < numConnectionLayers; l++)
            {
                int prevNeurons = structure[l];
                int nextNeurons = structure[l + 1];

                weights[l] = new double[nextNeurons][];
                biases[l] = new double[nextNeurons];
                globalOutputs[l + 1] = new double[nextNeurons];

                weightsGradient[l] = new double[nextNeurons][];
                biasesGradient[l] = new double[nextNeurons];
                prevWeightsGradient[l] = new double[nextNeurons][];
                prevBiasesGradient[l] = new double[nextNeurons];
                deltaWeights[l] = new double[nextNeurons][];
                deltaBiases[l] = new double[nextNeurons];

                for (int i = 0; i < nextNeurons; i++)
                {
                    weights[l][i] = new double[prevNeurons];
                    weightsGradient[l][i] = new double[prevNeurons];
                    prevWeightsGradient[l][i] = new double[prevNeurons];
                    deltaWeights[l][i] = new double[prevNeurons];

                    biases[l][i] = (rand.NextDouble() * 2 - 1) * 0.1;
                    deltaBiases[l][i] = InitialDelta;

                    for (int j = 0; j < prevNeurons; j++)
                    {
                        weights[l][i][j] = (rand.NextDouble() * 2 - 1) * 0.1;
                        deltaWeights[l][i][j] = InitialDelta;
                    }
                }
            }
        }

        private double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

        // Старый метод (оставляем для одиночных вызовов из UI, где скорость не критична)
        protected override double[] Compute(double[] input)
        {
            // Копируем вход
            Array.Copy(input, globalOutputs[0], input.Length);

            for (int l = 0; l < weights.Length; l++)
            {
                int nextNeurons = weights[l].Length;
                int prevNeurons = weights[l][0].Length;

                for (int i = 0; i < nextNeurons; i++)
                {
                    double sum = biases[l][i];
                    for (int j = 0; j < prevNeurons; j++)
                    {
                        sum += weights[l][i][j] * globalOutputs[l][j];
                    }
                    globalOutputs[l + 1][i] = Sigmoid(sum);
                }
            }
            return globalOutputs[globalOutputs.Length - 1];
        }


        private void ComputeFast(double[] input, double[][] outputBuffer)
        {
            // Копируем входные данные в буфер первого слоя
            // (input может быть длиннее или короче буфера, если что-то пошло не так, но предполагаем совпадение)
            Array.Copy(input, outputBuffer[0], input.Length);

            for (int l = 0; l < weights.Length; l++)
            {
                for (int i = 0; i < weights[l].Length; i++)
                {
                    double sum = biases[l][i];
                    // Разворачиваем цикл для скорости, если возможно, но стандартный тоже ок
                    for (int j = 0; j < weights[l][i].Length; j++)
                    {
                        sum += weights[l][i][j] * outputBuffer[l][j];
                    }
                    outputBuffer[l + 1][i] = Sigmoid(sum);
                }
            }
        }

        private void CalculateGradientsFast(double[] expected, double[][] outputs,
                                            double[][][] wGrad, double[][] bGrad,
                                            double[][] deltasBuffer)
        {
            int numLayers = layers.Length;
            int lastLayerIdx = numLayers - 1;

            // 1. Ошибка выходного слоя
            for (int i = 0; i < layers[lastLayerIdx]; i++)
            {
                double actual = outputs[lastLayerIdx][i];
                double error = -(expected[i] - actual);
                double derivative = actual * (1.0 - actual);
                deltasBuffer[lastLayerIdx][i] = error * derivative;
            }

            // 2. Обратное распространение
            for (int l = numLayers - 2; l >= 0; l--)
            {
                int currentLayerSize = layers[l + 1]; // Слой, для которого считали deltas на прошлом шаге
                int prevLayerSize = layers[l];      // Слой, для которого считаем сейчас

                for (int j = 0; j < prevLayerSize; j++)
                {
                    double errorSum = 0;
                    for (int i = 0; i < currentLayerSize; i++)
                    {
                        // Накапливаем градиенты весов (пока мы тут)
                        wGrad[l][i][j] += deltasBuffer[l + 1][i] * outputs[l][j];

                        // Считаем ошибку для предыдущего слоя
                        errorSum += deltasBuffer[l + 1][i] * weights[l][i][j];
                    }

                    double prevOut = outputs[l][j];
                    double derivative = prevOut * (1.0 - prevOut);
                    deltasBuffer[l][j] = errorSum * derivative;
                }

                // Накапливаем градиенты смещений
                for (int i = 0; i < currentLayerSize; i++)
                    bGrad[l][i] += deltasBuffer[l + 1][i];
            }
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            // ВАЖНО: раньше прогресс отправлялся только после завершения эпохи.
            // На больших сетях (например 4096->1200) одна эпоха может считаться очень долго,
            // из-за чего казалось, что "обучение не начинается". Теперь шлём прогресс внутри эпохи.

            Stopwatch sw = Stopwatch.StartNew();
            double currentError = double.MaxValue;

            int n = samplesSet?.Count ?? 0;
            if (n <= 0 || epochsCount <= 0)
            {
                OnTrainProgress(1.0, double.NaN, sw.Elapsed);
                return double.NaN;
            }

            int numConnLayers = layers.Length - 1;

            // Используем все ядра
            var parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };

            // Сколько образцов обрабатываем между обновлениями прогресса
            // (примерно 10 обновлений на эпоху, но не слишком мелко)
            int chunkSize = Math.Max(16, Math.Min(200, n / 10));

            for (int epoch = 0; epoch < epochsCount; epoch++)
            {
                ResetGradients();
                double totalError = 0.0;
                int processed = 0;

                if (parallel)
                {
                    // Обрабатываем датасет кусками, чтобы можно было обновлять прогресс
                    for (int start = 0; start < n; start += chunkSize)
                    {
                        int end = Math.Min(n, start + chunkSize);

                        Parallel.For(
                            start, end,
                            parallelOptions,
                            () => new ThreadLocalData(numConnLayers, layers),
                            (idx, loopState, localData) =>
                            {
                                var sample = samplesSet.samples[idx];

                                // 1) Прямой проход (в буферы потока)
                                ComputeFast(sample.input, localData.NeuronOutputs);

                                double[] actualOutput = localData.NeuronOutputs[localData.NeuronOutputs.Length - 1];
                                double[] expectedOutput = sample.Output;

                                // 2) Ошибка
                                double sampleError = 0.0;
                                for (int i = 0; i < actualOutput.Length; i++)
                                {
                                    double diff = expectedOutput[i] - actualOutput[i];
                                    sampleError += diff * diff;
                                }
                                localData.Loss += sampleError / 2.0;

                                // 3) Градиенты (в буферы потока)
                                CalculateGradientsFast(
                                    expectedOutput,
                                    localData.NeuronOutputs,
                                    localData.WGrad,
                                    localData.BGrad,
                                    localData.Delats);

                                return localData;
                            },
                            (finalData) =>
                            {
                                lock (syncLock)
                                {
                                    totalError += finalData.Loss;
                                    AddGradients(weightsGradient, finalData.WGrad);
                                    AddBiases(biasesGradient, finalData.BGrad);
                                }
                            });

                        processed = end;

                        // Промежуточный прогресс (внутри эпохи)
                        double avgErrSoFar = totalError / Math.Max(1, processed);
                        double progress = (epoch + (processed / (double)n)) / Math.Max(1, epochsCount);
                        OnTrainProgress(progress, avgErrSoFar, sw.Elapsed);

                        if (avgErrSoFar < acceptableError)
                            break;
                    }
                }
                else
                {
                    // Однопоточная версия — тоже с промежуточным прогрессом
                    double[][] tempOutputs = new double[layers.Length][];
                    double[][] tempDeltas = new double[layers.Length][];
                    for (int i = 0; i < layers.Length; i++)
                    {
                        tempOutputs[i] = new double[layers[i]];
                        tempDeltas[i] = new double[layers[i]];
                    }

                    for (int idx = 0; idx < n; idx++)
                    {
                        var sample = samplesSet.samples[idx];

                        ComputeFast(sample.input, tempOutputs);
                        double[] actual = tempOutputs[tempOutputs.Length - 1];
                        double[] expected = sample.Output;

                        double sampleError = 0.0;
                        for (int k = 0; k < actual.Length; k++)
                        {
                            double d = expected[k] - actual[k];
                            sampleError += d * d;
                        }
                        totalError += sampleError / 2.0;

                        CalculateGradientsFast(expected, tempOutputs, weightsGradient, biasesGradient, tempDeltas);

                        processed = idx + 1;

                        if (processed % chunkSize == 0 || processed == n)
                        {
                            double avgErrSoFar = totalError / Math.Max(1, processed);
                            double progress = (epoch + (processed / (double)n)) / Math.Max(1, epochsCount);
                            OnTrainProgress(progress, avgErrSoFar, sw.Elapsed);

                            if (avgErrSoFar < acceptableError)
                                break;
                        }
                    }
                }

                // Обновляем веса один раз за эпоху (как и было)
                UpdateWeightsRProp();

                currentError = totalError / n;

                // Прогресс по эпохам
                OnTrainProgress((epoch + 1.0) / epochsCount, currentError, sw.Elapsed);

                if (currentError < acceptableError)
                    break;
            }

            OnTrainProgress(1.0, currentError, sw.Elapsed);
            return currentError;
        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            // Обучение одному образу не меняем, там скорость не важна
            int iter = 0;
            // Выделяем временные буферы
            double[][] tempOutputs = new double[layers.Length][];
            double[][] tempDeltas = new double[layers.Length][];
            for (int i = 0; i < layers.Length; i++)
            {
                tempOutputs[i] = new double[layers[i]];
                tempDeltas[i] = new double[layers[i]];
            }

            while (iter < 15)
            {
                ResetGradients();
                ComputeFast(sample.input, tempOutputs);
                CalculateGradientsFast(sample.Output, tempOutputs, weightsGradient, biasesGradient, tempDeltas);
                UpdateWeightsRProp();

                double err = 0;
                double[] actual = tempOutputs[tempOutputs.Length - 1];
                for (int i = 0; i < actual.Length; i++)
                {
                    double d = sample.Output[i] - actual[i];
                    err += d * d / 2;
                }

                if (err < acceptableError) break;
                iter++;
            }
            return iter;
        }

        public double[] getOutput()
        {
            if (globalOutputs != null && globalOutputs.Length > 0)
                return globalOutputs[globalOutputs.Length - 1];
            return null;
        }

        private void UpdateWeightsRProp()
        {
            for (int l = 0; l < weights.Length; l++)
            {
                for (int i = 0; i < weights[l].Length; i++)
                {
                    double grad = biasesGradient[l][i];
                    double prevGrad = prevBiasesGradient[l][i];
                    double delta = deltaBiases[l][i];
                    double change = grad * prevGrad;

                    if (change > 0)
                    {
                        delta = Math.Min(delta * EtaPlus, DeltaMax);
                        biases[l][i] -= Math.Sign(grad) * delta;
                        prevBiasesGradient[l][i] = grad;
                    }
                    else if (change < 0)
                    {
                        delta = Math.Max(delta * EtaMinus, DeltaMin);
                        prevBiasesGradient[l][i] = 0;
                    }
                    else
                    {
                        biases[l][i] -= Math.Sign(grad) * delta;
                        prevBiasesGradient[l][i] = grad;
                    }
                    deltaBiases[l][i] = delta;

                    for (int j = 0; j < weights[l][i].Length; j++)
                    {
                        grad = weightsGradient[l][i][j];
                        prevGrad = prevWeightsGradient[l][i][j];
                        delta = deltaWeights[l][i][j];
                        change = grad * prevGrad;

                        if (change > 0)
                        {
                            delta = Math.Min(delta * EtaPlus, DeltaMax);
                            weights[l][i][j] -= Math.Sign(grad) * delta;
                            prevWeightsGradient[l][i][j] = grad;
                        }
                        else if (change < 0)
                        {
                            delta = Math.Max(delta * EtaMinus, DeltaMin);
                            prevWeightsGradient[l][i][j] = 0;
                        }
                        else
                        {
                            weights[l][i][j] -= Math.Sign(grad) * delta;
                            prevWeightsGradient[l][i][j] = grad;
                        }
                        deltaWeights[l][i][j] = delta;
                    }
                }
            }
        }

        private void ResetGradients()
        {
            for (int l = 0; l < weightsGradient.Length; l++)
            {
                Array.Clear(biasesGradient[l], 0, biasesGradient[l].Length);
                for (int i = 0; i < weightsGradient[l].Length; i++)
                    Array.Clear(weightsGradient[l][i], 0, weightsGradient[l][i].Length);
            }
        }

        private void AddGradients(double[][][] target, double[][][] source)
        {
            for (int l = 0; l < target.Length; l++)
                for (int i = 0; i < target[l].Length; i++)
                    for (int j = 0; j < target[l][i].Length; j++)
                        target[l][i][j] += source[l][i][j];
        }

        private void AddBiases(double[][] target, double[][] source)
        {
            for (int l = 0; l < target.Length; l++)
                for (int i = 0; i < target[l].Length; i++)
                    target[l][i] += source[l][i];
        }
    }
}