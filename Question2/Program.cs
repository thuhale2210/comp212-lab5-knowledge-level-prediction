using Microsoft.ML;
using System;
using System.IO;
using System.Linq;

namespace Question2
{
    internal class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "Student.csv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "StudentKnowledgeModel.zip");

        static void Main(string[] args)
        {
            // Create a new ML context
            var mlContext = new MLContext(seed: 0);

            // Load the data
            IDataView dataView = mlContext.Data.LoadFromTextFile<StudentData>(_dataPath, hasHeader: true, separatorChar: ',');

            // Split the data into train and test sets
            var dataSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            var trainData = dataSplit.TrainSet;
            var testData = dataSplit.TestSet;

            // Define the data preparation and training pipeline
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.Transforms.Concatenate("Features", "STG", "SCG", "STR", "LPR", "PEG"))
                .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy())
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // Train the model
            var model = pipeline.Fit(trainData);

            // Save model
            using (var fileStream = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(model, dataView.Schema, fileStream);
            }

            // Evaluate the model
            var testMetrics = mlContext.MulticlassClassification.Evaluate(model.Transform(testData));

            // Display the evaluation results
            Console.WriteLine("======== Model Evaluation Metrics ========");
            Console.WriteLine($" Log-loss: {testMetrics.LogLoss:F4}");
            Console.WriteLine($" Log-loss for class 'very_low': {testMetrics.PerClassLogLoss[0]:F4}");
            Console.WriteLine($" Log-loss for class 'Low': {testMetrics.PerClassLogLoss[1]:F4}");
            Console.WriteLine($" Log-loss for class 'Middle': {testMetrics.PerClassLogLoss[2]:F4}");
            Console.WriteLine($" Log-loss for class 'High': {testMetrics.PerClassLogLoss[3]:F4}");
            Console.WriteLine("==========================================\n");

            // Use the model for prediction
            var predictionEngine = mlContext.Model.CreatePredictionEngine<StudentData, StudentPrediction>(model);
            var sampleStudent1 = new StudentData { STG = 0.1f, SCG = 0.1f, STR = 0.7f, LPR = 0.15f, PEG = 0.9f };
            var sampleStudent2 = new StudentData { STG = 0.12f, SCG = 0.05f, STR = 0.8f, LPR = 0.12f, PEG = 0.11f };
            var prediction1 = predictionEngine.Predict(sampleStudent1);
            var prediction2 = predictionEngine.Predict(sampleStudent2);

            // Display the samples and prediction results
            Console.WriteLine("=========== Prediction Result ============");
            Console.WriteLine(" Sample Student 1:");
            Console.WriteLine($" STG: {sampleStudent1.STG}");
            Console.WriteLine($" SCG: {sampleStudent1.SCG}");
            Console.WriteLine($" STR: {sampleStudent1.STR}");
            Console.WriteLine($" LPR: {sampleStudent1.LPR}");
            Console.WriteLine($" PEG: {sampleStudent1.PEG}");
            Console.WriteLine($" Predicted knowledge level: {prediction1.PredictedLabel}");
            Console.WriteLine("------------------------------------------");
            Console.WriteLine(" Sample Student 2:");
            Console.WriteLine($" STG: {sampleStudent2.STG}");
            Console.WriteLine($" SCG: {sampleStudent2.SCG}");
            Console.WriteLine($" STR: {sampleStudent2.STR}");
            Console.WriteLine($" LPR: {sampleStudent2.LPR}");
            Console.WriteLine($" PEG: {sampleStudent2.PEG}");
            Console.WriteLine($" Predicted knowledge level: {prediction2.PredictedLabel}");
            Console.WriteLine("==========================================\n");
        }
    }
}
