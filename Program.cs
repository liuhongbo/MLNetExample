using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Microsoft.ML;

public class Customer
{
    [LoadColumn(0)]
    [ColumnName(@"Age")]
    public float Age { get; set; }

    [LoadColumn(1)]
    [ColumnName(@"Gender")]
    public string Gender { get; set; }

    [LoadColumn(2)]
    [ColumnName(@"Income")]
    public float Income { get; set; }

    [LoadColumn(3)]
    [ColumnName(@"MaritalStatus")]
    public string MaritalStatus { get; set; }

    [LoadColumn(4)]
    [ColumnName(@"Purchase")]
    public bool Purchase { get; set; }
}

public class Prediction
{
    [ColumnName("PredictedLabel")]
    public bool Purchase { get; set; }
}

class Program
{
    static void Main(string[] args)
    {
        var mlContext = new MLContext();

        var dataFile = "D:\\Medium\\MLNetExample\\customer.csv";
        var modelFile = "D:\\Medium\\MLNetExample\\model.mlnet";

        // Load the data from a CSV file
        var data = mlContext.Data.LoadFromTextFile<Customer>(path: dataFile,
                                                               separatorChar:',', 
                                                               hasHeader:true);


        // Split the data into training and testing datasets
        var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
        var trainData = split.TrainSet;
        var testData = split.TestSet;

        // Define the machine learning pipeline   
        var pipeline = mlContext.Transforms.Categorical.OneHotEncoding(new[] {  new InputOutputColumnPair(@"Age", @"Age"),
                                                                                new InputOutputColumnPair(@"Income", @"Income"),
                                                                                new InputOutputColumnPair(@"Gender", @"Gender"),
                                                                                new InputOutputColumnPair(@"MaritalStatus", @"MaritalStatus"), }, 
                                                                                outputKind: OneHotEncodingEstimator.OutputKind.Indicator)                                  
                                   .Append(mlContext.Transforms.Concatenate(@"Features", new[] { @"Gender", @"MaritalStatus", @"Age", @"Income" }))
                                   .Append(mlContext.BinaryClassification.Trainers.FastTree(new FastTreeBinaryTrainer.Options() { NumberOfLeaves = 4, MinimumExampleCountPerLeaf = 20, NumberOfTrees = 4, MaximumBinCountPerFeature = 254, FeatureFraction = 1, LearningRate = 0.1, LabelColumnName = @"Purchase", FeatureColumnName = @"Features" }));


        // Train the model
        var model = pipeline.Fit(trainData);

        // Evaluate the model on the testing dataset
        var predictions = model.Transform(testData);
        var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Purchase");
        Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");

        //Save model to file
        DataViewSchema dataViewSchema = data.Schema;
        using (var fs = File.Create(modelFile))
        {
            mlContext.Model.Save(model, dataViewSchema, fs);
        }

        // Load model from file
        mlContext.Model.Load(modelFile, out var _);
        var predictionEngine = mlContext.Model.CreatePredictionEngine<Customer, Prediction>(model);

        // Make a prediction on new data
        var customer = new Customer { Age = 35, Gender = "Male", Income = 50000, MaritalStatus = "Single" };
        var prediction = predictionEngine.Predict(customer);

        Console.WriteLine($"Prediction: {prediction.Purchase}");
    }
}
