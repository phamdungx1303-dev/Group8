using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;

namespace SentenceSimilarityApp
{
    // 1. Điều chỉnh cấu trúc dữ liệu đầu vào cho khớp với file trainnew.tsv
    public class SentencePair
    {
        // Sử dụng LoadColumn với đúng vị trí cột trong file TSV
        [LoadColumn(7)] public string Sentence1 { get; set; }
        [LoadColumn(8)] public string Sentence2 { get; set; }
        [LoadColumn(9)] public float Label { get; set; } // Điểm số từ 0.0 đến 5.0
    }

    public class SimilarityPrediction
    {
        [ColumnName("Score")]
        public float Score { get; set; }
    }

    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext(seed: 0);

            // 2. Nạp file dữ liệu trainnew.tsv
            // Đảm bảo đường dẫn này trỏ đúng đến file của bạn
            string dataPath = "C:\\Users\\QUANGANH\\source\\repos\\SentenceSimilarityApp\\SentenceSimilarityApp\\trainnew.tsv";

            // QUAN TRỌNG: separatorChar là '\t' vì đây là file TSV
            IDataView data = mlContext.Data.LoadFromTextFile<SentencePair>(
                dataPath,
                hasHeader: true, // File của bạn có dòng tiêu đề (index, genre,...)
                separatorChar: '\t');

            // Chia dữ liệu: 80% để train, 20% để test (đánh giá)
            var splitDataView = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

            // 3. Xây dựng Pipeline xử lý văn bản
            var pipeline = mlContext.Transforms.Text.NormalizeText("Normalized1", nameof(SentencePair.Sentence1))
            // Cắt câu 1 thành các từ rời rạc
            .Append(mlContext.Transforms.Text.TokenizeIntoWords("Tokens1", "Normalized1"))
            // Chuyển các từ thành Tọa độ Vector (Dùng mô hình GloVe 50 chiều)
            .Append(mlContext.Transforms.Text.ApplyWordEmbedding("Vector1", "Tokens1",
            Microsoft.ML.Transforms.Text.WordEmbeddingEstimator.PretrainedModelKind.GloVe50D))

             // Làm tương tự với Câu 2
            .Append(mlContext.Transforms.Text.NormalizeText("Normalized2", nameof(SentencePair.Sentence2)))
            .Append(mlContext.Transforms.Text.TokenizeIntoWords("Tokens2", "Normalized2"))
            .Append(mlContext.Transforms.Text.ApplyWordEmbedding("Vector2", "Tokens2",
            Microsoft.ML.Transforms.Text.WordEmbeddingEstimator.PretrainedModelKind.GloVe50D))

             // Ghép 2 tọa độ Vector của 2 câu lại với nhau để đưa vào thuật toán
            .Append(mlContext.Transforms.Concatenate("Features", "Vector1", "Vector2"))
                .Append(mlContext.Regression.Trainers.FastTree(
                    labelColumnName: "Label",
                    featureColumnName: "Features",
                    // Các tham số quan trọng cần tinh chỉnh:
                    numberOfLeaves: 20,         // Độ phức tạp của mỗi cây (Nên thử: 10, 20, 50)
                    numberOfTrees: 100,         // Số lượng cây (Nên thử: 50, 100, 200)
                    minimumExampleCountPerLeaf: 10, // Giúp chống Overfitting (Nên thử: 5, 10, 20)
                     learningRate: 0.2           // Tốc độ học (Nên thử: 0.05, 0.1, 0.2)
                ));

            // 4. Huấn luyện mô hình (Training)
            Console.WriteLine("Dang tien hành huan luyen mo hinh...");
            var model = pipeline.Fit(splitDataView.TrainSet);

            // 5. Đánh giá mô hình trên tập Test
            Console.WriteLine("Đang đanh gia mo hinh...");
            var predictions = model.Transform(splitDataView.TestSet);
            var metrics = mlContext.Regression.Evaluate(predictions, "Label");

            Console.WriteLine($"--- Đanh gia mo hinh ---");
            Console.WriteLine($"R-Squared (R^2): {metrics.RSquared:0.##} (Cang gan 1 cang tot)");
            Console.WriteLine($"Root Mean Squared Error (RMSE): {metrics.RootMeanSquaredError:0.##} (Cang nho cang tot");

            // 6. Dự đoán thử nghiệm một câu trong file của bạn
            var predictionEngine = mlContext.Model.CreatePredictionEngine<SentencePair, SimilarityPrediction>(model);
            var sample = new SentencePair
            {
                Sentence1 = "I don't like you",
                Sentence2 = "I hate you"
            };
            var result = predictionEngine.Predict(sample);

            Console.WriteLine($"--- Du đoan moi ---");
            Console.WriteLine($"Câu 1: {sample.Sentence1}");
            Console.WriteLine($"Câu 2: {sample.Sentence2}");
            Console.WriteLine($"Do tuong tu (Du đoan): {result.Score:0.##} / 5.0");
        }
    }
}