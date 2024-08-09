using Microsoft.ML.Data;


namespace Question2
{
    public class StudentData
    {
        [LoadColumn(0)]
        public float STG { get; set; }

        [LoadColumn(1)]
        public float SCG { get; set; }

        [LoadColumn(2)]
        public float STR { get; set; }

        [LoadColumn(3)]
        public float LPR { get; set; }

        [LoadColumn(4)]
        public float PEG { get; set; }

        [LoadColumn(5)]
        [ColumnName("Label")]
        public string UNS { get; set; }
    }

    public class StudentPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabel { get; set; }
    }

}
