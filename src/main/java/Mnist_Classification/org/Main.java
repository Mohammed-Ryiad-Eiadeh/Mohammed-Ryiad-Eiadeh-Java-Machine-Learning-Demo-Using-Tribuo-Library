package Mnist_Classification.org;

import org.tribuo.MutableDataset;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.dtree.CARTClassificationTrainer;
import org.tribuo.classification.ensemble.FullyWeightedVotingCombiner;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.liblinear.LibLinearClassificationTrainer;
import org.tribuo.classification.mnb.MultinomialNaiveBayesTrainer;
import org.tribuo.classification.sgd.fm.FMClassificationTrainer;
import org.tribuo.classification.sgd.kernel.KernelSVMTrainer;
import org.tribuo.classification.sgd.linear.LogisticRegressionTrainer;
import org.tribuo.classification.sgd.objectives.Hinge;
import org.tribuo.classification.xgboost.XGBoostClassificationTrainer;
import org.tribuo.common.nearest.KNNClassifierOptions;
import org.tribuo.data.columnar.FieldProcessor;
import org.tribuo.data.columnar.RowProcessor;
import org.tribuo.data.columnar.processors.field.DoubleFieldProcessor;
import org.tribuo.data.columnar.processors.response.FieldResponseProcessor;
import org.tribuo.data.csv.CSVDataSource;
import org.tribuo.ensemble.BaggingTrainer;
import org.tribuo.math.distance.DistanceType;
import org.tribuo.math.kernel.Linear;
import org.tribuo.math.optimisers.AdaGradRDA;

import java.nio.file.Paths;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;

public class Main {
    public static void main(String[] args) {
        // creat row processor to handle the header of the data
        var RowProcessor = GetRowProcessor();

        // read the training and testing dataset
        var TrainData = new CSVDataSource<>(Paths.get(System.getProperty("user.dir"), "\\mnist_test.csv"), RowProcessor, true);
        var TestData = new CSVDataSource<>(Paths.get(System.getProperty("user.dir"),"\\mnist_test.csv"), RowProcessor, true);

        // generate Tribuo form data portions of the read daasets
        var train = new MutableDataset<>(TrainData);
        var test = new MutableDataset<>(TestData);

        // show the training data features like size
        System.out.printf("train data size = %d, number of features = %d, number of labels = %d%n", train.size(),
                train.getFeatureMap().size(),
                train.getOutputInfo().size());

        // show the testing data features like size
        System.out.printf("test data size = %d, number of features = %d, number of labels = %d%n", test.size(),
                test.getFeatureMap().size(),
                test.getOutputInfo().size());

        // define label evaluation for evaluation purposes
        LabelEvaluation Evaluator;

        // define, train, and test factorization machine classifier
        var FactorizationMachine = new FMClassificationTrainer(new Hinge(), new AdaGradRDA(0.1, 0.8), 1, 1000, 1, Trainer.DEFAULT_SEED, 10, 0.2);
        var FMLearner = FactorizationMachine.train(train);
        Evaluator = new LabelEvaluator().evaluate(FMLearner, test);
        System.out.println("----------------Factorization Machine Performance\n" + Evaluator + "\n----------------Confusion Matrix \n" + Evaluator.getConfusionMatrix() + "\n\n");

        // define, train, and test bagging algorithm based on FM classifier
        var BaggingTrainer = new BaggingTrainer<>(FactorizationMachine, new FullyWeightedVotingCombiner(), 10);
        var BaggingLearner = BaggingTrainer.train(train);
        Evaluator = new LabelEvaluator().evaluate(BaggingLearner, test);
        System.out.println("----------------Bagging based FM Performance\n" + Evaluator + "\n----------------Confusion Matrix \n" + Evaluator.getConfusionMatrix() + "\n\n");

        // define, train, and test SVM classifier
        var SVMTrainer = new KernelSVMTrainer(new Linear(), 0.1, 1, Trainer.DEFAULT_SEED);
        var SVMLearner = SVMTrainer.train(train);
        Evaluator = new LabelEvaluator().evaluate(SVMLearner, test);
        System.out.println("----------------Support Vector Performance\n" + Evaluator + "\n----------------Confusion Matrix \n" + Evaluator.getConfusionMatrix() + "\n\n");

        // define, train, and test LR classifier
        var LRTrainer = new LogisticRegressionTrainer();
        var LRLearner = LRTrainer.train(train);
        Evaluator = new LabelEvaluator().evaluate(LRLearner, test);
        System.out.println("----------------Logistic Regression Performance\n" + Evaluator + "\n----------------Confusion Matrix \n" + Evaluator.getConfusionMatrix() + "\n\n");

        // define, train, and test naive bayes classifier
        var MBayesTrainer = new MultinomialNaiveBayesTrainer();
        var MBLearner = MBayesTrainer.train(train);
        Evaluator = new LabelEvaluator().evaluate(MBLearner, test);
        System.out.println("----------------MBayes Performance\n" + Evaluator + "\n----------------Confusion Matrix \n" + Evaluator.getConfusionMatrix() + "\n\n");

        // define, train, and test KNN classifier
        var KNNTrainer = new KNNClassifierOptions();
        KNNTrainer.knnK = 3;
        KNNTrainer.distType = DistanceType.COSINE;
        var KNNLearner = KNNTrainer.getTrainer().train(train);
        Evaluator = new LabelEvaluator().evaluate(KNNLearner, test);
        System.out.println("----------------KNN Performance\n" + Evaluator + "\n----------------Confusion Matrix \n" + Evaluator.getConfusionMatrix() + "\n\n");

        // define, train, and test CART classifier
        var CartTrainer = new CARTClassificationTrainer();
        var CartLearner = CartTrainer.train(train);
        Evaluator = new LabelEvaluator().evaluate(CartLearner, test);
        System.out.println("----------------CART Tree Performance\n" + Evaluator + "\n----------------Confusion Matrix \n" + Evaluator.getConfusionMatrix() + "\n\n");

        // define, train, and test LibLinear classifier
        var LibLinTrainer = new LibLinearClassificationTrainer();
        var LibLinLearner = LibLinTrainer.train(train);
        Evaluator = new LabelEvaluator().evaluate(LibLinLearner, test);
        System.out.println("----------------LibLinear Performance\n" + Evaluator + "\n----------------Confusion Matrix \n" + Evaluator.getConfusionMatrix() + "\n\n");

        // define, train, and test XGBoost classifier
        var XGBTrainer = new XGBoostClassificationTrainer(10);
        var XGBLearner = XGBTrainer.train(train);
        Evaluator = new LabelEvaluator().evaluate(XGBLearner, test);
        System.out.println("----------------XGBoost Performance\n" + Evaluator + "\n----------------Confusion Matrix \n" + Evaluator.getConfusionMatrix() + "\n\n");
    }

    /**
     * this method is used to construct Row Processor to process the data header
     * @return the constructed row processor
     */
    static RowProcessor<Label> GetRowProcessor() {
        // creat hashmap to hold the fields name of the dataset
        var FeatureProcessor = new HashMap<String, FieldProcessor>();
        FeatureProcessor.put("At.*", new DoubleFieldProcessor("At.*"));

        // creat class label processor for the classes in the dataset to construct the label factory
        var ClassProcessor = new FieldResponseProcessor<>("label", "nan", new LabelFactory());

        // return the row processor of the generated labels
        return new RowProcessor<>(new LinkedList<>(),
                null,
                ClassProcessor,
                new HashMap<>(),
                FeatureProcessor,
                Collections.emptySet());
    }
}