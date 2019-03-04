/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.ignite.example;

import org.apache.ignite.Ignite;
import org.apache.ignite.IgniteCache;
import org.apache.ignite.Ignition;
import org.apache.ignite.cache.affinity.rendezvous.RendezvousAffinityFunction;
import org.apache.ignite.cache.query.QueryCursor;
import org.apache.ignite.cache.query.ScanQuery;
import org.apache.ignite.configuration.CacheConfiguration;
import org.apache.ignite.configuration.IgniteConfiguration;
import org.apache.ignite.ml.math.primitives.vector.VectorUtils;
import org.apache.ignite.ml.nn.UpdatesStrategy;
import org.apache.ignite.ml.optimization.updatecalculators.SimpleGDParameterUpdate;
import org.apache.ignite.ml.optimization.updatecalculators.SimpleGDUpdateCalculator;
import org.apache.ignite.ml.regressions.logistic.binomial.LogisticRegressionModel;
import org.apache.ignite.ml.regressions.logistic.binomial.LogisticRegressionSGDTrainer;

import javax.cache.Cache;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.Scanner;

public class ClientNode {

    public static void main(String... args) throws FileNotFoundException {
        IgniteConfiguration configuration = new IgniteConfiguration();
        configuration.setClientMode(false);

        try (Ignite ignite = Ignition.start(configuration)) {
            IgniteCache<Integer, FraudObservation> trainData = getCache(ignite, "FRAUD_TRAIN");
            IgniteCache<Integer, FraudObservation> testData = getCache(ignite, "FRAUD_TEST");

            Loader loader = new Loader();
            loader.load("fraud-train.csv", trainData);
            loader.load("fraud-test.csv", testData);

            LogisticRegressionSGDTrainer<?> trainer = new LogisticRegressionSGDTrainer<>()
                    .withUpdatesStgy(new UpdatesStrategy<>(
                            new SimpleGDUpdateCalculator(0.2),
                            SimpleGDParameterUpdate::sumLocal,
                            SimpleGDParameterUpdate::avg
                    ))
                    .withMaxIterations(100000)
                    .withLocIterations(100)
                    .withBatchSize(10)
                    .withSeed(123L);

            LogisticRegressionModel mdl = trainer.fit(
                    ignite,
                    trainData,
                    (k, v) -> VectorUtils.of(v.getFeatures()),     // Feature extractor.
                    (k, v) -> v.getFraudClass()                    // Label extractor.
            ).withRawLabels(true);

            System.out.println(">>> -----------------------------");
            System.out.println(">>> | Prediction | Ground Truth |");
            System.out.println(">>> -----------------------------");

            int amountOfErrors = 0;
            int totalAmount = 0;
            int[][] confusionMtx = {{0, 0}, {0, 0}};

            try (QueryCursor<Cache.Entry<Integer, FraudObservation>> cursor = testData.query(new ScanQuery<>())) {
                for (Cache.Entry<Integer, FraudObservation> testEntry : cursor) {
                    FraudObservation observation = testEntry.getValue();

                    double groundTruth = observation.getFraudClass();
                    double prediction = mdl.apply(VectorUtils.of(observation.getFeatures()));

                    totalAmount++;
                    if ((int) groundTruth != (int) prediction)
                        amountOfErrors++;

                    int idx1 = (int) prediction;
                    int idx2 = (int) groundTruth;

                    confusionMtx[idx1][idx2]++;

                    System.out.printf(">>> | %.4f\t | %.0f\t\t\t|\n", prediction, groundTruth);
                }

                System.out.println(">>> -----------------------------");

                System.out.println("\n>>> Absolute amount of errors " + amountOfErrors);
                System.out.printf("\n>>> Accuracy %.4f\n", (1 - amountOfErrors / (double) totalAmount));
                System.out.printf("\n>>> Precision %.4f\n", (double) confusionMtx[0][0] / (double) (confusionMtx[0][0]
                    + confusionMtx[0][1]));
                System.out.printf("\n>>> Recall %.4f\n", (double) confusionMtx[0][0] / (double) (confusionMtx[0][0] +
                    confusionMtx[1][0]));

                System.out.println("\n>>> Confusion matrix is " + Arrays.deepToString(confusionMtx));

                System.out.println();

                System.out.format("%32s%32s%32s", "|", "NO FRAUD |", " FRAUD |\n");
                System.out.format("%32s%4d%28s%4d%28s", "NO FRAUD |", confusionMtx[0][0], " (true positives) |",
                    confusionMtx[0][1], " (false positives)  |\n");
                System.out.format("%32s%32s%32s", "|", "  |", "  |\n");
                System.out.format("%32s%32s%32s", "|", "  |", "  |\n");
                System.out.format("%32s%4d%28s%4d%28s", "FRAUD |", confusionMtx[1][0], " (false negatives) |",
                    confusionMtx[1][1], " (true negatives) |\n");

                System.out.println();
            }
        }
    }

    private static IgniteCache<Integer, FraudObservation> getCache(Ignite ignite, String cacheName) {

        CacheConfiguration<Integer, FraudObservation> cacheConfiguration = new CacheConfiguration<>();
        cacheConfiguration.setName(cacheName);
        cacheConfiguration.setAffinity(new RendezvousAffinityFunction(false, 10));

        IgniteCache<Integer, FraudObservation> cache = ignite.getOrCreateCache(cacheConfiguration);

        return cache;
    }

    private static class FraudObservation {

        private final double[] features;

        private final double fraudClass;

        public FraudObservation(double[] features, double fraudClass) {
            this.features = features;
            this.fraudClass = fraudClass;
        }

        public double[] getFeatures() {
            return features;
        }

        public double getFraudClass() {
            return fraudClass;
        }
    }

    private static class Loader {
        private void load(String fileName, IgniteCache<Integer, FraudObservation> cache) throws FileNotFoundException {
            ClassLoader classLoader = getClass().getClassLoader();

            File file = new File(classLoader.getResource(fileName).getFile());

            Scanner scanner = new Scanner(file);

            int cnt = 0;
            while (scanner.hasNextLine()) {
                String row = scanner.nextLine();
                String[] cells = row.split(",");
                double[] features = new double[cells.length - 1];

                for (int i = 0; i < cells.length - 1; i++)
                    features[i] = Double.valueOf(cells[i]);
                double fraudClass = Double.valueOf(cells[cells.length - 1]);

                cache.put(cnt++, new FraudObservation(features, fraudClass));
            }
        }
    }
}