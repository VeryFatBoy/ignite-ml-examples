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
import org.apache.ignite.ml.knn.NNClassificationModel;
import org.apache.ignite.ml.knn.classification.KNNClassificationTrainer;
import org.apache.ignite.ml.knn.classification.NNStrategy;
import org.apache.ignite.ml.math.distances.EuclideanDistance;

import javax.cache.Cache;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import org.apache.ignite.ml.math.primitives.vector.VectorUtils;

public class ClientNode {

    public static void main(String... args) throws FileNotFoundException {
        IgniteConfiguration configuration = new IgniteConfiguration();
        configuration.setClientMode(false);

        try (Ignite ignite = Ignition.start(configuration)) {
            IgniteCache<Integer, IrisObservation> trainData = getCache(ignite, "IRIS_TRAIN");
            IgniteCache<Integer, IrisObservation> testData = getCache(ignite, "IRIS_TEST");

            loadData("src/main/resources/iris-train.csv", trainData);
            loadData("src/main/resources/iris-test.csv", testData);

            KNNClassificationTrainer trainer = new KNNClassificationTrainer();

            NNClassificationModel mdl = trainer.fit(
                    ignite,
                    trainData,
                    (k, v) -> VectorUtils.of(v.getFeatures()),     // Feature extractor.
                    (k, v) -> v.getFlowerClass())                  // Label extractor.
                    .withK(3)
                    .withDistanceMeasure(new EuclideanDistance())
                    .withStrategy(NNStrategy.WEIGHTED);

            System.out.println(">>> -----------------------------");
            System.out.println(">>> | Prediction | Ground Truth |");
            System.out.println(">>> -----------------------------");

            int amountOfErrors = 0;
            int totalAmount = 0;

            try (QueryCursor<Cache.Entry<Integer, IrisObservation>> cursor = testData.query(new ScanQuery<>())) {
                for (Cache.Entry<Integer, IrisObservation> testEntry : cursor) {
                    IrisObservation observation = testEntry.getValue();

                    double groundTruth = observation.getFlowerClass();
                    double prediction = mdl.apply(VectorUtils.of(observation.getFeatures()));

                    totalAmount++;
                    if (groundTruth != prediction)
                        amountOfErrors++;

                    System.out.printf(">>> | %.0f\t\t\t | %.0f\t\t\t|\n", prediction, groundTruth);
                }

                System.out.println(">>> -----------------------------");

                System.out.println("\n>>> Absolute amount of errors " + amountOfErrors);
                System.out.printf("\n>>> Accuracy %.2f\n", (1 - amountOfErrors / (double) totalAmount));
            }
        }
    }

    private static void loadData(String fileName, IgniteCache<Integer, IrisObservation> cache)
            throws FileNotFoundException {
        Scanner scanner = new Scanner(new File(fileName));

        int cnt = 0;
        while (scanner.hasNextLine()) {
            String row = scanner.nextLine();
            String[] cells = row.split(",");
            double[] features = new double[cells.length - 1];

            for (int i = 0; i < cells.length - 1; i++)
                features[i] = Double.valueOf(cells[i]);
            double flowerClass = Double.valueOf(cells[cells.length - 1]);

            cache.put(cnt++, new IrisObservation(features, flowerClass));
        }
    }

    private static IgniteCache<Integer, IrisObservation> getCache(Ignite ignite, String cacheName) {

        CacheConfiguration<Integer, IrisObservation> cacheConfiguration = new CacheConfiguration<>();
        cacheConfiguration.setName(cacheName);
        cacheConfiguration.setAffinity(new RendezvousAffinityFunction(false, 10));

        IgniteCache<Integer, IrisObservation> cache = ignite.createCache(cacheConfiguration);

        return cache;
    }

    private static class IrisObservation {

        private final double[] features;

        private final double flowerClass;

        public IrisObservation(double[] features, double flowerClass) {
            this.features = features;
            this.flowerClass = flowerClass;
        }

        public double[] getFeatures() {
            return features;
        }

        public double getFlowerClass() {
            return flowerClass;
        }
    }
}