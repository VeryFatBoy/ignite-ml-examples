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
import org.apache.ignite.ml.clustering.kmeans.KMeansModel;
import org.apache.ignite.ml.clustering.kmeans.KMeansTrainer;
import org.apache.ignite.ml.math.Tracer;
import org.apache.ignite.ml.math.distances.EuclideanDistance;

import javax.cache.Cache;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.Scanner;
import org.apache.ignite.ml.math.primitives.vector.VectorUtils;

public class ClientNode {

    public static void main(String... args) throws FileNotFoundException {
        IgniteConfiguration configuration = new IgniteConfiguration();
        configuration.setClientMode(false);

        try (Ignite ignite = Ignition.start(configuration)) {
            IgniteCache<Integer, TitanicObservation> trainData = getCache(ignite, "TITANIC_TRAIN");
            IgniteCache<Integer, TitanicObservation> testData = getCache(ignite, "TITANIC_TEST");

            loadData("src/main/resources/titanic-train.csv", trainData);
            loadData("src/main/resources/titanic-test.csv", testData);

            KMeansTrainer trainer = new KMeansTrainer()
                    .withAmountOfClusters(2)
                    .withDistance(new EuclideanDistance())
                    .withSeed(123L);

            KMeansModel mdl = trainer.fit(
                    ignite,
                    trainData,
                    (k, v) -> VectorUtils.of(v.getFeatures()),        // Feature extractor.
                    (k, v) -> v.getSurvivedClass()                    // Label extractor.
            );

            System.out.println(">>> KMeans centroids");
            Tracer.showAscii(mdl.getCenters()[0]);
            Tracer.showAscii(mdl.getCenters()[1]);
            System.out.println(">>>");

            System.out.println(">>> -----------------------------");
            System.out.println(">>> | Prediction | Ground Truth |");
            System.out.println(">>> -----------------------------");

            int amountOfErrors = 0;
            int totalAmount = 0;
            int[][] confusionMtx = {{0, 0}, {0, 0}};

            try (QueryCursor<Cache.Entry<Integer, TitanicObservation>> cursor = testData.query(new ScanQuery<>())) {
                for (Cache.Entry<Integer, TitanicObservation> testEntry : cursor) {
                    TitanicObservation observation = testEntry.getValue();

                    double groundTruth = observation.getSurvivedClass();
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
                System.out.printf("\n>>> Precision %.4f\n", (double) confusionMtx[0][0] / (double) (confusionMtx[0][0] + confusionMtx[0][1]));
                System.out.printf("\n>>> Recall %.4f\n", (double) confusionMtx[0][0] / (double) (confusionMtx[0][0] + confusionMtx[1][0]));
                System.out.println("\n>>> Confusion matrix is " + Arrays.deepToString(confusionMtx));
            }
        }
    }

    private static void loadData(String fileName, IgniteCache<Integer, TitanicObservation> cache)
            throws FileNotFoundException {
        Scanner scanner = new Scanner(new File(fileName));

        int cnt = 0;
        while (scanner.hasNextLine()) {
            String row = scanner.nextLine();
            String[] cells = row.split(",");
            double[] features = new double[cells.length - 1];

            for (int i = 0; i < cells.length - 1; i++)
                features[i] = Double.valueOf(cells[i]);
            double survivedClass = Double.valueOf(cells[cells.length - 1]);

            cache.put(cnt++, new TitanicObservation(features, survivedClass));
        }
    }

    private static IgniteCache<Integer, TitanicObservation> getCache(Ignite ignite, String cacheName) {

        CacheConfiguration<Integer, TitanicObservation> cacheConfiguration = new CacheConfiguration<>();
        cacheConfiguration.setName(cacheName);
        cacheConfiguration.setAffinity(new RendezvousAffinityFunction(false, 10));

        IgniteCache<Integer, TitanicObservation> cache = ignite.createCache(cacheConfiguration);

        return cache;
    }

    private static class TitanicObservation {

        private final double[] features;

        private final double survivedClass;

        public TitanicObservation(double[] features, double survivedClass) {
            this.features = features;
            this.survivedClass = survivedClass;
        }

        public double[] getFeatures() {
            return features;
        }

        public double getSurvivedClass() {
            return survivedClass;
        }
    }
}