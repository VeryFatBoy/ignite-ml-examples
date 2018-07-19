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
import org.apache.ignite.ml.math.impls.vector.DenseLocalOnHeapVector;
import org.apache.ignite.ml.regressions.linear.LinearRegressionLSQRTrainer;
import org.apache.ignite.ml.regressions.linear.LinearRegressionModel;
import org.apache.ignite.ml.trainers.DatasetTrainer;

import javax.cache.Cache;
import java.util.Scanner;

public class ClientNode {

    public static void main(String... args) {
        IgniteConfiguration configuration = new IgniteConfiguration();
        configuration.setClientMode(false);

        try (Ignite ignite = Ignition.start(configuration)) {
            // Create caches for train and test data.
            IgniteCache<Integer, HouseObservation> trainData = createCache(ignite, "BOSTON_HOUSING_TRAIN");
            IgniteCache<Integer, HouseObservation> testData = createCache(ignite, "BOSTON_HOUSING_TEST");

            // Load train and test data into created caches.
            loadData("boston-housing-train.csv", trainData);
            loadData("boston-housing-test.csv", testData);

            // Create a linear regression trainer and train the model.
            DatasetTrainer<LinearRegressionModel, Double> trainer = new LinearRegressionLSQRTrainer();

            System.out.println("Training started");
            LinearRegressionModel mdl = trainer.fit(
                    ignite,
                    trainData,
                    (k, v) -> v.getFeatures(),  // Feature extractor.
                    (k, v) -> v.getPrice()      // Label extractor.
            );
            System.out.println("Training completed");

            // Calculate score (R^2) on the test set. The coefficient R^2 is defined as (1 - u/v), where u is the
            // residual sum of squares ((y_true - y_pred) ** 2).sum() and v is the total sum of squares
            // ((y_true - y_true.mean()) ** 2).sum(). The best possible score is 1.0 and it can be negative (because the
            // model can be arbitrarily worse). A constant model that always predicts the expected value of y,
            // disregarding the input features, would get a R^2 score of 0.0.
            double meanPrice = getMeanPrice(testData);
            double u = 0, v = 0;

            try (QueryCursor<Cache.Entry<Integer, HouseObservation>> cursor = testData.query(new ScanQuery<>())) {
                for (Cache.Entry<Integer, HouseObservation> testEntry : cursor) {
                    HouseObservation observation = testEntry.getValue();

                    double realPrice = observation.getPrice();
                    double predictedPrice = mdl.apply(new DenseLocalOnHeapVector(observation.getFeatures()));

                    u += Math.pow(realPrice - predictedPrice, 2);
                    v += Math.pow(realPrice - meanPrice, 2);
                }
            }

            double score = 1 - u / v;

            System.out.println("Score : " + score);
        }
    }

    private static IgniteCache<Integer, HouseObservation> createCache(Ignite ignite, String name) {
        CacheConfiguration<Integer, HouseObservation> cacheConfiguration = new CacheConfiguration<>();
        cacheConfiguration.setAffinity(new RendezvousAffinityFunction(false, 10));
        cacheConfiguration.setName(name);

        return ignite.createCache(cacheConfiguration);
    }

    private static void loadData(String fileName, IgniteCache<Integer, HouseObservation> cache) {
        Scanner scanner = new Scanner(ClientNode.class.getClassLoader().getResourceAsStream(fileName));

        int cnt = 0;
        while (scanner.hasNextLine()) {
            String row = scanner.nextLine();
            String[] cells = row.split(",");
            double[] features = new double[cells.length - 1];

            for (int i = 0; i < cells.length - 1; i++)
                features[i] = Double.valueOf(cells[i]);

            double price = Double.valueOf(cells[cells.length - 1]);

            cache.put(cnt++, new HouseObservation(features, price));
        }
    }

    private static double getMeanPrice(IgniteCache<Integer, HouseObservation> cache) {
        int cnt = 0;
        double price = 0;

        try (QueryCursor<Cache.Entry<Integer, HouseObservation>> cursor = cache.query(new ScanQuery<>())) {
            for (Cache.Entry<Integer, HouseObservation> testEntry : cursor) {
                HouseObservation observation = testEntry.getValue();
                cnt += 1;
                price += observation.getPrice();
            }
        }

        return price / cnt;
    }

    private static class HouseObservation {

        private final double[] features;

        private final double price;

        public HouseObservation(double[] features, double price) {
            this.features = features;
            this.price = price;
        }

        public double[] getFeatures() {
            return features;
        }

        public double getPrice() {
            return price;
        }
    }
}
