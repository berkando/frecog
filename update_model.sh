#!/usr/bin/env bash

./lib/openface-0.2.1//util/align-dlib.py ./training-images/ align outerEyesAndNose ./aligned-images/ --size 96
./lib/openface-0.2.1/batch-represent/main.lua -outDir ./generated-embeddings/ -data ./aligned-images/
./lib/openface-0.2.1//demos/classifier.py train ./generated-embeddings/

