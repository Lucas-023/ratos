# MABe Challenge: Mouse Behavior Recognition

This repository contains the source code and documentation for my solution to the MABe (Multi-Agent Behavior) Challenge. The goal of this project is to automate the recognition of social behaviors in mice using pose estimation data derived from video recordings.

Understanding animal behavior is crucial for neuroscience and genetics. In this challenge, we deal with unlabelled video data converted into Parquet files. These files contain time-series data of keypoints (pose estimation) for multiple mice interacting in an arena.

### The Task:

Classify frame-by-frame behavior into specific categories (e.g., attack, investigation, mount, other).

### Data Description:

The dataset consists of Parquet files rather than raw video. This provides a structured representation of the mice's movements.

Input: Parquet files containing pose coordinates ($x, y$) for body parts (snout, ears, tail base, etc.) over time.

Format: Compressed columnar storage (Parquet) for efficient loading of large time-series.

Targets: Behavioral labels corresponding to specific time windows.

In this project the main idea that is curently implemented is a catboost model but even with the end of the competition this repository will still be updated as a form of learning in time series problems
