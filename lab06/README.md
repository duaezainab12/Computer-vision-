# Landmark Recognition Using SIFT

A computer vision application that recognizes famous landmarks from query images using the **Scale-Invariant Feature Transform (SIFT)**, Brute-Force matching with Lowe's ratio test, and RANSAC homography verification. It also localizes the recognized landmark inside the query image.

Built with Python and OpenCV 5.0 and run in Google Colab. The dataset (5 landmarks, 10 images each) is downloaded automatically from Wikimedia Commons.

## Features

- Reference database of 5 landmarks: Eiffel Tower, Taj Mahal, Colosseum, Statue of Liberty, Great Wall of China
- SIFT keypoints and 128-D descriptors (grayscale, CLAHE, RootSIFT)
- Brute-Force matcher (Euclidean / L2), kNN with k = 2, Lowe's ratio test (0.75)
- Landmark ranking and a rejection threshold for unknown images
- RANSAC homography and a polygon drawn around the landmark
- Automatic evaluation under 6 conditions, saved as a CSV table

## Project structure

```
.
├── Landmark_Recognition_SIFT.ipynb   # main notebook (all tasks)
├── landmark_sift.py                  # same pipeline as a standalone script
├── landmark_db/                      # reference images, one folder per landmark
├── unknown_images/                   # landmarks NOT in the database
├── images/                           # screenshots used in this README
└── results/                          # results_table.csv, descriptors.npz, figures
```

## How to run

**Colab:** upload the notebook and choose **Runtime → Run all**. The dataset downloads automatically.

**Local:**
```bash
pip install opencv-python numpy matplotlib pandas requests
python landmark_sift.py --eval --figs
python landmark_sift.py --query photo.jpg
```

## Pipeline

| Step | Description |
|---|---|
| 1. Database | 10 images per landmark, organized by folder |
| 2. Feature extraction | Grayscale, CLAHE, SIFT keypoints, 128-D RootSIFT descriptors |
| 3. Query processing | Same preprocessing and extraction for the unknown image |
| 4. Matching | BFMatcher (L2), kNN k=2, Lowe's ratio 0.75, one-to-one filtering |
| 5. Recognition | Rank landmarks by RANSAC inliers and matches, then apply thresholds |
| 6. Localization | `findHomography` with RANSAC, project reference corners, draw polygon |
| 7. Evaluation | Original, rotated, resized, dark, different viewpoint, unknown landmark |

### Decision rule

An image is not classified by the largest match count alone. A landmark is accepted only if:

1. Reliable matches ≥ 20 (`MIN_GOOD`)
2. Geometric verification passes: RANSAC inliers ≥ 12 (`MIN_INLIERS`), convex polygon of plausible size
3. The best landmark has at least 1.5× the inliers of the runner-up (`MARGIN`)

Otherwise the image is rejected as unknown, and a message is shown instead of a polygon.

## Results

### Database and keypoints

![Database overview](images/database_overview.png)

![Reference keypoints](images/reference_keypoints.png)

### Query processing

![Query processing](images/query_processing.png)

### Recognition example (Colosseum)

The ranking for the demo query was Colosseum (3409 matches, 3407 inliers), then Eiffel Tower, Great Wall and Statue of Liberty (9 each, 4 inliers) and Taj Mahal (8, 4 inliers).

![Demo result](images/demo_result.png)

### Experimental results table

Each landmark's first image is held out of the database and used as the query, with variants generated automatically. Unknown tests use 2 images each of the Pyramids of Giza and Sydney Opera House.

| Landmark | Condition | Predicted | Keypoints | Reliable Matches | RANSAC Inliers | Decision | Correct | Time (s) |
|---|---|---|---|---|---|---|---|---|
| Colosseum | Original | Colosseum | 4219 | 3409 | 3407 | Recognized | Yes | 22.916 |
| Colosseum | Rotated (30 deg) | Colosseum | 4742 | 2170 | 2164 | Recognized | Yes | 25.535 |
| Colosseum | Resized (50%) | Colosseum | 1302 | 875 | 873 | Recognized | Yes | 6.583 |
| Colosseum | Low brightness | Colosseum | 3750 | 2781 | 2774 | Recognized | Yes | 20.618 |
| Colosseum | Different viewpoint | Colosseum | 5021 | 2373 | 2356 | Recognized | Yes | 26.629 |
| Eiffel Tower | Original | Eiffel Tower | 2578 | 30 | 23 | Recognized | Yes | 13.667 |
| Eiffel Tower | Rotated (30 deg) | Eiffel Tower | 3174 | 26 | 21 | Recognized | Yes | 17.324 |
| Eiffel Tower | Resized (50%) | Unknown | 604 | 4 | 0 | Rejected | No | 3.022 |
| Eiffel Tower | Low brightness | Eiffel Tower | 2110 | 25 | 21 | Recognized | Yes | 11.161 |
| Eiffel Tower | Different viewpoint | Eiffel Tower | 3378 | 23 | 19 | Recognized | Yes | 18.019 |
| Great Wall of China | Original | Great Wall of China | 7007 | 138 | 86 | Recognized | Yes | 36.108 |
| Great Wall of China | Rotated (30 deg) | Great Wall of China | 7624 | 129 | 79 | Recognized | Yes | 39.161 |
| Great Wall of China | Resized (50%) | Great Wall of China | 1938 | 34 | 23 | Recognized | Yes | 10.071 |
| Great Wall of China | Low brightness | Great Wall of China | 6171 | 118 | 85 | Recognized | Yes | 31.942 |
| Great Wall of China | Different viewpoint | Great Wall of China | 8199 | 69 | 43 | Recognized | Yes | 41.795 |
| Statue of Liberty | Original | Unknown | 3546 | 25 | 10 | Rejected | No | 18.866 |
| Statue of Liberty | Rotated (30 deg) | Statue of Liberty | 3272 | 23 | 12 | Recognized | Yes | 17.569 |
| Statue of Liberty | Resized (50%) | Unknown | 1165 | 14 | 11 | Rejected | No | 5.817 |
| Statue of Liberty | Low brightness | Statue of Liberty | 2674 | 26 | 12 | Recognized | Yes | 13.919 |
| Statue of Liberty | Different viewpoint | Unknown | 4460 | 29 | 10 | Rejected | No | 23.432 |
| Taj Mahal | Original | Taj Mahal | 3672 | 2747 | 2745 | Recognized | Yes | 19.48 |
| Taj Mahal | Rotated (30 deg) | Taj Mahal | 3692 | 1664 | 1651 | Recognized | Yes | 19.88 |
| Taj Mahal | Resized (50%) | Taj Mahal | 1251 | 868 | 863 | Recognized | Yes | 6.228 |
| Taj Mahal | Low brightness | Taj Mahal | 2681 | 1943 | 1941 | Recognized | Yes | 14.575 |
| Taj Mahal | Different viewpoint | Taj Mahal | 4510 | 1973 | 1958 | Recognized | Yes | 23.817 |
| Pyramids of Giza | Unknown landmark | Unknown | 2086 | 4 | 0 | Rejected | Yes | 11.199 |
| Pyramids of Giza | Unknown landmark | Unknown | 3077 | 6 | 0 | Rejected | Yes | 16.133 |
| Sydney Opera House | Unknown landmark | Unknown | 3035 | 10 | 4 | Rejected | Yes | 16.627 |
| Sydney Opera House | Unknown landmark | Unknown | 1739 | 11 | 5 | Rejected | Yes | 8.874 |

### Summary by condition

| Condition | Accuracy | Tests | Mean Matches | Mean Inliers | Mean Time (s) |
|---|---|---|---|---|---|
| Original | 80% | 5 | 1269.8 | 1254.2 | 22.21 |
| Rotated (30 deg) | 100% | 5 | 802.4 | 785.4 | 23.89 |
| Resized (50%) | 60% | 5 | 359.0 | 354.0 | 6.34 |
| Low brightness | 100% | 5 | 978.6 | 966.6 | 18.44 |
| Different viewpoint | 80% | 5 | 893.4 | 877.2 | 26.74 |
| Unknown landmark | 100% | 4 | 7.75 | 2.25 | 13.21 |

### Summary by landmark (known landmarks only)

| Landmark | Correct | Tests | Accuracy |
|---|---|---|---|
| Colosseum | 5 | 5 | 100% |
| Eiffel Tower | 4 | 5 | 80% |
| Great Wall of China | 5 | 5 | 100% |
| Statue of Liberty | 2 | 5 | 40% |
| Taj Mahal | 5 | 5 | 100% |

### Overall

| Metric | Value |
|---|---|
| Overall accuracy | **86.2%** (25 / 29) |
| Known-landmark accuracy | 84.0% (21 / 25) |
| Unknown-landmark rejection | 100% (4 / 4) |
| Mean processing time | 18.65 s per query (Colab CPU, brute-force against 50 reference images) |

### Output screenshots

| Condition | Screenshot |
|---|---|
| Original | ![Original](images/result_original.png) |
| Rotated (30°) | ![Rotated](images/result_rotated.png) |
| Low brightness | ![Low brightness](images/result_low_brightness.png) |
| Different viewpoint | ![Different viewpoint](images/result_different_viewpoint.png) |
| Unknown landmark (rejected) | ![Unknown](images/result_unknown_landmark.png) |

## Discussion

**Performance.** Rotation and low brightness were handled perfectly (100%), which fits SIFT's rotation invariance and the CLAHE step. All 4 unknown images were rejected, with only 4 to 11 reliable matches and at most 5 inliers, well under the thresholds. Localization polygons were drawn only when geometric verification succeeded.

**Failures.** The 4 errors were all false rejections of correct landmarks, none were wrong labels:
- Eiffel Tower, resized 50%: only 4 reliable matches. The thin lattice structure loses most keypoints at half resolution (604 keypoints versus 2578 originally).
- Statue of Liberty (original, resized, different viewpoint): 14 to 29 reliable matches but only 10 or 11 RANSAC inliers, just under the threshold of 12. Its smooth surfaces give few distinctive keypoints.

**Caveat on the very high match counts.** Colosseum and Taj Mahal show 1,000 to 3,400 matches with about 99.9% inliers, which is far higher than the Eiffel Tower (about 25) or Statue of Liberty (about 25). The demo figure shows near-perfect horizontal match lines, so the database most likely contains a near-duplicate of the held-out photo (the same picture uploaded under several file names or resolutions). Their 100% accuracy is therefore probably optimistic. The Eiffel Tower and Statue of Liberty results, where the held-out image has no near-duplicate, are the more realistic measure of generalization to a new photo.

**Limitations.**
- Few-keypoint structures (lattice towers, smooth statues) are fragile under downscaling and viewpoint change.
- A fixed inlier threshold of 12 rejects some true matches, and lowering it would risk accepting unknown images (unknown images reached at most 5 inliers, so a threshold of about 8 to 10 would likely be safe on this data).
- Brute-force matching against every reference image takes 6 to 42 s per query on Colab CPU and will not scale. FLANN or a vocabulary tree would be faster.
- Only 4 unknown images were tested, and the Wikimedia photos vary in quality.

## Image credits

The dataset is downloaded from [Wikimedia Commons](https://commons.wikimedia.org/). Individual images carry their own licenses (typically CC BY-SA or public domain), so check them before redistributing.

## References

- D. G. Lowe, "Distinctive Image Features from Scale-Invariant Keypoints," *International Journal of Computer Vision*, 60(2), 2004.
- R. Arandjelović and A. Zisserman, "Three things everyone should know to improve object retrieval," CVPR, 2012 (RootSIFT).
- [OpenCV documentation: feature detection and description](https://docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html)
