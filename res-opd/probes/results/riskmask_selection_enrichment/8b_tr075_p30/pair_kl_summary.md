# Same-Image RKL Signal Probe

## Inputs

- trace_jsonl: `res-opd/probes/results/riskmask_selection_enrichment/8b_tr075_p30/pair_kl_trace.jsonl`
- records: 990
- tokens: 582153
- object_mentions: 16446
- labeled_object_mentions: 16435
- base_hallucination_rate: 0.1614

## Object Mention Summary

| group | count | RKL mean | RKL p75 | FKL mean | JSD mean | student NLL | student entropy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| correct_object | 13782 | 0.0292 | 0.0177 | 0.0294 | 0.0065 | 0.2500 | 0.4715 |
| hallucinated_object | 2653 | 0.0501 | 0.0320 | 0.0505 | 0.0110 | 0.3870 | 0.6609 |
| unknown_object | 11 | 0.0180 | 0.0211 | 0.0173 | 0.0043 | 0.3020 | 0.5573 |

## High-Divergence Buckets

| metric | top frac | selected | halluc | correct | precision | recall | correct FPR | lift | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rkl_student_to_teacher_mean | 0.10 | 1644 | 398 | 1246 | 0.2421 | 0.1500 | 0.0904 | 1.4997 | 0.0723 |
| rkl_student_to_teacher_mean | 0.20 | 3287 | 716 | 2571 | 0.2178 | 0.2699 | 0.1865 | 1.3494 | 0.0290 |
| rkl_student_to_teacher_mean | 0.25 | 4109 | 864 | 3245 | 0.2103 | 0.3257 | 0.2355 | 1.3026 | 0.0199 |
| rkl_student_to_teacher_max | 0.10 | 1644 | 378 | 1266 | 0.2299 | 0.1425 | 0.0919 | 1.4244 | 0.0801 |
| rkl_student_to_teacher_max | 0.20 | 3287 | 696 | 2591 | 0.2117 | 0.2623 | 0.1880 | 1.3117 | 0.0312 |
| rkl_student_to_teacher_max | 0.25 | 4109 | 838 | 3271 | 0.2039 | 0.3159 | 0.2373 | 1.2634 | 0.0224 |
| fkl_teacher_to_student_mean | 0.10 | 1644 | 396 | 1248 | 0.2409 | 0.1493 | 0.0906 | 1.4922 | 0.0733 |
| fkl_teacher_to_student_mean | 0.20 | 3287 | 711 | 2576 | 0.2163 | 0.2680 | 0.1869 | 1.3400 | 0.0296 |
| fkl_teacher_to_student_mean | 0.25 | 4109 | 861 | 3248 | 0.2095 | 0.3245 | 0.2357 | 1.2981 | 0.0198 |
| fkl_teacher_to_student_max | 0.10 | 1644 | 376 | 1268 | 0.2287 | 0.1417 | 0.0920 | 1.4168 | 0.0809 |
| fkl_teacher_to_student_max | 0.20 | 3287 | 692 | 2595 | 0.2105 | 0.2608 | 0.1883 | 1.3042 | 0.0315 |
| fkl_teacher_to_student_max | 0.25 | 4109 | 833 | 3276 | 0.2027 | 0.3140 | 0.2377 | 1.2559 | 0.0223 |
| jsd_mean | 0.10 | 1644 | 391 | 1253 | 0.2378 | 0.1474 | 0.0909 | 1.4734 | 0.0176 |
| jsd_mean | 0.20 | 3287 | 712 | 2575 | 0.2166 | 0.2684 | 0.1868 | 1.3419 | 0.0072 |
| jsd_mean | 0.25 | 4109 | 865 | 3244 | 0.2105 | 0.3260 | 0.2354 | 1.3041 | 0.0049 |
| jsd_max | 0.10 | 1644 | 378 | 1266 | 0.2299 | 0.1425 | 0.0919 | 1.4244 | 0.0195 |
| jsd_max | 0.20 | 3287 | 694 | 2593 | 0.2111 | 0.2616 | 0.1881 | 1.3080 | 0.0077 |
| jsd_max | 0.25 | 4109 | 838 | 3271 | 0.2039 | 0.3159 | 0.2373 | 1.2634 | 0.0055 |
| student_nll_mean | 0.10 | 1644 | 442 | 1202 | 0.2689 | 0.1666 | 0.0872 | 1.6655 | 0.7947 |
| student_nll_mean | 0.20 | 3287 | 743 | 2544 | 0.2260 | 0.2801 | 0.1846 | 1.4003 | 0.4634 |
| student_nll_mean | 0.25 | 4109 | 895 | 3214 | 0.2178 | 0.3374 | 0.2332 | 1.3493 | 0.3577 |
| student_entropy_mean | 0.10 | 1644 | 442 | 1202 | 0.2689 | 0.1666 | 0.0872 | 1.6655 | 1.2912 |
| student_entropy_mean | 0.20 | 3287 | 810 | 2477 | 0.2464 | 0.3053 | 0.1797 | 1.5266 | 0.9161 |
| student_entropy_mean | 0.25 | 4109 | 966 | 3143 | 0.2351 | 0.3641 | 0.2281 | 1.4564 | 0.7828 |

## Top High-RKL Mentions

| image | type | object | mention | RKL | JSD | context |
| --- | --- | --- | --- | --- | --- | --- |
| 1532 | correct_object | car | Jeep | 8.9988 | 0.6906 | ...plate is partially visible, showing “9-3227”.   - In the center, a black SUV (a Jeep Grand Cherokee, identifiable by the “JEEP” logo and “GRAND CHEROKEE” tex |
| 157767 | correct_object | cup | cup | 4.2619 | 0.4061 | ...acket and dark trousers stands directly in front of the table, holding a yellow cup or glass in his right hand and a small plate in his left. He is looking s |
| 566758 | correct_object | bus | bus | 3.8986 | 0.5201 | ...there is a red and black decal that reads “FRESH AIR” in white letters.   - The bus has a “School Bus” emblem on the front, just above the grille. - **Window |
| 88432 | hallucinated_object | bus | BUS | 3.5727 | 0.2005 | ...- A red and white “NO PARKING ANY TIME” sign.   - Below that, a blue and white “BUS STOP” sign with a white bus icon.   - A smaller, partially obscured sign  |
| 1532 | correct_object | car | JEEP | 3.4360 | 0.2697 | ...7”.   - In the center, a black SUV (a Jeep Grand Cherokee, identifiable by the “JEEP” logo and “GRAND CHEROKEE” text on the rear) is directly ahead. A spare  |
| 280710 | correct_object | bus | BUS | 3.1858 | 0.3548 | ...hind a black metal fence.   - The bus has large, bold yellow text on its side: “BUS COMPANY” and “SIGHTSEEING TOUR OF LONDON”.   - The number “1” is visible  |
| 1532 | correct_object | car | SUV | 3.0971 | 0.2357 | ...icense plate is partially visible, showing “9-3227”.   - In the center, a black SUV (a Jeep Grand Cherokee, identifiable by the “JEEP” logo and “GRAND CHEROK |
| 104803 | hallucinated_object | chair | seat | 2.8966 | 0.4466 | ...rk environment, likely a bathroom. The most striking feature is that the toilet seat and lid are covered in a thick, glittery, silver or chrome-like material |
| 251119 | hallucinated_object | orange | Orange | 2.8444 | 0.3285 | ...n a blue background. The cup is sealed with a clear plastic lid. - **Right Cup (Orange Juice):** This cup has a label with a yellow and orange color scheme.  |
| 378139 | correct_object | car | car | 2.6823 | 0.2567 | ..., and signs of urban life, such as bicycles parked along the sidewalk and a red car parked near the curb.  **Trees and Vegetation:** Green trees are interspe |
| 181499 | correct_object | laptop | computer | 1.9876 | 0.3262 | ...ext on the screen is partially legible and reads: "Please do not power off your computer until the installation is complete." Below this, the "Windows" logo  |
| 126110 | hallucinated_object | car | van | 1.9581 | 0.2813 | ...resolution, grainy photograph taken from the side of a yellow vehicle, likely a van or small truck, focusing on its driver’s side window. The image quality i |
| 473869 | correct_object | spoon | spoon | 1.8599 | 0.3973 | ...ight hand holds a small, patterned white bowl with a red rim, and he is using a spoon to scoop something from it, likely a sauce or topping, toward a dish in |
| 271997 | correct_object | person | boy | 1.8344 | 0.3384 | This is a vintage, full-face portrait of a young boy, likely taken in a studio setting for a school or formal photo. The image has t... |
| 465675 | correct_object | boat | motorboat | 1.6166 | 0.3042 | ...nted photograph.  **Main Subjects:** The central focus is a small, dark-colored motorboat and a vintage tractor parked side-by-side on the beach.  - **The Bo |
| 226171 | hallucinated_object | traffic light | streetlights | 1.5743 | 0.2004 | ... large featured image at the top showing a nighttime street scene with cars and streetlights. Below this, there are several smaller thumbnail images arranged |
| 124636 | hallucinated_object | person | Mother | 1.5600 | 0.3335 | ...ng distinct shadows on the ground.  **Main Subjects:**  1.  **The Standing Cow (Mother):**     *   **Appearance:** This is a large, robust cow with a predomi |
| 345252 | correct_object | tv | monitor | 1.5029 | 0.3255 | ...r or RPG, with green and yellowish elements visible in the dark background. The monitor is positioned directly in front of the user. - **Keyboard and Mouse:* |
| 12576 | correct_object | pizza | pizza | 1.5001 | 0.3456 | ... a large, open pizza box in the immediate foreground, containing a whole, uncut pizza. The pizza has a golden-brown, slightly charred crust and is covered in |
| 143572 | hallucinated_object | car | car | 1.4983 | 0.3606 | ... and text:   - On the left side, the word “esurance” is visible in white, with “car insurance” written in smaller text below it.   - In the center, the word  |
